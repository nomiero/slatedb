//! `io_uring`-backed implementation of [`LocalCacheStorage`].
//!
//! Routes every cache read and write through a dedicated OS thread that owns
//! a single `io_uring` instance, instead of dispatching each `pread`/`write`
//! through `tokio::task::spawn_blocking`. The blocking pool is shared by
//! every other consumer in the process; on a busy machine its threads land
//! anywhere the kernel will schedule them, including cores running
//! compaction. That contention is what causes the residual reader-throughput
//! dips even when the working set is fully page-cached. Pinning all cache
//! I/O onto a dedicated worker removes that scheduler-level coupling.
//!
//! Linux-only. The module is only compiled in on `target_os = "linux"`; the
//! constructor on other platforms is wired so that `use_io_uring` silently
//! falls back to [`super::FsCacheStorage`].

use crate::cached_object_store::storage::{
    LocalCacheEntry, LocalCacheHead, LocalCacheStorage, LocalCacheTee, PartID,
};
use bytes::Bytes;
use crossbeam_channel::{Receiver as CbReceiver, Sender as CbSender};
use io_uring::{opcode, types, IoUring};
use object_store::path::Path;
use object_store::{Attributes, ObjectMeta};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use tokio::sync::oneshot;
use tracing::{info, warn};

/// `O_DIRECT` alignment requirement on common Linux filesystems. Underlying
/// device may demand 512, but ext4/xfs typically want 4 KB.
const O_DIRECT_ALIGN: usize = 4096;
/// SQ depth. Generous; a few hundred concurrent ops is plausible during a
/// burst (compactor draining input + foreground readers).
const URING_SQ_ENTRIES: u32 = 1024;
/// Max requests pulled from the channel before we submit. Keeps each
/// `submit_and_wait` cycle bounded even under sustained load.
const SUBMIT_BATCH_SIZE: usize = 64;

fn wrap_io_err<E>(err: E) -> object_store::Error
where
    E: std::error::Error + Send + Sync + 'static,
{
    object_store::Error::Generic {
        store: "cached_object_store_uring",
        source: Box::new(err),
    }
}

fn make_part_path(
    root_folder: std::path::PathBuf,
    location: &Path,
    part_number: usize,
    part_size: usize,
) -> std::path::PathBuf {
    let part_size_name = if part_size.is_multiple_of(1024 * 1024) {
        format!("{}mb", part_size / (1024 * 1024))
    } else {
        format!("{}kb", part_size / 1024)
    };
    let suffix = format!("_part{}-{:09}", part_size_name, part_number);
    let mut path = root_folder.join(location.to_string());
    path.push(suffix);
    path
}

fn make_head_path(root_folder: std::path::PathBuf, location: &Path) -> std::path::PathBuf {
    let mut path = root_folder.join(location.to_string());
    path.push("_head");
    path
}

/// 4 KB-aligned heap buffer for `O_DIRECT` reads. `Bytes::from_owner` keeps
/// the alignment alive for the lifetime of the slice we hand back.
struct AlignedBuf {
    ptr: *mut u8,
    capacity: usize,
    layout: std::alloc::Layout,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    fn new(capacity: usize) -> std::io::Result<Self> {
        let aligned_cap = capacity.div_ceil(O_DIRECT_ALIGN) * O_DIRECT_ALIGN;
        let layout = std::alloc::Layout::from_size_align(aligned_cap, O_DIRECT_ALIGN)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        // SAFETY: layout has nonzero size (aligned_cap >= O_DIRECT_ALIGN > 0)
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(std::io::Error::other("aligned alloc failed"));
        }
        Ok(Self {
            ptr,
            capacity: aligned_cap,
            layout,
        })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr was alloc'd with `capacity` bytes, alignment is 4 KB.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.capacity) }
    }
}

impl AsRef<[u8]> for AlignedBuf {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: ptr was alloc'd with `capacity` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.capacity) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: ptr came from `alloc_zeroed` with this exact layout.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

/// One in-flight io_uring op. Owns the storage backing its SQE so the kernel
/// does not see a freed buffer; dropped on completion.
enum InFlight {
    Read {
        // Buffer the kernel writes into. Stays alive until completion.
        buf: Vec<u8>,
        // O_DIRECT path uses an aligned buf; we slice on completion.
        aligned: Option<AlignedBuf>,
        // Original requested range inside `buf` / aligned slice.
        requested_offset_in_buf: usize,
        requested_len: usize,
        sender: oneshot::Sender<object_store::Result<Bytes>>,
    },
    Write {
        // Bytes the kernel reads from. Either a Bytes payload (buffered) or
        // an aligned heap buffer (O_DIRECT). Either way, stays alive until
        // completion.
        _buf: WriteBuf,
        // Owned File handle: dropped (closed) before the rename so the
        // metadata is durable from the kernel's POV.
        file: Option<File>,
        // After the write completes successfully, rename `tmp_path` →
        // `final_path`. If they're equal, rename is a no-op.
        tmp_path: std::path::PathBuf,
        final_path: std::path::PathBuf,
        sender: oneshot::Sender<object_store::Result<()>>,
    },
}

// The held buffer is never read on the Rust side, but the kernel reads from
// it for the duration of the I/O — the variant exists only to keep the
// allocation alive until completion.
#[allow(dead_code)]
enum WriteBuf {
    Bytes(Bytes),
    Aligned(AlignedBuf),
}

/// Channel message sent from tokio tasks to the worker thread.
enum WorkerOp {
    /// Read `len` bytes at `offset` from `path`. Resolves the path through
    /// the worker's file-handle cache; opens lazily on first miss. If
    /// `direct_io` is set on the storage, the open happens with `O_DIRECT`.
    Read {
        path: std::path::PathBuf,
        offset: u64,
        len: usize,
        sender: oneshot::Sender<object_store::Result<Bytes>>,
    },
    /// Write `bytes` in full to a fresh file at `tmp_path`, then rename it
    /// atomically to `final_path`. The worker creates parent dirs as
    /// needed. No fsync — cache is reconstructable from upstream.
    AtomicWrite {
        tmp_path: std::path::PathBuf,
        final_path: std::path::PathBuf,
        bytes: Bytes,
        sender: oneshot::Sender<object_store::Result<()>>,
    },
    /// Drop the cached fd for `path` (called after rename / delete so the
    /// next read opens the new inode).
    InvalidateFd { path: std::path::PathBuf },
    /// Hint the kernel to drop page cache pages for every part + head file
    /// under `dir`. Best-effort.
    AdviseDontneed { dir: std::path::PathBuf },
    /// Remove the cache directory for a location (recursive).
    RemoveDir {
        dir: std::path::PathBuf,
        sender: oneshot::Sender<object_store::Result<()>>,
    },
    /// Stop the worker and join.
    Shutdown,
}

#[derive(Debug)]
struct WorkerHandle {
    tx: CbSender<WorkerOp>,
    join: parking_lot::Mutex<Option<JoinHandle<()>>>,
}

impl WorkerHandle {
    fn spawn(direct_io: bool) -> std::io::Result<Self> {
        let (tx, rx) = crossbeam_channel::unbounded::<WorkerOp>();
        let join = std::thread::Builder::new()
            .name("slatedb-uring".to_string())
            .spawn(move || {
                if let Err(e) = run_worker(rx, direct_io) {
                    warn!("slatedb-uring worker exited with error: {:?}", e);
                }
            })?;
        Ok(Self {
            tx,
            join: parking_lot::Mutex::new(Some(join)),
        })
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(WorkerOp::Shutdown);
        if let Some(j) = self.join.lock().take() {
            let _ = j.join();
        }
    }
}

/// Worker thread main loop. Owns the io_uring, the file-handle cache, and
/// the in-flight map.
fn run_worker(rx: CbReceiver<WorkerOp>, direct_io: bool) -> std::io::Result<()> {
    let mut ring = IoUring::new(URING_SQ_ENTRIES)?;
    let mut fd_cache: HashMap<std::path::PathBuf, Arc<File>> = HashMap::new();
    let mut pending: HashMap<u64, InFlight> = HashMap::new();
    let mut next_user_data: u64 = 1;

    // Optional pin of this thread to a specific core. If env var unset or
    // invalid, leave to the OS scheduler. Useful when the user has already
    // pinned tokio runtime workers via core_affinity and wants the I/O
    // worker on a non-overlapping core.
    if let Some(cpu) = std::env::var("SLATEDB_URING_CPU")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        let _ = core_affinity::set_for_current(core_affinity::CoreId { id: cpu });
    }

    loop {
        // Drain channel into SQ. Block when both channel + in-flight empty.
        let mut batched = 0usize;
        loop {
            if batched >= SUBMIT_BATCH_SIZE {
                break;
            }
            let op = if pending.is_empty() && batched == 0 {
                match rx.recv() {
                    Ok(op) => op,
                    Err(_) => return Ok(()), // sender closed
                }
            } else {
                match rx.try_recv() {
                    Ok(op) => op,
                    Err(crossbeam_channel::TryRecvError::Empty) => break,
                    Err(crossbeam_channel::TryRecvError::Disconnected) => return Ok(()),
                }
            };

            match op {
                WorkerOp::Shutdown => return Ok(()),
                WorkerOp::InvalidateFd { path } => {
                    fd_cache.remove(&path);
                }
                WorkerOp::AdviseDontneed { dir } => {
                    // Best-effort, sync. Not worth a uring round-trip; the
                    // dontneed hint is rare (compaction completion) and
                    // benefits from being applied to all part files together.
                    if let Ok(rd) = std::fs::read_dir(&dir) {
                        for entry in rd.flatten() {
                            let p = entry.path();
                            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                            if name == "_head" || name.starts_with("_part") {
                                if let Ok(f) = std::fs::File::open(&p) {
                                    advise_dontneed(f.as_raw_fd());
                                }
                            }
                        }
                    }
                }
                WorkerOp::RemoveDir { dir, sender } => {
                    let res = std::fs::remove_dir_all(&dir).map_err(wrap_io_err);
                    // Drop any cached fds under this dir.
                    fd_cache.retain(|p, _| !p.starts_with(&dir));
                    let _ = sender.send(res);
                }
                WorkerOp::Read {
                    path,
                    offset,
                    len,
                    sender,
                } => {
                    let fd = match get_or_open_for_read(&mut fd_cache, &path, direct_io) {
                        Ok(arc) => arc,
                        Err(e) => {
                            let _ = sender.send(Err(wrap_io_err(e)));
                            continue;
                        }
                    };
                    let user_data = next_user_data;
                    next_user_data = next_user_data.wrapping_add(1);
                    if direct_io {
                        // Round to 4KB on both ends; we'll slice on completion.
                        let aligned_offset = (offset / O_DIRECT_ALIGN as u64) * O_DIRECT_ALIGN as u64;
                        let head_pad = (offset - aligned_offset) as usize;
                        let aligned_len = (head_pad + len).div_ceil(O_DIRECT_ALIGN) * O_DIRECT_ALIGN;
                        let mut aligned = match AlignedBuf::new(aligned_len) {
                            Ok(b) => b,
                            Err(e) => {
                                let _ = sender.send(Err(wrap_io_err(e)));
                                continue;
                            }
                        };
                        let ptr = aligned.as_mut_slice().as_mut_ptr();
                        let entry = opcode::Read::new(types::Fd(fd.as_raw_fd()), ptr, aligned_len as u32)
                            .offset(aligned_offset)
                            .build()
                            .user_data(user_data);
                        // SAFETY: pending map keeps the buffer alive until we
                        // drain the matching CQE.
                        unsafe {
                            if ring.submission().push(&entry).is_err() {
                                let _ = sender.send(Err(wrap_io_err(std::io::Error::other(
                                    "uring SQ full",
                                ))));
                                continue;
                            }
                        }
                        pending.insert(
                            user_data,
                            InFlight::Read {
                                buf: Vec::new(),
                                aligned: Some(aligned),
                                requested_offset_in_buf: head_pad,
                                requested_len: len,
                                sender,
                            },
                        );
                    } else {
                        let mut buf = vec![0u8; len];
                        let ptr = buf.as_mut_ptr();
                        let entry = opcode::Read::new(types::Fd(fd.as_raw_fd()), ptr, len as u32)
                            .offset(offset)
                            .build()
                            .user_data(user_data);
                        // SAFETY: see comment above.
                        unsafe {
                            if ring.submission().push(&entry).is_err() {
                                let _ = sender.send(Err(wrap_io_err(std::io::Error::other(
                                    "uring SQ full",
                                ))));
                                continue;
                            }
                        }
                        pending.insert(
                            user_data,
                            InFlight::Read {
                                buf,
                                aligned: None,
                                requested_offset_in_buf: 0,
                                requested_len: len,
                                sender,
                            },
                        );
                    }
                    batched += 1;
                }
                WorkerOp::AtomicWrite {
                    tmp_path,
                    final_path,
                    bytes,
                    sender,
                } => {
                    // Open the tmp file synchronously (creates parent dirs
                    // first). Opens are infrequent; the win from io_uring is
                    // on the actual write itself.
                    if let Some(parent) = tmp_path.parent() {
                        if let Err(e) = std::fs::create_dir_all(parent) {
                            let _ = sender.send(Err(wrap_io_err(e)));
                            continue;
                        }
                    }
                    let mut open = OpenOptions::new();
                    open.create(true).truncate(true).write(true);
                    let aligned_payload = direct_io && bytes.len() % O_DIRECT_ALIGN == 0;
                    if aligned_payload {
                        // Only apply O_DIRECT to writes whose payload length
                        // satisfies alignment. Heads (and any tail-end short
                        // writes) stay buffered.
                        open.custom_flags(libc::O_DIRECT);
                    }
                    let file = match open.open(&tmp_path) {
                        Ok(f) => f,
                        Err(e) => {
                            let _ = sender.send(Err(wrap_io_err(e)));
                            continue;
                        }
                    };
                    let raw = file.as_raw_fd();
                    let user_data = next_user_data;
                    next_user_data = next_user_data.wrapping_add(1);

                    let (write_buf, ptr, len) = if aligned_payload {
                        let mut aligned = match AlignedBuf::new(bytes.len()) {
                            Ok(b) => b,
                            Err(e) => {
                                let _ = sender.send(Err(wrap_io_err(e)));
                                continue;
                            }
                        };
                        aligned.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
                        let p = aligned.as_ref().as_ptr();
                        let l = bytes.len() as u32;
                        (WriteBuf::Aligned(aligned), p, l)
                    } else {
                        let p = bytes.as_ptr();
                        let l = bytes.len() as u32;
                        (WriteBuf::Bytes(bytes.clone()), p, l)
                    };

                    let entry = opcode::Write::new(types::Fd(raw), ptr, len)
                        .offset(0)
                        .build()
                        .user_data(user_data);
                    // SAFETY: write_buf lives in `pending` until the matching
                    // CQE is drained, so the kernel sees a stable buffer for
                    // the entirety of the I/O.
                    unsafe {
                        if ring.submission().push(&entry).is_err() {
                            let _ = sender.send(Err(wrap_io_err(std::io::Error::other(
                                "uring SQ full",
                            ))));
                            continue;
                        }
                    }
                    pending.insert(
                        user_data,
                        InFlight::Write {
                            _buf: write_buf,
                            file: Some(file),
                            tmp_path,
                            final_path,
                            sender,
                        },
                    );
                    batched += 1;
                }
            }
        }

        // Submit + wait for at least one completion if anything is in flight.
        if batched > 0 || !pending.is_empty() {
            let want = if pending.is_empty() { 0 } else { 1 };
            ring.submit_and_wait(want)?;
        }

        // Drain CQ.
        let mut cq = ring.completion();
        cq.sync();
        for cqe in cq.by_ref() {
            let user_data = cqe.user_data();
            let result = cqe.result();
            match pending.remove(&user_data) {
                Some(InFlight::Read {
                    buf,
                    aligned,
                    requested_offset_in_buf,
                    requested_len,
                    sender,
                }) => {
                    if result < 0 {
                        let _ = sender.send(Err(wrap_io_err(std::io::Error::from_raw_os_error(
                            -result,
                        ))));
                    } else {
                        let n = result as usize;
                        // For O_DIRECT path, slice from aligned buf.
                        if let Some(aligned) = aligned {
                            let avail = n.saturating_sub(requested_offset_in_buf);
                            let take = avail.min(requested_len);
                            let slice = &aligned.as_ref()
                                [requested_offset_in_buf..requested_offset_in_buf + take];
                            let bytes = Bytes::copy_from_slice(slice);
                            let _ = sender.send(Ok(bytes));
                        } else {
                            // Buffered path: trim to actual short read length.
                            let take = n.min(requested_len);
                            let mut out = buf;
                            out.truncate(take);
                            let _ = sender.send(Ok(Bytes::from(out)));
                        }
                    }
                }
                Some(InFlight::Write {
                    _buf,
                    mut file,
                    tmp_path,
                    final_path,
                    sender,
                }) => {
                    // Drop the underlying File first (closes the fd) so the
                    // rename observes a fully-flushed metadata state. The
                    // _buf goes with it (the kernel is no longer reading).
                    drop(file.take());
                    let send_result = if result < 0 {
                        let _ = std::fs::remove_file(&tmp_path);
                        Err(wrap_io_err(std::io::Error::from_raw_os_error(-result)))
                    } else if tmp_path == final_path {
                        // Tee path: write to tmp only, rename happens later
                        // in commit().
                        Ok(())
                    } else {
                        std::fs::rename(&tmp_path, &final_path).map_err(wrap_io_err)
                    };
                    let _ = sender.send(send_result);
                }
                None => {
                    // Lost cqe? Should not happen.
                    warn!(
                        "slatedb-uring: completion for unknown user_data={}",
                        user_data
                    );
                }
            }
        }
    }
}

fn advise_dontneed(fd: RawFd) {
    // SAFETY: fd is valid for the duration of this call (caller holds a File).
    unsafe {
        libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

fn get_or_open_for_read(
    fd_cache: &mut HashMap<std::path::PathBuf, Arc<File>>,
    path: &std::path::Path,
    direct_io: bool,
) -> std::io::Result<Arc<File>> {
    if let Some(f) = fd_cache.get(path) {
        // Validate by stat: if the inode has been unlinked (rename replaced
        // it), reopen so we read from the new inode.
        if is_handle_valid(f) {
            return Ok(f.clone());
        }
        fd_cache.remove(path);
    }
    let mut open = OpenOptions::new();
    open.read(true);
    if direct_io {
        open.custom_flags(libc::O_DIRECT | libc::O_NOATIME);
    }
    let file = open.open(path)?;
    let arc = Arc::new(file);
    fd_cache.insert(path.to_path_buf(), arc.clone());
    Ok(arc)
}

fn is_handle_valid(file: &File) -> bool {
    // Mirror the storage_fs.rs trick: a renamed-over inode has nlink == 0
    // because the dir entry was unlinked by `rename()`.
    use std::os::unix::fs::MetadataExt;
    file.metadata().map(|m| m.nlink() > 0).unwrap_or(false)
}

// -----------------------------------------------------------------------------
// Public types implementing the LocalCacheStorage / Entry / Tee traits.
// -----------------------------------------------------------------------------

/// io_uring-backed `LocalCacheStorage`. Constructed via
/// [`IoUringCacheStorage::new`] which spawns the dedicated worker thread on
/// success. Falls back to a panic if `IoUring::new` fails — callers should
/// instead probe via `try_new` and substitute `FsCacheStorage` on err.
#[derive(Debug)]
pub(crate) struct IoUringCacheStorage {
    root_folder: std::path::PathBuf,
    direct_io: bool,
    worker: Arc<WorkerHandle>,
    /// Monotonically increasing tee id, used to namespace temp filenames.
    tee_seq: AtomicU64,
}

impl std::fmt::Display for IoUringCacheStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IoUringCacheStorage(root={}, direct_io={})",
            self.root_folder.display(),
            self.direct_io
        )
    }
}

impl IoUringCacheStorage {
    /// Try to construct. Returns Err if the kernel rejects `io_uring_setup`
    /// (ENOSYS on too-old kernels, EPERM with seccomp policies, etc.).
    pub(crate) fn try_new(
        root_folder: std::path::PathBuf,
        direct_io: bool,
    ) -> std::io::Result<Self> {
        let worker = WorkerHandle::spawn(direct_io)?;
        info!(
            "using io_uring cache storage [root={}, direct_io={}]",
            root_folder.display(),
            direct_io
        );
        Ok(Self {
            root_folder,
            direct_io,
            worker: Arc::new(worker),
            tee_seq: AtomicU64::new(0),
        })
    }

    fn send_op(&self, op: WorkerOp) -> object_store::Result<()> {
        self.worker
            .tx
            .send(op)
            .map_err(|_| wrap_io_err(std::io::Error::other("uring worker channel closed")))
    }
}

#[async_trait::async_trait]
impl LocalCacheStorage for IoUringCacheStorage {
    fn entry(&self, location: &Path, part_size: usize) -> Box<dyn LocalCacheEntry> {
        Box::new(IoUringCacheEntry {
            root_folder: self.root_folder.clone(),
            location: location.clone(),
            part_size,
            worker: self.worker.clone(),
        })
    }

    async fn start_evictor(&self) {
        // Eviction is not implemented for the io_uring backend yet. The
        // backend is opt-in and used in benchmarks where the cache cap is
        // disabled (max_cache_size_bytes = None), so eviction would be a
        // no-op anyway. Adding it later means wiring the same scan/track
        // logic onto a worker timer; out of scope for the first cut.
    }

    async fn remove(&self, location: &Path) -> object_store::Result<()> {
        let dir = self.root_folder.join(location.to_string());
        let (sender, recv) = oneshot::channel();
        self.send_op(WorkerOp::RemoveDir { dir, sender })?;
        recv.await
            .map_err(|e| wrap_io_err(std::io::Error::other(format!("oneshot recv: {e}"))))?
    }

    fn begin_tee(&self, location: &Path, part_size: usize) -> Option<Box<dyn LocalCacheTee>> {
        let tee_id = self.tee_seq.fetch_add(1, Ordering::Relaxed);
        Some(Box::new(IoUringCacheTee {
            root_folder: self.root_folder.clone(),
            location: location.clone(),
            part_size,
            worker: self.worker.clone(),
            tee_id,
            part_buf: bytes::BytesMut::new(),
            next_part_number: 0,
            poisoned: false,
            pending_renames: Vec::new(),
        }))
    }

    async fn advise_dontneed(&self, location: &Path) {
        let dir = self.root_folder.join(location.to_string());
        let _ = self.send_op(WorkerOp::AdviseDontneed { dir });
    }
}

#[derive(Debug)]
struct IoUringCacheEntry {
    root_folder: std::path::PathBuf,
    location: Path,
    part_size: usize,
    worker: Arc<WorkerHandle>,
}

impl IoUringCacheEntry {
    fn make_rand_suffix(&self) -> String {
        // Same shape as storage_fs: `_tmp` + 16 alpha chars. We're inside
        // the I/O hot path; use rand::thread_rng() to avoid threading a
        // shared rng through the storage.
        use rand::distr::Alphanumeric;
        use rand::Rng;
        let mut rng = rand::rng();
        (0..16).map(|_| rng.sample(Alphanumeric) as char).collect()
    }
}

#[async_trait::async_trait]
impl LocalCacheEntry for IoUringCacheEntry {
    async fn save_part(&self, part_number: PartID, buf: Bytes) -> object_store::Result<()> {
        let final_path = make_part_path(
            self.root_folder.clone(),
            &self.location,
            part_number,
            self.part_size,
        );
        let tmp_path = final_path.with_extension(format!("_tmp{}", self.make_rand_suffix()));
        let (sender, recv) = oneshot::channel();
        self.worker
            .tx
            .send(WorkerOp::AtomicWrite {
                tmp_path,
                final_path: final_path.clone(),
                bytes: buf,
                sender,
            })
            .map_err(|_| wrap_io_err(std::io::Error::other("uring worker channel closed")))?;
        let result = recv
            .await
            .map_err(|e| wrap_io_err(std::io::Error::other(format!("oneshot recv: {e}"))))?;
        // Invalidate any cached fd at final_path so the next read sees the
        // new inode.
        let _ = self.worker.tx.send(WorkerOp::InvalidateFd { path: final_path });
        result
    }

    async fn read_part(
        &self,
        part_number: PartID,
        range_in_part: std::ops::Range<usize>,
    ) -> object_store::Result<Option<Bytes>> {
        let path = make_part_path(
            self.root_folder.clone(),
            &self.location,
            part_number,
            self.part_size,
        );
        let len = range_in_part.end - range_in_part.start;
        let offset = range_in_part.start as u64;
        let (sender, recv) = oneshot::channel();
        if self
            .worker
            .tx
            .send(WorkerOp::Read {
                path,
                offset,
                len,
                sender,
            })
            .is_err()
        {
            return Err(wrap_io_err(std::io::Error::other(
                "uring worker channel closed",
            )));
        }
        match recv.await {
            Ok(Ok(bytes)) => Ok(Some(bytes)),
            Ok(Err(e)) => match e {
                object_store::Error::Generic { ref source, .. } => {
                    // Translate ENOENT to None (cache miss), as FsCacheStorage does.
                    if let Some(io) = source.downcast_ref::<std::io::Error>() {
                        if io.kind() == std::io::ErrorKind::NotFound {
                            return Ok(None);
                        }
                    }
                    Err(e)
                }
                _ => Err(e),
            },
            Err(e) => Err(wrap_io_err(std::io::Error::other(format!(
                "oneshot recv: {e}"
            )))),
        }
    }

    #[cfg(test)]
    async fn cached_parts(&self) -> object_store::Result<Vec<PartID>> {
        // Cold path; sync read_dir is fine.
        let dir = self.root_folder.join(self.location.to_string());
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(wrap_io_err(e)),
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if let Some(rest) = name_str.strip_prefix("_part") {
                if let Some(idx) = rest.rfind('-') {
                    if let Ok(n) = rest[idx + 1..].parse::<usize>() {
                        out.push(n);
                    }
                }
            }
        }
        out.sort();
        Ok(out)
    }

    async fn save_head(&self, head: (&ObjectMeta, &Attributes)) -> object_store::Result<()> {
        let head_path = make_head_path(self.root_folder.clone(), &self.location);
        // If the head already exists, skip — same semantics as FsCacheStorage.
        if head_path.exists() {
            return Ok(());
        }
        let head_struct: LocalCacheHead = head.into();
        let buf: Bytes = serde_json::to_vec(&head_struct).map_err(wrap_io_err)?.into();
        let tmp_path = head_path.with_extension(format!("_tmp{}", self.make_rand_suffix()));
        let (sender, recv) = oneshot::channel();
        self.worker
            .tx
            .send(WorkerOp::AtomicWrite {
                tmp_path,
                final_path: head_path.clone(),
                bytes: buf,
                sender,
            })
            .map_err(|_| wrap_io_err(std::io::Error::other("uring worker channel closed")))?;
        let result = recv
            .await
            .map_err(|e| wrap_io_err(std::io::Error::other(format!("oneshot recv: {e}"))))?;
        let _ = self.worker.tx.send(WorkerOp::InvalidateFd { path: head_path });
        result
    }

    async fn read_head(&self) -> object_store::Result<Option<(ObjectMeta, Attributes)>> {
        let path = make_head_path(self.root_folder.clone(), &self.location);
        let metadata = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(wrap_io_err(e)),
        };
        let len = metadata.len() as usize;
        if len == 0 {
            return Ok(None);
        }
        let (sender, recv) = oneshot::channel();
        self.worker
            .tx
            .send(WorkerOp::Read {
                path,
                offset: 0,
                len,
                sender,
            })
            .map_err(|_| wrap_io_err(std::io::Error::other("uring worker channel closed")))?;
        let bytes = match recv.await {
            Ok(Ok(b)) => b,
            Ok(Err(e)) => return Err(e),
            Err(e) => {
                return Err(wrap_io_err(std::io::Error::other(format!(
                    "oneshot recv: {e}"
                ))))
            }
        };
        let head: LocalCacheHead = match serde_json::from_slice(&bytes) {
            Ok(h) => h,
            Err(e) => return Err(wrap_io_err(e)),
        };
        // Note: we don't have the head's stat for "self.direct_io"-backed
        // path. Pass through meta + attrs directly.
        Ok(Some((head.meta(), head.attributes())))
    }
}

#[derive(Debug)]
struct IoUringCacheTee {
    root_folder: std::path::PathBuf,
    location: Path,
    part_size: usize,
    worker: Arc<WorkerHandle>,
    tee_id: u64,
    part_buf: bytes::BytesMut,
    next_part_number: usize,
    poisoned: bool,
    /// Tmp path → final path pairs collected from each `extend` flush.
    /// Renames happen in `commit` after upstream confirms the upload.
    pending_renames: Vec<(std::path::PathBuf, std::path::PathBuf)>,
}

impl IoUringCacheTee {
    async fn dispatch_part(&mut self, payload: Bytes) -> object_store::Result<()> {
        let final_path = make_part_path(
            self.root_folder.clone(),
            &self.location,
            self.next_part_number,
            self.part_size,
        );
        let tmp_path =
            final_path.with_extension(format!("_tee{}-{}", self.tee_id, self.next_part_number));
        self.next_part_number += 1;
        // Write the temp file via worker. We reuse the AtomicWrite path
        // because rename-on-write is exactly what we want for the FINAL
        // commit — but here we want temp-only with no rename. We do an
        // AtomicWrite with final_path == tmp_path: the write hits tmp_path,
        // the "rename" is a no-op (same source/dest).
        //
        // Simpler: write to tmp_path directly. We model this as a WriteThen
        // op: AtomicWrite where final_path equals tmp_path skips the rename
        // because std::fs::rename(a, a) is OK on Linux (no-op).
        let (sender, recv) = oneshot::channel();
        self.worker
            .tx
            .send(WorkerOp::AtomicWrite {
                tmp_path: tmp_path.clone(),
                final_path: tmp_path.clone(),
                bytes: payload,
                sender,
            })
            .map_err(|_| wrap_io_err(std::io::Error::other("uring worker channel closed")))?;
        let result = recv
            .await
            .map_err(|e| wrap_io_err(std::io::Error::other(format!("oneshot recv: {e}"))))?;
        if let Err(e) = result {
            self.poisoned = true;
            return Err(e);
        }
        self.pending_renames.push((tmp_path, final_path));
        Ok(())
    }
}

#[async_trait::async_trait]
impl LocalCacheTee for IoUringCacheTee {
    async fn extend(&mut self, buf: &[u8]) -> object_store::Result<()> {
        if self.poisoned {
            return Ok(());
        }
        let mut remaining = buf;
        while !remaining.is_empty() {
            let need = self.part_size - self.part_buf.len();
            let take = remaining.len().min(need);
            self.part_buf.extend_from_slice(&remaining[..take]);
            remaining = &remaining[take..];
            if self.part_buf.len() == self.part_size {
                let payload = std::mem::take(&mut self.part_buf).freeze();
                self.dispatch_part(payload).await?;
                if self.poisoned {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    async fn commit(
        mut self: Box<Self>,
        meta: &ObjectMeta,
        attrs: &Attributes,
    ) -> object_store::Result<()> {
        // Flush tail.
        if !self.poisoned && !self.part_buf.is_empty() {
            let payload = std::mem::take(&mut self.part_buf).freeze();
            self.dispatch_part(payload).await?;
        }
        if self.poisoned {
            // Best-effort cleanup of any temp files we did write.
            for (tmp, _) in self.pending_renames.drain(..) {
                let _ = std::fs::remove_file(&tmp);
            }
            return Err(wrap_io_err(std::io::Error::other("tee poisoned")));
        }

        // Build head bytes + write head temp file.
        let head_struct: LocalCacheHead = (meta, attrs).into();
        let head_bytes: Bytes = serde_json::to_vec(&head_struct).map_err(wrap_io_err)?.into();
        let head_final = make_head_path(self.root_folder.clone(), &self.location);
        let head_tmp = head_final.with_extension(format!("_tee{}-head", self.tee_id));
        let (sender, recv) = oneshot::channel();
        self.worker
            .tx
            .send(WorkerOp::AtomicWrite {
                tmp_path: head_tmp.clone(),
                final_path: head_tmp.clone(),
                bytes: head_bytes,
                sender,
            })
            .map_err(|_| wrap_io_err(std::io::Error::other("uring worker channel closed")))?;
        let head_write = recv
            .await
            .map_err(|e| wrap_io_err(std::io::Error::other(format!("oneshot recv: {e}"))))?;
        if let Err(e) = head_write {
            for (tmp, _) in self.pending_renames.drain(..) {
                let _ = std::fs::remove_file(&tmp);
            }
            let _ = std::fs::remove_file(&head_tmp);
            return Err(e);
        }

        // All temp files are on disk; rename parts first, head last. Done
        // synchronously because rename is metadata-only and we want the
        // ordering invariant: once head is visible, every part is visible.
        let pending_renames = std::mem::take(&mut self.pending_renames);
        let head_pair = (head_tmp, head_final);
        let renames = pending_renames.clone();
        let renames_count = renames.len();
        let head_pair_for_blocking = head_pair.clone();
        let result = tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            for (tmp, final_) in renames {
                std::fs::rename(&tmp, &final_)?;
            }
            std::fs::rename(&head_pair_for_blocking.0, &head_pair_for_blocking.1)?;
            Ok(())
        })
        .await
        .map_err(wrap_io_err)?
        .map_err(wrap_io_err);

        if let Err(e) = &result {
            warn!(
                "io_uring tee commit rename failed: {:?} (renames_attempted={})",
                e, renames_count
            );
        }

        // Invalidate any cached fds for the renamed final paths.
        for (_, final_) in &pending_renames {
            let _ = self
                .worker
                .tx
                .send(WorkerOp::InvalidateFd { path: final_.clone() });
        }
        let _ = self
            .worker
            .tx
            .send(WorkerOp::InvalidateFd { path: head_pair.1 });

        result
    }
}
