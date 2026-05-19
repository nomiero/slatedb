pub(crate) use object_store::CachedObjectStore;
#[allow(unused_imports)]
pub use storage::{LocalCacheEntry, LocalCacheHead, LocalCacheStorage, LocalCacheTee, PartID};
pub use storage_fs::FsCacheStorage;

pub mod admission;
pub mod stats;

mod object_store;
mod storage;
mod storage_fs;
#[cfg(target_os = "linux")]
mod storage_uring;
