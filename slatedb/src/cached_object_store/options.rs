use std::path::PathBuf;
use std::time::Duration;

use duration_str::deserialize_option_duration;
use serde::{Deserialize, Serialize, Serializer};

/// Configuration for [`super::CachedObjectStore`] when constructed via
/// [`super::CachedObjectStore::from_config`].
///
/// Users who want the bundled disk cache construct this, hand it to
/// `from_config` along with a backing object store, and pass the
/// resulting `CachedObjectStore` to `Db::builder`. The SlateDB core no
/// longer owns this configuration; it is purely an input to the cache
/// constructor.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ObjectStoreCacheOptions {
    /// Root folder where part files are stored. Required.
    pub root_folder: Option<PathBuf>,

    /// The limit of the cache size in bytes. Default is 16 GB on
    /// 64, bit systems and 4 GB on 32, bit systems.
    pub max_cache_size_bytes: Option<usize>,

    /// The size of each part file, must be a multiple of 1 KB.
    /// Default is 4 MB.
    pub part_size_bytes: usize,

    /// Whether to cache PUT operations to disk. When enabled, writes
    /// admitted by the intent based policy are also written to the
    /// local cache. Default is false.
    pub cache_puts: bool,

    /// Interval to scan the cache directory and rebuild the in,
    /// memory map for the evictor. Default is 1 hour. If `None`, the
    /// directory is scanned only once on startup.
    #[serde(deserialize_with = "deserialize_option_duration")]
    #[serde(
        serialize_with = "serialize_option_duration",
        skip_serializing_if = "Option::is_none"
    )]
    pub scan_interval: Option<Duration>,

    /// The maximum number of file handles to keep open. When the
    /// limit is reached, the least recently used handle is closed.
    /// Default is 1000.
    pub max_open_file_handles: usize,
}

impl Default for ObjectStoreCacheOptions {
    fn default() -> Self {
        Self {
            root_folder: None,
            #[cfg(target_pointer_width = "32")]
            max_cache_size_bytes: Some(usize::MAX),
            #[cfg(not(target_pointer_width = "32"))]
            max_cache_size_bytes: Some(16 * 1024 * 1024 * 1024),
            part_size_bytes: 4 * 1024 * 1024,
            cache_puts: false,
            scan_interval: Some(Duration::from_secs(3600)),
            max_open_file_handles: 1000,
        }
    }
}

fn serialize_option_duration<S>(
    duration: &Option<Duration>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match duration {
        Some(d) => {
            let secs = d.as_secs();
            let millis = d.subsec_millis();
            let duration_str = if secs > 0 && millis > 0 {
                format!("{secs}s+{millis:03}ms")
            } else if millis > 0 {
                format!("{millis:03}ms")
            } else {
                format!("{secs}s")
            };
            serializer.serialize_str(&duration_str)
        }
        None => serializer.serialize_none(),
    }
}
