use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use bytes::{BufMut, Bytes};

/// A named, configurable filter policy.
///
/// Policies control how filters are built, encoded, and decoded. The engine
/// stores `name()` per-SST so that the reader knows whether it can decode and
/// use the filter block.
pub trait FilterPolicy: Send + Sync {
    /// An identifier for this policy. Stored per-SST so the reader knows
    /// whether it can decode and use the filter block.
    ///
    /// The name should encode anything that affects **compatibility** —
    /// i.e., anything that would make a filter unreadable or produce wrong
    /// results if mismatched. For the built-in bloom filter:
    /// - `bits_per_key` is not included — changing it affects quality (FP
    ///   rate) but not the encoding format, so any reader can decode any
    ///   bloom filter regardless of bits_per_key.
    /// - `prefix_extractor` name is included — it changes which hashes are
    ///   stored, so querying with a different extractor produces false
    ///   negatives.
    ///
    /// Examples: `"slatedb.BloomFilter"`,
    /// `"slatedb.BloomFilter:prefix=fixed3"`.
    fn name(&self) -> &str;

    /// Creates a new builder for constructing a filter.
    fn builder(&self) -> Box<dyn FilterBuilder>;

    /// Decodes a previously encoded filter.
    ///
    /// The engine validates that the SST's filter policy name matches
    /// `self.name()` before calling this to ensure the policy can deserialize
    /// the data.
    fn decode(&self, data: &[u8]) -> Arc<dyn Filter>;

    /// Estimates the encoded size in bytes for a filter with `num_keys` keys.
    ///
    /// This is a hint used by the SST builder to reserve buffer space before
    /// the filter is built. It does not need to be exact.
    ///
    /// For the built-in bloom filter, the actual filter is sized from the
    /// real number of hashes collected during `add_key`, not from this estimate,
    /// so overestimates waste a small allocation and underestimates just trigger
    /// a reallocation.
    fn estimate_size(&self, num_keys: usize) -> usize;
}

/// Accumulator for keys during SST construction that produces a [`Filter`].
pub trait FilterBuilder: Send {
    /// Adds a key to the filter being built.
    fn add_key(&mut self, key: &[u8]);

    /// Finalizes and returns the completed filter.
    ///
    /// If no keys were added, returns a filter that returns `true` for all
    /// queries. The SST builder may also skip writing the filter block
    /// entirely when zero keys have been added.
    fn build(&self) -> Arc<dyn Filter>;
}

/// A read-only filter that answers membership queries.
pub trait Filter: Send + Sync {
    /// Returns `true` if the filter cannot rule out the query.
    /// A return value of `false` guarantees no matching key exists.
    fn might_match(&self, query: &FilterQuery) -> bool;

    /// Serializes the filter into the provided buffer.
    fn encode(&self, writer: &mut dyn BufMut);

    /// Returns the size of the filter's data in bytes.
    ///
    /// This should reflect the underlying data structure size (e.g., the
    /// bit array length for a bloom filter). Can be used for memory
    /// tracking, cache accounting, etc.
    fn size(&self) -> usize;
}

/// A membership query passed to [`Filter::might_match`].
#[derive(Debug, Clone)]
pub struct FilterQuery {
    /// The kind of query (point or prefix).
    pub kind: FilterQueryKind,
    /// Opaque hints provided by the caller (e.g., version bounds).
    /// Keyed by a string name so custom filters can look up relevant hints.
    pub hints: HashMap<String, Bytes>,
}

impl FilterQuery {
    /// Creates a point query with no hints.
    pub fn point(key: Bytes) -> Self {
        Self {
            kind: FilterQueryKind::Point(key),
            hints: HashMap::new(),
        }
    }

    /// Creates a prefix query with no hints.
    pub fn prefix(prefix: Bytes) -> Self {
        Self {
            kind: FilterQueryKind::Prefix(prefix),
            hints: HashMap::new(),
        }
    }

    /// Attaches hints to the query.
    pub fn with_hints(mut self, hints: HashMap<String, Bytes>) -> Self {
        self.hints = hints;
        self
    }
}

/// The kind of filter query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterQueryKind {
    /// Used to test whether a specific key might exist in the SST.
    Point(Bytes),
    /// Used to test whether any key with the given prefix might exist in the SST.
    Prefix(Bytes),
}

/// Extractor for a prefix from a key for use in prefix-based filtering.
///
/// Used on the write path to hash prefixes into the filter during SST
/// construction.
pub trait PrefixExtractor: Send + Sync {
    /// A unique name identifying this extractor's configuration.
    ///
    /// Changing the extractor (e.g. switching from a 4-byte fixed prefix
    /// to a delimiter-based one) changes which hashes are stored in the
    /// filter, so existing filters become invalid. The built-in
    /// `BloomFilterPolicy` includes this name in the policy name it
    /// writes to SST metadata (e.g. `"slatedb.BloomFilter:prefix=fixed3"`),
    /// which lets the reader detect the mismatch and skip the filter
    /// instead of returning wrong results.
    fn name(&self) -> &str;

    /// Returns whether the given prefix is a valid output of `extract()`.
    ///
    /// This is used on the read path to verify that a scan prefix provided
    /// by the user matches the prefix format indexed in the filter. If this
    /// returns `false`, the filter must NOT be consulted; doing so can
    /// produce false negatives.
    fn in_domain(&self, prefix: &[u8]) -> bool;

    /// Extracts the prefix from a key. Returns `None` if the key does not
    /// contain a recognizable prefix (i.e., `in_domain` would return `false`
    /// for the key).
    fn extract<'a>(&self, key: &'a [u8]) -> Option<&'a [u8]>;
}

/// A prefix extractor that uses a fixed number of bytes as the prefix.
///
/// Keys shorter than `prefix_len` are not in domain (both `in_domain` and
/// `extract` return `false`/`None`). Keys at least `prefix_len` bytes long
/// have their first `prefix_len` bytes extracted as the prefix.
///
/// # Example
///
/// ```
/// use slatedb::filter_policy::{PrefixExtractor, FixedPrefixExtractor};
///
/// let extractor = FixedPrefixExtractor::new(3);
/// assert_eq!(extractor.name(), "fixed3");
/// assert_eq!(extractor.extract(b"abcdef"), Some(&b"abc"[..]));
/// assert_eq!(extractor.extract(b"ab"), None);
/// assert!(extractor.in_domain(b"abc"));
/// assert!(!extractor.in_domain(b"ab"));
/// ```
pub struct FixedPrefixExtractor {
    prefix_len: usize,
    name: String,
}

impl FixedPrefixExtractor {
    pub fn new(prefix_len: usize) -> Self {
        assert!(prefix_len > 0, "prefix_len must be greater than 0");
        Self {
            prefix_len,
            name: format!("fixed{}", prefix_len),
        }
    }
}

impl fmt::Debug for FixedPrefixExtractor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FixedPrefixExtractor")
            .field("prefix_len", &self.prefix_len)
            .finish()
    }
}

impl PrefixExtractor for FixedPrefixExtractor {
    fn name(&self) -> &str {
        &self.name
    }

    fn in_domain(&self, prefix: &[u8]) -> bool {
        prefix.len() >= self.prefix_len
    }

    fn extract<'a>(&self, key: &'a [u8]) -> Option<&'a [u8]> {
        if key.len() >= self.prefix_len {
            Some(&key[..self.prefix_len])
        } else {
            None
        }
    }
}
