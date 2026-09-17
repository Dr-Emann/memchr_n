#![doc = include_str!("../README.md")]
#![deny(unnameable_types, unreachable_pub, missing_docs)]

mod bitset;
mod search;
mod swar;
mod vector;

use crate::bitset::{ByteRange, inclusive_range};
use crate::search::{BitsetLookup, FixedNibble, FixedNibbleTable, NibbleLookup, SearchPlan};
use core::fmt;
use core::ops::RangeBounds;

pub use bitset::ByteSet;

#[cfg(feature = "manual_level")]
pub use fearless_simd::Level;
#[cfg(not(feature = "manual_level"))]
use fearless_simd::Level;

/// Selects the engine of kernels used to build a [`MemchrN`].
///
/// # Examples
///
/// ```
/// use memchr_n::{Backend, MemchrN};
///
/// let finder = MemchrN::new_with_backend(b"!?", Backend::Swar);
/// assert_eq!(finder.find(b"well, hello!"), Some(11));
/// ```
#[derive(Copy, Clone, Default, Debug)]
#[non_exhaustive]
pub enum Backend {
    /// The best kernels the running CPU supports.
    #[default]
    Auto,
    /// Word-at-a-time kernels, even on a target that has vectors.
    Swar,
    /// The kernels supported by an explicit [`Level`].
    #[cfg(feature = "manual_level")]
    Level(Level),
}

/// Searches for bytes belonging to a fixed set.
///
/// # Examples
///
/// ```
/// use memchr_n::MemchrN;
///
/// let finder = MemchrN::new(b"aeiou");
/// assert_eq!(finder.find(b"rhythm and blues"), Some(7));
/// assert_eq!(finder.iter(b"rust").collect::<Vec<_>>(), vec![1]);
/// ```
#[derive(Clone)]
pub struct MemchrN {
    search: SearchPlan,
}

impl fmt::Debug for MemchrN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemchrN")
            .field("engine", &self.search.engine())
            .field("kernel_kind", &self.search.kernel_kind())
            .finish_non_exhaustive()
    }
}

impl MemchrN {
    /// Builds a searcher for the distinct values in `bytes` using the best supported kernels.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::MemchrN;
    ///
    /// let finder = MemchrN::new(b"!?");
    /// assert_eq!(finder.find(b"well, hello!"), Some(11));
    /// ```
    #[inline]
    pub fn new(bytes: &[u8]) -> Self {
        Self::new_with_backend(bytes, Backend::Auto)
    }

    /// Builds a searcher for the distinct values in `bytes` using `backend`.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::{Backend, MemchrN};
    ///
    /// let finder = MemchrN::new_with_backend(b"!?", Backend::Swar);
    /// assert_eq!(finder.find(b"well, hello!"), Some(11));
    /// ```
    pub fn new_with_backend(bytes: &[u8], backend: Backend) -> Self {
        Self::from_byte_set_with_backend(ByteSet::from_bytes(bytes), backend)
    }

    /// Builds a searcher for the members of `set` using the best supported kernels.
    ///
    /// A [`ByteSet`] describes any set of byte values, so this is the most general way to build a
    /// searcher. An empty set never matches.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::{ByteSet, MemchrN};
    ///
    /// let mut set = ByteSet::from_bytes(b"_");
    /// set.add_range(b'a'..=b'z');
    ///
    /// let finder = MemchrN::from_byte_set(set);
    /// assert_eq!(finder.find(b"  x"), Some(2));
    /// ```
    #[inline]
    pub fn from_byte_set(set: ByteSet) -> Self {
        Self::from_byte_set_with_backend(set, Backend::Auto)
    }

    /// Builds a searcher for the members of `set` using `backend`.
    ///
    /// An empty set never matches.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::{Backend, ByteSet, MemchrN};
    ///
    /// let mut set = ByteSet::full();
    /// set.remove_range(b'a'..=b'z');
    ///
    /// let finder = MemchrN::from_byte_set_with_backend(set, Backend::Swar);
    /// assert_eq!(finder.find(b"hello world"), Some(5));
    /// ```
    pub fn from_byte_set_with_backend(set: ByteSet, backend: Backend) -> Self {
        const MEMBERS_MAX: usize = 16;

        let engine = backend.engine();
        let mut members = [0; MEMBERS_MAX];
        let Some(count) = set.write_members(&mut members) else {
            if let Some(byte) = set.excluded_byte() {
                return Self::of_not_byte(engine, byte);
            }
            // Ranges remain cheap even when too large for a named-member kernel.
            if let Some(range) = set.as_contiguous_range() {
                return Self::of_range(engine, range);
            }
            return Self::of_small_set(engine, &set)
                .unwrap_or_else(|| Self::of_bitset_lookup(engine, set));
        };

        let members = &members[..usize::from(count)];
        match *members {
            [] => Self::of_never(),
            [first] => Self::of_needles(engine, [first]),
            [first, second] => Self::of_needles(engine, [first, second]),
            [first, second, third] => Self::of_needles(engine, [first, second, third]),
            [start, .., last] if usize::from(last - start) + 1 == members.len() => {
                Self::of_range(engine, ByteRange { start, last })
            }
            _ => Self::of_fixed_nibble_set(engine, members)
                .or_else(|| Self::of_small_set(engine, &set))
                .unwrap_or_else(|| Self::of_bitset_lookup(engine, set)),
        }
    }

    /// Builds a searcher for every byte except `byte` using the best supported kernels.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::MemchrN;
    ///
    /// let finder = MemchrN::from_not_byte(b' ');
    /// assert_eq!(finder.find(b"   hello"), Some(3));
    /// ```
    #[inline]
    pub fn from_not_byte(byte: u8) -> Self {
        Self::from_not_byte_with_backend(byte, Backend::Auto)
    }

    /// Builds a searcher for every byte except `byte` using `backend`.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::{Backend, MemchrN};
    ///
    /// let finder = MemchrN::from_not_byte_with_backend(b' ', Backend::Swar);
    /// assert_eq!(finder.iter(b" a b ").collect::<Vec<_>>(), vec![1, 3]);
    /// ```
    pub fn from_not_byte_with_backend(byte: u8, backend: Backend) -> Self {
        Self::of_not_byte(backend.engine(), byte)
    }

    /// Builds a searcher for the bytes in a [`RangeBounds<u8>`] using the best supported kernels.
    ///
    /// An empty range matches nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::MemchrN;
    ///
    /// let finder = MemchrN::from_range(b'0'..b':');
    /// assert_eq!(finder.find(b"page 2"), Some(5));
    /// ```
    #[inline]
    pub fn from_range(range: impl RangeBounds<u8>) -> Self {
        Self::from_range_with_backend(range, Backend::Auto)
    }

    /// Builds a searcher for the bytes in a [`RangeBounds<u8>`] using `backend`.
    ///
    /// An empty range matches nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::{Backend, MemchrN};
    ///
    /// let finder = MemchrN::from_range_with_backend(b'0'..b':', Backend::Swar);
    /// assert_eq!(finder.find(b"page 2"), Some(5));
    /// ```
    pub fn from_range_with_backend(range: impl RangeBounds<u8>, backend: Backend) -> Self {
        let mut set = ByteSet::new();
        if let Some(range) = inclusive_range(range) {
            set.add_byte_range(range);
        }
        Self::from_byte_set_with_backend(set, backend)
    }

    /// Returns the offset of the first matching byte in `haystack`.
    ///
    /// Prefer [`iter`](Self::iter) rather than calling this function repeatedly
    /// to iterate over the instances of matching bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::MemchrN;
    ///
    /// let finder = MemchrN::new(b"aeiou");
    /// assert_eq!(finder.find(b"rhythm and blues"), Some(7));
    /// ```
    #[inline]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        self.search.first_match(haystack)
    }

    /// Returns an [`Iter`] over the offsets of every matching byte in `haystack`.
    ///
    /// If only the first match is needed, prefer [`find`](Self::find).
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::MemchrN;
    ///
    /// let finder = MemchrN::new(b"aeiou");
    /// assert_eq!(finder.iter(b"hello").collect::<Vec<_>>(), vec![1, 4]);
    /// ```
    #[inline]
    pub fn iter<'a>(&'a self, haystack: &'a [u8]) -> Iter<'a> {
        Iter {
            finder: self,
            state: IterState {
                haystack,
                scan_offset: 0,
                match_base: 0,
            },
            match_bits: 0,
        }
    }
}

impl FromIterator<u8> for MemchrN {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        Self::from_byte_set(ByteSet::from_iter(iter))
    }
}

/// Iterates over the offsets of bytes matched by a [`MemchrN`].
///
/// Values of this type are created by [`MemchrN::iter`].
///
/// # Examples
///
/// ```
/// use memchr_n::{Iter, MemchrN};
///
/// let finder = MemchrN::new(b"aeiou");
/// let matches: Iter<'_> = finder.iter(b"hello");
/// assert_eq!(matches.collect::<Vec<_>>(), vec![1, 4]);
/// ```
#[derive(Clone, Debug)]
pub struct Iter<'a> {
    finder: &'a MemchrN,
    state: IterState<'a>,
    match_bits: MatchedBitset,
}

type MatchedBitset = u128;

/// A haystack and the window of it covered by the current match batch.
///
/// The batch spans `match_base..scan_offset`, which is never wider than a [`MatchedBitset`], so
/// any offset below `scan_offset` is reachable as a shift of the bitset.
#[derive(Clone, Debug)]
struct IterState<'a> {
    haystack: &'a [u8],
    scan_offset: usize,
    match_base: usize,
}

impl<'a> Iter<'a> {
    /// Discards remaining matches before the byte offset `idx`.
    ///
    /// The next call to [`next`](Iterator::next) yields the first remaining match
    /// at or after `idx`. An index at or before the last yielded offset has no
    /// effect, and advancing never rewinds the iterator. An index at or beyond
    /// the haystack length exhausts the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::MemchrN;
    ///
    /// let finder = MemchrN::new(b"aeiou");
    /// let mut matches = finder.iter(b"hello world");
    /// assert_eq!(matches.next(), Some(1));
    /// matches.advance_to(4);
    /// assert_eq!(matches.next(), Some(4));
    /// matches.advance_to(1);
    /// assert_eq!(matches.next(), Some(7));
    /// ```
    #[inline]
    pub fn advance_to(&mut self, idx: usize) {
        if idx >= self.state.scan_offset {
            self.match_bits = 0;
            self.state.scan_offset = idx.min(self.state.haystack.len());
            self.state.match_base = self.state.scan_offset;
        } else if idx > self.state.match_base {
            self.match_bits &= MatchedBitset::MAX << (idx - self.state.match_base);
        }
    }

    #[inline]
    fn refill(&mut self) -> Option<()> {
        if self.state.scan_offset == self.state.haystack.len() {
            return None;
        }
        // SAFETY: scans and `advance_to` keep `scan_offset` within the haystack length.
        self.match_bits = unsafe { self.finder.search.next_match_batch(&mut self.state) };
        (self.match_bits != 0).then_some(())
    }

    #[inline]
    fn take_lowest(&mut self) -> usize {
        debug_assert!(self.match_bits != 0);

        let bit = self.match_bits.trailing_zeros() as usize;
        self.match_bits &= self.match_bits - 1;
        let res = self.state.match_base + bit;

        // SAFETY: callers ensure `match_bits` is nonzero. `ScanOps::next_match_batch` guarantees
        // each set bit names an in-bounds byte relative to `match_base`; consuming or skipping
        // matches only clears bits.
        unsafe { core::hint::assert_unchecked(res < self.state.haystack.len()) };
        res
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.match_bits == 0 {
            self.refill()?;
        }
        Some(self.take_lowest())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let min = self.match_bits.count_ones() as usize;
        let max = min.checked_add(self.state.haystack.len() - self.state.scan_offset);
        (min, max)
    }

    fn count(self) -> usize {
        let mut total = self.match_bits.count_ones() as usize;
        // SAFETY: scans and `advance_to` keep `scan_offset` within the haystack length.
        let unscanned = unsafe { self.state.haystack.get_unchecked(self.state.scan_offset..) };
        if !unscanned.is_empty() {
            total += self.finder.search.count_all(unscanned);
        }
        total
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let mut remaining = n;
        loop {
            let held = self.match_bits.count_ones() as usize;
            if held > remaining {
                break;
            }
            remaining -= held;
            self.match_bits = 0;
            self.refill()?;
        }
        for _ in 0..remaining {
            self.match_bits &= self.match_bits - 1;
        }
        Some(self.take_lowest())
    }
}

impl core::iter::FusedIterator for Iter<'_> {}

#[derive(Copy, Clone, Debug)]
enum Engine {
    Vector(Level),
    Swar,
}

impl Backend {
    fn engine(self) -> Engine {
        match self {
            Backend::Auto => {
                let level = Level::new();
                if level.is_fallback() {
                    Engine::Swar
                } else {
                    Engine::Vector(level)
                }
            }
            Backend::Swar => Engine::Swar,
            #[cfg(feature = "manual_level")]
            Backend::Level(level) => Engine::Vector(level),
        }
    }
}

impl Engine {
    #[must_use]
    fn with_byte_shuffle(self) -> Self {
        match self {
            Engine::Vector(level) => {
                if vector::has_byte_shuffle(level) {
                    Engine::Vector(level)
                } else {
                    Engine::Swar
                }
            }
            Engine::Swar => Engine::Swar,
        }
    }
}

impl MemchrN {
    fn of_needles<const N: usize>(engine: Engine, needles: [u8; N]) -> Self
    where
        swar::kernels::AnyOf<N>: swar::Kernel,
        vector::kernels::AnyOf<N>: vector::Kernel,
    {
        let search = match engine {
            Engine::Vector(level) => {
                SearchPlan::vector(level, vector::kernels::AnyOf::new(needles))
            }
            Engine::Swar => SearchPlan::swar(swar::kernels::AnyOf::new(needles)),
        };
        Self { search }
    }

    fn of_not_byte(engine: Engine, byte: u8) -> Self {
        let search = match engine {
            Engine::Vector(level) => SearchPlan::vector(level, vector::kernels::NotByte::new(byte)),
            Engine::Swar => SearchPlan::swar(swar::kernels::NotByte::new(byte)),
        };
        Self { search }
    }

    fn of_range(engine: Engine, range: ByteRange) -> Self {
        let search = match engine {
            Engine::Vector(level) => {
                SearchPlan::vector(level, vector::kernels::OneRange::new(range))
            }
            Engine::Swar => SearchPlan::swar(swar::kernels::OneRange::new(range)),
        };
        Self { search }
    }

    fn of_small_set(engine: Engine, byte_set: &ByteSet) -> Option<Self> {
        let Engine::Vector(level) = engine else {
            return None;
        };
        if !vector::has_byte_shuffle(level) {
            return None;
        }
        let (lo_lookup, hi_lookup) = extract_nibble_lookups(byte_set)?;
        Some(Self {
            search: SearchPlan::vector(level, vector::kernels::SmallSet::new(lo_lookup, hi_lookup)),
        })
    }

    fn of_fixed_nibble_set(engine: Engine, possible_set: &[u8]) -> Option<Self> {
        let Engine::Vector(level) = engine else {
            return None;
        };
        if !vector::has_byte_shuffle(level) {
            return None;
        }
        let fixed_nibble_table = extract_fixed_nibble_table(possible_set)?;
        Some(Self {
            search: SearchPlan::vector(
                level,
                vector::kernels::FixedNibbleSet::new(fixed_nibble_table),
            ),
        })
    }

    fn of_bitset_lookup(engine: Engine, byte_set: ByteSet) -> Self {
        let kernel = BitsetLookup::new(byte_set);
        let search = match engine.with_byte_shuffle() {
            Engine::Vector(level) => SearchPlan::vector(level, kernel),
            Engine::Swar => SearchPlan::swar(kernel),
        };
        Self { search }
    }

    fn of_never() -> Self {
        Self {
            search: SearchPlan::never(),
        }
    }
}

/// Builds exact membership lookups for [`vector::kernels::SmallSet`], in low/high order.
///
/// The kernel accepts a byte when `lo_lookup[byte & 0x0F] & hi_lookup[byte >> 4] != 0`.
/// Each lookup entry is a `u8`, so there are eight bits available to identify groups of
/// bytes. A bit matches every combination of the high and low nibbles carrying it:
/// a rectangle in the 16-by-16 nibble matrix. For example, one bit can represent all
/// four bytes `{0x12, 0x13, 0xA2, 0xA3}` by marking high nibbles `{1, A}` and low
/// nibbles `{2, 3}`. The limit is eight groups, not eight bytes.
///
/// First, give each distinct nonempty row mask its own bit. High nibbles with the
/// same row mask share that bit, and every low nibble in the mask carries it too.
/// Because the rows are identical, every combination matched by the bit belongs
/// to the set. If this needs more than eight bits, transpose the matrix and group
/// identical columns instead, then return the lookups in the same low/high order.
///
/// This is a heuristic, not a search for a minimum rectangle cover. If both passes
/// need more than eight bits, return `None` so the caller uses a bitset lookup;
/// a different choice of overlapping rectangles might still fit in eight bits.
fn extract_nibble_lookups(byte_set: &ByteSet) -> Option<(NibbleLookup, NibbleLookup)> {
    let rows = byte_set.nibble_rows();
    if let Some((hi_lookup, lo_lookup)) = group_nibbles_by_mask(&rows) {
        return Some((lo_lookup, hi_lookup));
    }

    let mut columns = [0u16; 16];
    for (hi, &row) in rows.iter().enumerate() {
        let mut row = row;
        while row != 0 {
            let lo = row.trailing_zeros();
            columns[lo as usize] |= 1 << hi;
            row &= row - 1;
        }
    }
    group_nibbles_by_mask(&columns)
}

/// Encodes at most eight distinct nonempty masks as two intersecting nibble lookups.
///
/// Returns `(key_lookup, member_lookup)` such that their entries share a bit exactly
/// when `member_masks[key_nibble]` contains `member`. Each distinct mask gets a unique bit, stored
/// at every key with that mask and at every member it contains. A key gets only its
/// own mask's bit; a member can carry bits from several masks. Empty masks get no
/// bit, and a ninth distinct nonempty mask returns `None`.
///
/// For rows, keys are high nibbles and members are low nibbles. For columns, these
/// roles reverse. Sharing a bit only between identical masks prevents a key from
/// accepting members that belong exclusively to another mask.
fn group_nibbles_by_mask(member_masks: &[u16; 16]) -> Option<(NibbleLookup, NibbleLookup)> {
    const MAX_GROUPS: usize = 8;

    let mut key_lookup = NibbleLookup::default();
    let mut member_lookup = NibbleLookup::default();
    let mut group_masks = [0u16; MAX_GROUPS];
    let mut group_count = 0;

    for (key_nibble, &member_mask) in member_masks.iter().enumerate() {
        if member_mask == 0 {
            continue;
        }
        let existing_group = group_masks[..group_count]
            .iter()
            .position(|&candidate| candidate == member_mask)
            .map(|idx| idx as u8);
        let group_index = match existing_group {
            Some(group_index) => group_index,
            None => {
                if group_count == MAX_GROUPS {
                    return None;
                }
                let group_index = group_count as u8;
                group_masks[group_count] = member_mask;
                group_count += 1;
                let mut remaining_member_mask = member_mask;
                while remaining_member_mask != 0 {
                    member_lookup.set(remaining_member_mask.trailing_zeros() as u8, group_index);
                    remaining_member_mask &= remaining_member_mask - 1;
                }
                group_index
            }
        };
        key_lookup.set(key_nibble as u8, group_index);
    }
    Some((key_lookup, member_lookup))
}

fn extract_fixed_nibble_table(items: &[u8]) -> Option<FixedNibbleTable> {
    let first = *items.first()?;
    let (lo_nibble, hi_nibble) = (first & 0x0F, first >> 4);

    let (mut lo_constant, mut hi_constant) = (true, true);
    for &item in items {
        lo_constant &= item & 0x0F == lo_nibble;
        hi_constant &= item >> 4 == hi_nibble;
    }

    // An empty slot must contain a byte whose variable nibble differs from its index.
    // Zero works except at index zero, which uses the opposite one-bit nibble.
    if lo_constant {
        let mut table = [0; 16];
        table[0] = 0x10;
        for &item in items {
            table[usize::from(item >> 4)] = item;
        }
        Some(FixedNibbleTable {
            fixed_nibble: FixedNibble::Low,
            table,
        })
    } else if hi_constant {
        let mut table = [0; 16];
        table[0] = 0x01;
        for &item in items {
            table[usize::from(item & 0x0F)] = item;
        }
        Some(FixedNibbleTable {
            fixed_nibble: FixedNibble::High,
            table,
        })
    } else {
        None
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MemchrN>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use core::ops::Bound;

    // Construction is cheap enough to use per search, so keep the value within one cache line.
    #[test]
    fn memchr_n_stays_small() {
        assert!(
            size_of::<MemchrN>() <= 64,
            "MemchrN is {} bytes",
            size_of::<MemchrN>()
        );
    }

    #[test]
    fn debug_names_the_chosen_kernel() {
        let debug = format!("{:?}", MemchrN::new(b"az"));
        assert!(debug.contains("TwoBytes"), "{debug}");
    }

    #[track_caller]
    fn assert_nibble_lookups_match(set: &[u8]) {
        let (lo_lookup, hi_lookup) = extract_nibble_lookups(&ByteSet::from_bytes(set))
            .expect("the set should fit in eight nibble groups");
        for byte in 0..=u8::MAX {
            let matched =
                lo_lookup.0[usize::from(byte & 0x0F)] & hi_lookup.0[usize::from(byte >> 4)] != 0;
            assert_eq!(matched, set.contains(&byte), "byte {byte:#04x} in {set:?}");
        }
    }

    #[test]
    fn nibble_lookups_admit_exactly_their_set() {
        for set in [
            &[][..],
            &[0xFF],
            &[0x12, 0x13, 0xA2, 0xA3],
            &[0x12, 0x13, 0xA2, 0xA3, 0xB3, 0xB4],
        ] {
            assert_nibble_lookups_match(set);
        }
    }

    // Diagonal entries cannot share a group without also matching off-diagonal bytes.
    const DIAGONAL_SET: [u8; 9] = [0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

    #[test]
    fn nibble_lookups_respect_the_eight_group_limit() {
        assert_nibble_lookups_match(&DIAGONAL_SET[1..]);

        let byte_set = ByteSet::from_bytes(&DIAGONAL_SET);
        assert!(extract_nibble_lookups(&byte_set).is_none());
        let debug = format!(
            "{:?}",
            MemchrN::from_byte_set_with_backend(byte_set, Backend::Auto)
        );
        assert!(debug.contains("BitsetLookup"), "{debug}");
    }

    // Column lo contains rows {lo, lo + 1}, giving eight columns but nine distinct rows.
    fn column_grouped_set() -> Vec<u8> {
        let mut set = Vec::new();
        for lo in 0..8u8 {
            set.push((lo << 4) | lo);
            set.push(((lo + 1) << 4) | lo);
        }
        set
    }

    #[test]
    fn nibble_lookups_try_columns_when_rows_exceed_eight_groups() {
        let set = column_grouped_set();
        let byte_set = ByteSet::from_bytes(&set);
        assert!(group_nibbles_by_mask(&byte_set.nibble_rows()).is_none());
        assert_nibble_lookups_match(&set);
    }

    fn shuffling_vector_engine() -> bool {
        match Backend::Auto.engine() {
            Engine::Vector(level) => vector::has_byte_shuffle(level),
            Engine::Swar => false,
        }
    }

    #[test]
    fn character_classes_use_the_small_set_kernel() {
        for (name, set) in character_classes() {
            assert_nibble_lookups_match(&set);
            if shuffling_vector_engine() {
                let debug = format!("{:?}", MemchrN::new(&set));
                assert!(debug.contains("SmallSet"), "{name}: {debug}");
            }
        }
    }

    #[test]
    fn a_shared_nibble_prefers_the_fixed_nibble_kernel() {
        if !shuffling_vector_engine() {
            return;
        }
        // Both kernels accept these sets; FixedNibble needs one lookup instead of two.
        for set in [[0x03, 0x23, 0x53, 0xA3], [0x51, 0x53, 0x56, 0x5F]] {
            let debug = format!("{:?}", MemchrN::new(&set));
            assert!(debug.contains("FixedNibble"), "{set:?}: {debug}");
        }
    }

    #[test]
    fn auto_uses_swar_when_simd_is_unavailable() {
        for finder in [
            MemchrN::new(b"x"),
            MemchrN::new(b"xy"),
            MemchrN::new(b"xyz"),
            MemchrN::from_range(b'0'..=b'9'),
            MemchrN::from_not_byte(b'.'),
        ] {
            let uses_swar = match finder.search.engine() {
                Engine::Swar => true,
                Engine::Vector(level) => {
                    assert!(!level.is_fallback());
                    false
                }
            };
            assert_eq!(uses_swar, Level::new().is_fallback());
        }
    }

    #[cfg(feature = "manual_level")]
    #[test]
    fn explicit_level_preserves_the_vector_engine() {
        let level = Level::baseline();
        let finder = MemchrN::new_with_backend(b"x", Backend::Level(level));
        match finder.search.engine() {
            Engine::Vector(selected) => assert_eq!(selected.is_fallback(), level.is_fallback()),
            Engine::Swar => panic!("explicit SIMD level was replaced with SWAR"),
        }
    }

    #[test]
    fn from_range_accepts_inclusive_exclusive_and_unbounded_bounds() {
        let haystack = [0, 1, 2, 3, 254, 255];

        fn offsets(range: impl RangeBounds<u8>, haystack: &[u8]) -> Vec<usize> {
            let finder = MemchrN::from_range(range);
            let mut offsets = Vec::new();
            for offset in finder.iter(haystack) {
                offsets.push(offset);
            }
            offsets
        }

        assert_eq!(offsets(.., &haystack), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(offsets(..3, &haystack), vec![0, 1, 2]);
        assert_eq!(offsets(1..3, &haystack), vec![1, 2]);
        assert_eq!(offsets(1..=3, &haystack), vec![1, 2, 3]);
        assert_eq!(offsets(254.., &haystack), vec![4, 5]);
    }

    #[test]
    fn range_matches_single_bytes_across_both_bounds() {
        for backend in [Backend::Auto, Backend::Swar] {
            for (start, last) in [(b'0', b'9'), (0, 127), (127, 130), (128, 255), (0, 255)] {
                let searcher = MemchrN::from_range_with_backend(start..=last, backend);
                for byte in 0..=u8::MAX {
                    let expected = start <= byte && byte <= last;
                    let haystack = [byte];
                    assert_eq!(searcher.find(&haystack), expected.then_some(0));
                    assert_eq!(searcher.iter(&haystack).next(), expected.then_some(0));
                    assert_eq!(searcher.iter(&haystack).count(), usize::from(expected));
                }
            }
        }
    }

    #[test]
    fn from_range_treats_bounds_outside_the_byte_domain_as_empty() {
        let haystack = [0, 1, 254, 255];
        let past_max = (Bound::Excluded(u8::MAX), Bound::Unbounded);
        let before_min = (Bound::Unbounded, Bound::Excluded(u8::MIN));

        for range in [past_max, before_min] {
            assert_eq!(MemchrN::from_range(range).find(&haystack), None);
        }
        assert_eq!(MemchrN::from_range(3..3).find(&haystack), None);
        assert_eq!(
            MemchrN::from_range_with_backend(..0, Backend::Swar).find(&haystack),
            None
        );
    }

    #[test]
    fn add_byte_range_matches_adding_each_byte() {
        for start in 0..=u8::MAX {
            for last in start..=u8::MAX {
                let mut ranged = ByteSet::new();
                ranged.add_byte_range(ByteRange { start, last });

                let mut one_at_a_time = ByteSet::new();
                for byte in start..=last {
                    one_at_a_time.add(byte);
                }

                assert_eq!(ranged, one_at_a_time, "{start}..={last}");
            }
        }
    }

    #[test]
    fn add_byte_range_of_empty_range_adds_nothing() {
        let mut set = ByteSet::from_bytes(b"abc");
        let before = set;
        set.add_byte_range(ByteRange { start: 10, last: 9 });
        assert_eq!(set, before);
    }

    fn members(set: &ByteSet) -> Vec<u8> {
        let all: Vec<u8> = (0..=u8::MAX).collect();
        MemchrN::from_byte_set_with_backend(*set, Backend::Auto)
            .iter(&all)
            .map(|offset| all[offset])
            .collect()
    }

    fn assert_same_set_and_kernel(bulk: &ByteSet, one_at_a_time: &ByteSet, case: &str) {
        // Equal members alone would miss representation-selection regressions.
        assert_eq!(members(bulk), members(one_at_a_time), "{case}");
        for backend in [Backend::Auto, Backend::Swar] {
            assert_eq!(
                format!("{:?}", MemchrN::from_byte_set_with_backend(*bulk, backend)),
                format!(
                    "{:?}",
                    MemchrN::from_byte_set_with_backend(*one_at_a_time, backend)
                ),
                "{case} on {backend:?}"
            );
        }
    }

    #[test]
    fn from_bytes_matches_adding_each_byte() {
        let alnum: Vec<u8> = (b'0'..=b'9')
            .chain(b'a'..=b'z')
            .chain(b'A'..=b'Z')
            .collect();
        let mut contiguous_out_of_order: Vec<u8> = (0..=23).collect();
        contiguous_out_of_order.push(100);
        contiguous_out_of_order.extend(24..=99);

        let cases: &[(&str, Vec<u8>)] = &[
            ("62 alnum, scattered", alnum),
            ("0..=200 contiguous", (0..=200).collect()),
            ("every byte", (0..=u8::MAX).collect()),
            ("60 bytes, 3 distinct", b"cba".repeat(20)),
            ("60 bytes, 1 distinct", b"z".repeat(60)),
            ("100 bytes, 24 distinct", (0..100).map(|i| i % 24).collect()),
            ("100 bytes, 25 distinct", (0..100).map(|i| i % 25).collect()),
            ("contiguous, out of order", contiguous_out_of_order),
            (
                "25 scattered",
                (0..25).map(|i: u8| i.wrapping_mul(7)).collect(),
            ),
            ("one past the members a kernel can name", (0..17).collect()),
        ];

        for (case, bytes) in cases {
            let mut one_at_a_time = ByteSet::new();
            for &byte in bytes {
                one_at_a_time.add(byte);
            }
            assert_same_set_and_kernel(&ByteSet::from_bytes(bytes), &one_at_a_time, case);
        }
    }

    #[test]
    fn add_wide_range_to_non_empty_matches_adding_each_byte() {
        let seeds: &[&[u8]] = &[b"", b"z", b"\x00", b"\x7f", b"az", b"\x00\xff", b"aeiouAEI"];
        for seed in seeds {
            for (start, last) in [(0u8, 255u8), (0x80, 0xFF), (10, 40), (100, 124), (60, 200)] {
                let mut ranged = ByteSet::from_bytes(seed);
                ranged.add_byte_range(ByteRange { start, last });

                let mut one_at_a_time = ByteSet::from_bytes(seed);
                for byte in start..=last {
                    one_at_a_time.add(byte);
                }
                let case = format!("{seed:?} + {start}..={last}");
                assert_same_set_and_kernel(&ranged, &one_at_a_time, &case);
            }
        }
    }

    #[test]
    fn add_two_byte_ranges_matches_adding_each_byte() {
        // Includes bitset word boundaries and both ends of the byte domain.
        let bounds = [0u8, 1, 7, 23, 24, 25, 63, 64, 127, 128, 200, 254, 255];
        for &first_start in &bounds {
            for &first_last in bounds.iter().filter(|&&b| b >= first_start) {
                for &second_start in &bounds {
                    for &second_last in bounds.iter().filter(|&&b| b >= second_start) {
                        let mut ranged = ByteSet::new();
                        ranged.add_byte_range(ByteRange {
                            start: first_start,
                            last: first_last,
                        });
                        ranged.add_byte_range(ByteRange {
                            start: second_start,
                            last: second_last,
                        });

                        let mut one_at_a_time = ByteSet::new();
                        for byte in first_start..=first_last {
                            one_at_a_time.add(byte);
                        }
                        for byte in second_start..=second_last {
                            one_at_a_time.add(byte);
                        }

                        assert_eq!(
                            ranged, one_at_a_time,
                            "{first_start}..={first_last} then {second_start}..={second_last}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn add_keeps_a_byte_disjoint_from_an_existing_range() {
        let mut set = ByteSet::new();
        set.add_byte_range(ByteRange {
            start: 0,
            last: 100,
        });
        set.add(200);
        assert_eq!(members(&set), (0..=100).chain([200]).collect::<Vec<u8>>());
    }

    #[test]
    fn add_byte_range_works_in_const_context() {
        const DIGITS: ByteSet = {
            let mut set = ByteSet::new();
            set.add_byte_range(ByteRange {
                start: b'0',
                last: b'9',
            });
            set
        };
        assert_eq!(members(&DIGITS), b"0123456789");
    }

    #[track_caller]
    fn assert_same_finder(built: &MemchrN, expected: &MemchrN, case: &str) {
        let all: Vec<u8> = (0..=u8::MAX).collect();
        assert_eq!(
            built.iter(&all).collect::<Vec<_>>(),
            expected.iter(&all).collect::<Vec<_>>(),
            "{case}"
        );
        // Equal members alone would miss representation-selection regressions.
        assert_eq!(format!("{built:?}"), format!("{expected:?}"), "{case}");
    }

    #[test]
    fn from_byte_set_matches_the_dedicated_constructors() {
        let mut vowels = ByteSet::new();
        for &byte in b"aeiou" {
            vowels.add(byte);
        }
        let mut digits = ByteSet::new();
        digits.add_range(b'0'..=b'9');
        let mut not_dot = ByteSet::from_bytes(b".");
        not_dot.invert();
        let mut printable = ByteSet::full();
        printable.remove_range(0..=b' ' - 1);
        printable.remove(0x7F);
        printable.remove_range(0x80..=0xFF);

        assert_same_finder(
            &MemchrN::from_byte_set(vowels),
            &MemchrN::new(b"aeiou"),
            "list",
        );
        assert_same_finder(
            &MemchrN::from_byte_set(digits),
            &MemchrN::from_range(b'0'..=b'9'),
            "range",
        );
        assert_same_finder(
            &MemchrN::from_byte_set(not_dot),
            &MemchrN::from_not_byte(b'.'),
            "complement",
        );
        assert_same_finder(
            &MemchrN::from_byte_set(printable),
            &MemchrN::from_range(b' '..0x7F),
            "range carved out of a full set",
        );
        assert_same_finder(
            &MemchrN::from_byte_set(ByteSet::new()),
            &MemchrN::new(b""),
            "empty",
        );
    }

    #[test]
    fn from_byte_set_honors_the_backend() {
        let mut set = ByteSet::new();
        set.add_range(b'a'..=b'f');
        set.add_range(b'0'..=b'9');

        for backend in [Backend::Auto, Backend::Swar] {
            assert_same_finder(
                &MemchrN::from_byte_set_with_backend(set, backend),
                &MemchrN::new_with_backend(b"0123456789abcdef", backend),
                &format!("{backend:?}"),
            );
        }
    }

    fn build_word(bytes: &[u8]) -> MemchrN {
        MemchrN::new_with_backend(bytes, Backend::Swar)
    }

    fn build(bytes: &[u8]) -> MemchrN {
        MemchrN::new(bytes)
    }

    fn naive(set: &[u8], haystack: &[u8]) -> Vec<usize> {
        let mut offsets = Vec::new();
        for (offset, byte) in haystack.iter().enumerate() {
            if set.contains(byte) {
                offsets.push(offset);
            }
        }
        offsets
    }

    fn ascii_class(keep: fn(&u8) -> bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        for byte in 0..=127u8 {
            if keep(&byte) {
                bytes.push(byte);
            }
        }
        bytes
    }

    /// Common byte classes that fit in eight groups despite some having more than eight bytes.
    fn character_classes() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("alphanumeric", ascii_class(u8::is_ascii_alphanumeric)),
            ("alphabetic", ascii_class(u8::is_ascii_alphabetic)),
            ("punctuation", ascii_class(u8::is_ascii_punctuation)),
            ("hex digit", ascii_class(u8::is_ascii_hexdigit)),
            ("whitespace", ascii_class(u8::is_ascii_whitespace)),
            ("url unsafe", b" \"<>#%{}|".to_vec()),
            (
                "shell metacharacter",
                b"|&;<>()$`\\\"' \t\n*?[#~=%".to_vec(),
            ),
            ("regex metacharacter", b"\\.+*?()|[]{}^$".to_vec()),
            ("c escape", b"\\\"\n\r\t\x07\x08\x0c\x0b\0".to_vec()),
        ]
    }

    fn sets() -> Vec<Vec<u8>> {
        vec![
            vec![],
            b"z".to_vec(),
            b"az".to_vec(),
            b"azQ".to_vec(),
            b"aeiouAEI".to_vec(),
            b"abcdefghjl".to_vec(),
            b"0123456789abcdef".to_vec(),
            ascii_class(u8::is_ascii_alphanumeric),
            ascii_class(u8::is_ascii_punctuation),
            column_grouped_set(),
            DIAGONAL_SET.to_vec(),
            (b'0'..=b'9').collect(),
            (0x80..=0xFF).collect(),
            (0..=255u8).step_by(3).collect(),
        ]
    }

    fn haystack(len: usize) -> Vec<u8> {
        // The period does not align with 64-byte chunks.
        (0..len).map(|i| ((i * 37 + i / 7) % 251) as u8).collect()
    }

    #[test]
    fn matches_naive_across_lengths() {
        for set in sets() {
            for (name, searcher) in [("vector", build(&set)), ("word", build_word(&set))] {
                let lens = (0..=80).chain([127, 128, 129, 255, 256, 1000]);
                for len in lens {
                    let haystack = haystack(len);
                    let expected = naive(&set, &haystack);
                    let got: Vec<usize> = searcher.iter(&haystack).collect();
                    assert_eq!(got, expected, "{name} set {set:?} len {len}");
                    assert_eq!(
                        searcher.iter(&haystack).count(),
                        expected.len(),
                        "{name} count for set {set:?} len {len}"
                    );
                }
            }
        }
    }

    #[test]
    fn find_matches_naive() {
        for set in sets() {
            for (name, searcher) in [("vector", build(&set)), ("word", build_word(&set))] {
                let lens = (0..=80).chain([127, 128, 129, 255, 256, 1000]);
                for len in lens {
                    let haystack = haystack(len);
                    let expected = naive(&set, &haystack).first().copied();
                    assert_eq!(
                        searcher.find(&haystack),
                        expected,
                        "{name} set {set:?} len {len}"
                    );
                }
            }
        }
    }

    #[test]
    fn overlapping_tail_does_not_repeat_matches() {
        // The dense set exercises scalar `BitsetLookup` and contains `x`.
        let dense: Vec<u8> = (0..=u8::MAX).step_by(3).collect();
        for searcher in [build(b"x"), build_word(b"x"), build_word(&dense)] {
            for len in 0..192 {
                let haystack = vec![b'x'; len];
                let expected: Vec<usize> = (0..len).collect();
                assert_eq!(
                    searcher.iter(&haystack).collect::<Vec<_>>(),
                    expected,
                    "{len}"
                );
                assert_eq!(searcher.iter(&haystack).count(), len, "{len}");
            }
        }
    }

    #[test]
    fn advance_to_preserves_matches_at_and_after_the_index() {
        for backend in [Backend::Auto, Backend::Swar] {
            let finder = MemchrN::new_with_backend(b"x", backend);
            for len in [0, 1, 7, 8, 15, 16, 17, 63, 64, 65, 127, 128, 129, 257] {
                let haystack = vec![b'x'; len];
                for consumed in 0..=len {
                    let mut iter = finder.iter(&haystack);
                    for offset in 0..consumed {
                        assert_eq!(iter.next(), Some(offset));
                    }
                    for idx in 0..=len + 1 {
                        let mut iter = iter.clone();
                        iter.advance_to(idx);
                        let start = consumed.max(idx).min(len);
                        assert_eq!(iter.clone().count(), len - start);
                        let (min, max) = iter.size_hint();
                        assert!(min <= len - start);
                        assert!(max.unwrap() >= len - start);
                        for expected in start..len {
                            assert_eq!(iter.next(), Some(expected));
                        }
                        assert_eq!(iter.next(), None);
                        iter.advance_to(0);
                        assert_eq!(iter.next(), None);
                    }
                }
            }
        }
    }

    #[test]
    fn repeated_advance_to_matches_naive() {
        for set in sets() {
            for backend in [Backend::Auto, Backend::Swar] {
                let finder = MemchrN::new_with_backend(&set, backend);
                let haystack = haystack(1000);
                let expected = naive(&set, &haystack);
                let mut position = 0;
                let mut iter = finder.iter(&haystack);
                for idx in [
                    0,
                    17,
                    3,
                    17,
                    64,
                    127,
                    128,
                    129,
                    400,
                    256,
                    999,
                    usize::MAX,
                    0,
                ] {
                    iter.advance_to(idx);
                    while position < expected.len() && expected[position] < idx {
                        position += 1;
                    }
                    assert_eq!(iter.clone().count(), expected.len() - position);
                    iter.advance_to(0);
                    assert_eq!(iter.next(), expected.get(position).copied());
                    position = (position + 1).min(expected.len());
                    assert_eq!(iter.nth(1), expected.get(position + 1).copied());
                    position = (position + 2).min(expected.len());
                }
            }
        }
    }

    #[test]
    fn nth_matches_naive() {
        for set in sets() {
            for (name, searcher) in [("vector", build(&set)), ("word", build_word(&set))] {
                for len in (0..=80).chain([127, 128, 129, 255, 256, 1000]) {
                    let haystack = haystack(len);
                    let expected = naive(&set, &haystack);
                    for n in [0, 1, 2, 3, 7, 63, 64, 65, 100] {
                        assert_eq!(
                            searcher.iter(&haystack).nth(n),
                            expected.get(n).copied(),
                            "{name} set {set:?} len {len} nth {n}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn find_reports_every_offset() {
        for (name, searcher) in [("vector", build(b"x")), ("word", build_word(b"x"))] {
            for len in 0..=80 {
                for offset in 0..len {
                    let mut haystack = vec![b'.'; len];
                    haystack[offset] = b'x';
                    assert_eq!(
                        searcher.find(&haystack),
                        Some(offset),
                        "{name} len {len} offset {offset}"
                    );
                }
                assert_eq!(
                    searcher.find(&vec![b'.'; len]),
                    None,
                    "{name} miss len {len}"
                );
            }
        }
    }

    #[test]
    fn find_reports_first_offset_across_vector_batches() {
        for set in sets() {
            let Some(&needle) = set.first() else {
                continue;
            };
            let mut pad = 0;
            while set.contains(&pad) {
                pad += 1;
            }
            let searcher = build(&set);
            for len in [
                63, 64, 65, 79, 80, 81, 127, 128, 129, 255, 256, 257, 511, 512, 513,
            ] {
                for alignment in 0..64 {
                    let mut storage = vec![needle; len + 128];
                    let haystack = &mut storage[alignment..alignment + len];
                    haystack.fill(pad);
                    assert_eq!(searcher.find(haystack), None);
                    for offset in (0..len).rev() {
                        haystack[offset] = needle;
                        assert_eq!(
                            searcher.find(haystack),
                            Some(offset),
                            "set {set:?} len {len} alignment {alignment} offset {offset}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn per_byte_matcher_agrees_with_the_set() {
        for set in sets() {
            let pad = (0..=u8::MAX)
                .find(|byte| !set.contains(byte))
                .expect("no set here holds every byte");

            for (name, searcher) in [("vector", build(&set)), ("word", build_word(&set))] {
                for byte in 0..=u8::MAX {
                    // Covers both scalar probes and staged tails.
                    for len in [1, 2, 3, 4, 5, 7, 8, 9, 15] {
                        for offset in 0..len {
                            let mut haystack = vec![pad; len];
                            haystack[offset] = byte;
                            let expected = set.contains(&byte).then_some(offset);
                            assert_eq!(
                                searcher.find(&haystack),
                                expected,
                                "{name} set {set:?} byte {byte} len {len} offset {offset}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn nth_advances_like_repeated_next() {
        let set = b"aeiouAEI";
        let searcher = build(set);
        let haystack = haystack(1000);
        let expected = naive(set, &haystack);
        for step in [0, 1, 5, 64, 200] {
            let mut iter = searcher.iter(&haystack);
            let mut index = 0;
            let mut got = Vec::new();
            while let Some(offset) = iter.nth(step) {
                index += step;
                got.push((index, offset));
                index += 1;
            }
            let mut want = Vec::new();
            let mut index = step;
            while index < expected.len() {
                want.push((index, expected[index]));
                index += step + 1;
            }
            assert_eq!(got, want, "step {step}");
        }
    }

    #[test]
    fn count_matches_iteration_after_partial_consumption() {
        let searcher = build(b"aeiouAEI");
        let haystack = haystack(1000);
        for skip in [0, 1, 2, 7, 40] {
            let mut iter = searcher.iter(&haystack);
            let mut taken = 0;
            for _ in 0..skip {
                if iter.next().is_some() {
                    taken += 1;
                }
            }
            assert_eq!(iter.count() + taken, naive(b"aeiouAEI", &haystack).len());
        }
    }

    #[test]
    fn reports_a_match_from_either_half_of_a_pair() {
        for searcher in [build(b"x"), build_word(b"x")] {
            for len in [128, 192, 256, 300] {
                for offset in 0..len {
                    let mut haystack = vec![b'.'; len];
                    haystack[offset] = b'x';
                    assert_eq!(
                        searcher.iter(&haystack).collect::<Vec<_>>(),
                        vec![offset],
                        "len {len} offset {offset}"
                    );
                    assert_eq!(
                        searcher.find(&haystack),
                        Some(offset),
                        "find, len {len} offset {offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn find_agrees_with_the_iterator() {
        for set in sets() {
            for (name, searcher) in [("vector", build(&set)), ("word", build_word(&set))] {
                for len in (0..=80).chain([127, 128, 129, 191, 192, 255, 256, 1000]) {
                    let haystack = haystack(len);
                    assert_eq!(
                        searcher.find(&haystack),
                        searcher.iter(&haystack).next(),
                        "{name} set {set:?} len {len}"
                    );
                }
            }
        }
    }

    #[test]
    fn counts_do_not_overflow_the_accumulator() {
        let searcher = build(b"x");
        let len = 64 * (512 + 3);
        let haystack = vec![b'x'; len];
        assert_eq!(searcher.iter(&haystack).count(), len);
    }

    fn assert_not_byte(searcher: &MemchrN, excluded: u8, haystack: &[u8]) {
        let mut expected = Vec::new();
        for (offset, &byte) in haystack.iter().enumerate() {
            if byte != excluded {
                expected.push(offset);
            }
        }
        assert_eq!(searcher.find(haystack), expected.first().copied());
        assert_eq!(searcher.iter(haystack).collect::<Vec<_>>(), expected);
        assert_eq!(searcher.iter(haystack).count(), expected.len());
        for n in [0, 1, 7, 63, 64, 127, 128, haystack.len()] {
            let mut iter = searcher.iter(haystack);
            assert_eq!(iter.nth(n), expected.get(n).copied());
            assert_eq!(iter.count(), expected.len().saturating_sub(n + 1));
        }
        let mut iter = searcher.iter(haystack);
        let taken = usize::from(iter.next().is_some());
        assert_eq!(iter.count(), expected.len() - taken);
    }

    #[test]
    fn not_byte_constructors_and_selection_agree() {
        let haystack: Vec<u8> = (0..=u8::MAX).collect();
        for excluded in 0..=u8::MAX {
            let mut bytes = Vec::new();
            for byte in 0..=u8::MAX {
                if byte != excluded {
                    bytes.extend([byte, byte]);
                }
            }
            for backend in [Backend::Auto, Backend::Swar] {
                for searcher in [
                    MemchrN::from_not_byte_with_backend(excluded, backend),
                    MemchrN::new_with_backend(&bytes, backend),
                ] {
                    assert_eq!(format!("{:?}", searcher.search.kernel_kind()), "NotByte");
                    assert_not_byte(&searcher, excluded, &haystack);
                    for byte in 0..=u8::MAX {
                        assert_eq!(searcher.find(&[byte]), (byte != excluded).then_some(0));
                    }
                }
                for (range, excluded) in [(1..=255, 0), (0..=254, 255)] {
                    let searcher = MemchrN::from_range_with_backend(range, backend);
                    assert_eq!(format!("{:?}", searcher.search.kernel_kind()), "NotByte");
                    assert_not_byte(&searcher, excluded, &haystack);
                }
            }
            let searcher = MemchrN::from_not_byte(excluded);
            assert_not_byte(&searcher, excluded, &haystack);
            let searcher: MemchrN = bytes.into_iter().collect();
            assert_eq!(format!("{:?}", searcher.search.kernel_kind()), "NotByte");
            assert_not_byte(&searcher, excluded, &haystack);
        }
    }

    #[test]
    fn not_byte_handles_alignment_tails_and_match_density() {
        for excluded in [0, 1, b' ', 127, 128, 254, 255] {
            for backend in [Backend::Auto, Backend::Swar] {
                let searcher = MemchrN::from_not_byte_with_backend(excluded, backend);
                for len in [
                    0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192,
                    193, 255, 256, 257, 513,
                ] {
                    for alignment in 0..64 {
                        let mut storage = vec![excluded; len + 64];
                        let haystack = &mut storage[alignment..alignment + len];
                        assert_not_byte(&searcher, excluded, haystack);
                        if len != 0 {
                            for offset in [0, len / 2, len - 1] {
                                haystack[offset] = excluded.wrapping_add(1);
                                assert_not_byte(&searcher, excluded, haystack);
                                haystack[offset] = excluded;
                            }
                        }
                        haystack.fill(excluded.wrapping_add(1));
                        assert_not_byte(&searcher, excluded, haystack);
                        for offset in (0..len).step_by(3) {
                            haystack[offset] = excluded;
                        }
                        assert_not_byte(&searcher, excluded, haystack);
                    }
                }
                assert_not_byte(
                    &searcher,
                    excluded,
                    &vec![excluded.wrapping_add(1); 64 * 515],
                );
            }
        }
    }
}
