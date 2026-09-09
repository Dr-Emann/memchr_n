#![deny(unnameable_types, unreachable_pub)]

mod bitset;
mod search;
mod swar;
mod vector;

use crate::bitset::ByteSet;
use crate::search::{
    BitsetLookup, FixedNibble, FixedNibbleTable, KernelData, KernelKind, NibbleLookup, ScanOps,
    SearchPlan, never_scan_ops,
};
use core::fmt;
use core::ops::{Bound, RangeBounds};
use core::range::RangeInclusive;
use fearless_simd::dispatch;

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
            .field("engine", &self.search.engine)
            .field("kernel_kind", &self.search.kernel_kind)
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
        Self::from_set(ByteSet::from_bytes(bytes), backend)
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
            set.add_range(range);
        }
        Self::from_set(set, backend)
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
        // SAFETY: `self.search` keeps the scan operations, kernel data, and supported SIMD level together.
        unsafe { (self.search.scan_ops.first_match)(&self.search.kernel_data, haystack) }
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
        Self::from_set(ByteSet::from_iter(iter), Backend::Auto)
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
pub struct Iter<'a> {
    finder: &'a MemchrN,
    state: IterState<'a>,
    match_bits: MatchedBitset,
}

type MatchedBitset = u128;

struct IterState<'a> {
    haystack: &'a [u8],
    scan_offset: usize,
    match_base: usize,
}

impl<'a> Iter<'a> {
    #[inline]
    fn refill(&mut self) -> Option<()> {
        if self.state.scan_offset == self.state.haystack.len() {
            return None;
        }
        // SAFETY: `SearchPlan` keeps the scan operations, kernel data, and supported SIMD level together.
        self.match_bits = unsafe {
            (self.finder.search.scan_ops.next_match_batch)(
                &self.finder.search.kernel_data,
                &mut self.state,
            )
        };
        (self.match_bits != 0).then_some(())
    }

    #[inline]
    fn take_lowest(&mut self) -> usize {
        debug_assert!(self.match_bits != 0);

        let bit = self.match_bits.trailing_zeros() as usize;
        self.match_bits &= self.match_bits - 1;
        self.state.match_base + bit
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
        // SAFETY: `scan_offset` only ever moves to an offset a scan reached, so it is in bounds.
        let unscanned = unsafe { self.state.haystack.get_unchecked(self.state.scan_offset..) };
        if !unscanned.is_empty() {
            // SAFETY: `SearchPlan` keeps the scan operations, kernel data, and supported SIMD level together.
            total += unsafe {
                (self.finder.search.scan_ops.count_all)(&self.finder.search.kernel_data, unscanned)
            };
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

#[derive(Copy, Clone, Debug)]
enum Engine {
    Vector(Level),
    Swar,
}

impl Backend {
    fn engine(self) -> Engine {
        match self {
            Backend::Auto => Engine::Vector(Level::new()),
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
    fn from_set(set: ByteSet, backend: Backend) -> Self {
        const MEMBERS_MAX: usize = 16;

        let engine = backend.engine();
        let mut members = [0; MEMBERS_MAX];
        if let Some(count) = set.write_members(&mut members) {
            let members = &members[..usize::from(count)];

            match *members {
                [] => Self::of_never(),
                [first] => Self::of_needles(engine, [first]),
                [first, second] => Self::of_needles(engine, [first, second]),
                [first, second, third] => Self::of_needles(engine, [first, second, third]),
                [start, .., last] if usize::from(last - start) + 1 == members.len() => {
                    Self::of_range(engine, RangeInclusive { start, last })
                }
                _ => Self::of_small_set(engine, members)
                    .or_else(|| Self::of_fixed_nibble_set(engine, members))
                    .unwrap_or_else(|| Self::of_bitset_lookup(engine, set)),
            }
        } else {
            // Ranges remain cheap even when too large for a named-member kernel.
            match set.as_contiguous_range() {
                Some(range) => Self::of_range(engine, range),
                None => Self::of_bitset_lookup(engine, set),
            }
        }
    }

    fn of_needles<const N: usize>(engine: Engine, needles: [u8; N]) -> Self {
        let kernel_kind = match N {
            1 => KernelKind::OneByte,
            2 => KernelKind::TwoBytes,
            3 => KernelKind::ThreeBytes,
            _ => unreachable!(),
        };
        let mut splatted_needles = [[0; _]; 3];
        for (dst, needle) in splatted_needles.iter_mut().zip(needles) {
            *dst = [needle; _];
        }
        let kernel_data = KernelData { splatted_needles };
        match engine {
            Engine::Vector(level) => Self {
                search: SearchPlan {
                    kernel_data,
                    scan_ops: dispatch!(level, simd => vector::scan_ops::<_, vector::kernels::AnyOf<_, N>>(simd)),
                    engine,
                    kernel_kind,
                },
            },
            Engine::Swar => Self {
                search: SearchPlan {
                    kernel_data,
                    scan_ops: swar::scan_ops::<swar::kernels::AnyOf<N>>(),
                    engine,
                    kernel_kind,
                },
            },
        }
    }

    fn of_range(engine: Engine, range: RangeInclusive<u8>) -> Self {
        let kernel_kind = KernelKind::OneRange;
        match engine {
            Engine::Vector(level) => Self {
                search: SearchPlan {
                    kernel_data: KernelData {
                        splatted_bounds: [[range.start; _], [range.last; _]],
                    },
                    scan_ops: dispatch!(level, simd => vector::scan_ops::<_, vector::kernels::OneRange<_>>(simd)),
                    engine,
                    kernel_kind,
                },
            },
            Engine::Swar => Self {
                search: SearchPlan {
                    kernel_data: KernelData {
                        range_masks: swar::kernels::OneRange::new(range),
                    },
                    scan_ops: swar::scan_ops::<swar::kernels::OneRange>(),
                    engine,
                    kernel_kind,
                },
            },
        }
    }

    fn of_small_set(engine: Engine, possible_set: &[u8]) -> Option<Self> {
        if possible_set.len() > 8 {
            return None;
        }
        let Engine::Vector(level) = engine else {
            return None;
        };
        if !vector::has_byte_shuffle(level) {
            return None;
        }
        let mut lo_lookup = NibbleLookup::default();
        let mut hi_lookup = NibbleLookup::default();
        for (i, &item) in possible_set.iter().enumerate() {
            lo_lookup.set(item & 0x0F, i as u8);
            hi_lookup.set(item >> 4, i as u8);
        }
        Some(Self {
            search: SearchPlan {
                kernel_data: KernelData {
                    nibble_lookups: [lo_lookup, hi_lookup],
                },
                scan_ops: dispatch!(level, simd => vector::scan_ops::<_, vector::kernels::SmallSet>(simd)),
                engine,
                kernel_kind: KernelKind::SmallSet,
            },
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
            search: SearchPlan {
                kernel_data: KernelData { fixed_nibble_table },
                scan_ops: dispatch!(level, simd => vector::scan_ops::<_, vector::kernels::FixedNibbleSet>(simd)),
                engine,
                kernel_kind: KernelKind::FixedNibble,
            },
        })
    }

    fn of_bitset_lookup(engine: Engine, byte_set: ByteSet) -> Self {
        let engine = engine.with_byte_shuffle();
        let (kernel_kind, kernel_data) = (KernelKind::BitsetLookup, KernelData { byte_set });
        match engine {
            Engine::Vector(level) => Self {
                search: SearchPlan {
                    kernel_data,
                    scan_ops: dispatch!(level, simd => vector::scan_ops::<_, BitsetLookup>(simd)),
                    engine,
                    kernel_kind,
                },
            },
            Engine::Swar => Self {
                search: SearchPlan {
                    kernel_data,
                    scan_ops: swar::scan_ops::<BitsetLookup>(),
                    engine,
                    kernel_kind,
                },
            },
        }
    }

    fn of_never() -> Self {
        Self {
            search: SearchPlan {
                kernel_data: KernelData { no_data: () },
                scan_ops: never_scan_ops(),
                engine: Engine::Swar,
                kernel_kind: KernelKind::Never,
            },
        }
    }
}

fn inclusive_range(range: impl RangeBounds<u8>) -> Option<RangeInclusive<u8>> {
    let start = match range.start_bound() {
        Bound::Included(start) => Some(*start),
        Bound::Excluded(start) => start.checked_add(1),
        Bound::Unbounded => Some(u8::MIN),
    };
    let last = match range.end_bound() {
        Bound::Included(last) => Some(*last),
        Bound::Excluded(last) => last.checked_sub(1),
        Bound::Unbounded => Some(u8::MAX),
    };
    let Some(start) = start else {
        return None;
    };
    let Some(last) = last else {
        return None;
    };
    if start > last {
        return None;
    }
    Some(RangeInclusive { start, last })
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
    fn add_range_matches_adding_each_byte() {
        for start in 0..=u8::MAX {
            for last in start..=u8::MAX {
                let mut ranged = ByteSet::new();
                ranged.add_range(RangeInclusive { start, last });

                let mut one_at_a_time = ByteSet::new();
                for byte in start..=last {
                    one_at_a_time.add(byte);
                }

                assert_eq!(ranged, one_at_a_time, "{start}..={last}");
            }
        }
    }

    #[test]
    fn add_range_of_empty_range_adds_nothing() {
        let mut set = ByteSet::from_bytes(b"abc");
        let before = set;
        set.add_range(RangeInclusive { start: 10, last: 9 });
        assert_eq!(set, before);
    }

    fn members(set: &ByteSet) -> Vec<u8> {
        let all: Vec<u8> = (0..=u8::MAX).collect();
        MemchrN::from_set(*set, Backend::Auto)
            .iter(&all)
            .map(|offset| all[offset])
            .collect()
    }

    fn assert_same_set_and_kernel(bulk: &ByteSet, one_at_a_time: &ByteSet, case: &str) {
        // Equal members alone would miss representation-selection regressions.
        assert_eq!(members(bulk), members(one_at_a_time), "{case}");
        for backend in [Backend::Auto, Backend::Swar] {
            assert_eq!(
                format!("{:?}", MemchrN::from_set(*bulk, backend)),
                format!("{:?}", MemchrN::from_set(*one_at_a_time, backend)),
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
                ranged.add_range(RangeInclusive { start, last });

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
    fn add_two_ranges_matches_adding_each_byte() {
        // Includes bitset word boundaries and both ends of the byte domain.
        let bounds = [0u8, 1, 7, 23, 24, 25, 63, 64, 127, 128, 200, 254, 255];
        for &first_start in &bounds {
            for &first_last in bounds.iter().filter(|&&b| b >= first_start) {
                for &second_start in &bounds {
                    for &second_last in bounds.iter().filter(|&&b| b >= second_start) {
                        let mut ranged = ByteSet::new();
                        ranged.add_range(RangeInclusive {
                            start: first_start,
                            last: first_last,
                        });
                        ranged.add_range(RangeInclusive {
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
        set.add_range(RangeInclusive {
            start: 0,
            last: 100,
        });
        set.add(200);
        assert_eq!(members(&set), (0..=100).chain([200]).collect::<Vec<u8>>());
    }

    #[test]
    fn add_range_works_in_const_context() {
        const DIGITS: ByteSet = {
            let mut set = ByteSet::new();
            set.add_range(RangeInclusive {
                start: b'0',
                last: b'9',
            });
            set
        };
        assert_eq!(members(&DIGITS), b"0123456789");
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

    fn sets() -> Vec<Vec<u8>> {
        vec![
            vec![],
            b"z".to_vec(),
            b"az".to_vec(),
            b"azQ".to_vec(),
            b"aeiouAEI".to_vec(),
            b"abcdefghjl".to_vec(),
            b"0123456789abcdef".to_vec(),
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
}
