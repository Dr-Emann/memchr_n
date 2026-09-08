#![deny(unnameable_types, unreachable_pub)]

mod bitset;
mod bytewise;
mod kind;
mod swar;
mod vector;

use crate::bitset::Bitset;
use crate::kind::Kind;
use core::fmt;
use core::range::RangeInclusive;
use fearless_simd::dispatch;

#[cfg(feature = "manual_level")]
pub use fearless_simd::Level;
#[cfg(not(feature = "manual_level"))]
use fearless_simd::Level;

// Matches of one scan, the `i`th bit (numbered from lsb to msb) is 1 if the `i`th byte matched
type MatchedBitset = u128;

/// Which family of kernels a [`MemchrN`] is built from.
#[derive(Copy, Clone, Default, Debug)]
#[non_exhaustive]
pub enum Backend {
    /// The best kernels the running CPU supports.
    #[default]
    Auto,
    /// Word-at-a-time kernels, even on a target that has vectors.
    Scalar,
    /// An explicit [`Level`]
    #[cfg(feature = "manual_level")]
    Level(Level),
}

impl Backend {
    fn family(self) -> Family {
        match self {
            Backend::Auto => Family::Vector(Level::new()),
            Backend::Scalar => Family::Scalar,
            #[cfg(feature = "manual_level")]
            Backend::Level(level) => Family::Vector(level),
        }
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    // Part of the public contract, and cheap to keep from regressing.
    assert_send_sync::<MemchrN>();
};

/// A searcher for a fixed set of bytes.
#[derive(Clone)]
pub struct MemchrN {
    // `family` and `kind`, only for `Debug`; `data` and `scan` are what
    // searching goes through.
    family: Family,
    kind: Kind,
    data: KernelData,
    scan: &'static Scan,
}

impl fmt::Debug for MemchrN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemchrN")
            .field("family", &self.family)
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

#[derive(Copy, Clone, Debug)]
enum Family {
    Vector(Level),
    Scalar,
}

impl Family {
    // Get a family that would be used if we want to use shuffles
    #[must_use]
    fn for_shuffle(self) -> Self {
        match self {
            Family::Vector(level) => {
                if vector::has_byte_shuffle(level) {
                    Family::Vector(level)
                } else {
                    Family::Scalar
                }
            }
            Family::Scalar => Family::Scalar,
        }
    }
}

impl MemchrN {
    /// Build a searcher for the distinct bytes of `bytes`, on the best kernels the running
    /// CPU supports.
    #[inline]
    pub fn new(bytes: &[u8]) -> Self {
        Self::new_with(bytes, Backend::Auto)
    }

    /// [`new`](Self::new), on a chosen [`Backend`].
    pub fn new_with(bytes: &[u8], backend: Backend) -> Self {
        Self::from_set(Bitset::from_bytes(bytes), backend)
    }

    /// Builds a searcher which will match the bytes within the provided range
    ///
    /// An empty range matches nothing.
    #[inline]
    pub fn from_range(range: core::ops::RangeInclusive<u8>) -> Self {
        Self::from_range_with(range, Backend::Auto)
    }

    /// [`from_range`](Self::from_range), on a chosen [`Backend`].
    pub fn from_range_with(range: core::ops::RangeInclusive<u8>, backend: Backend) -> Self {
        let mut set = Bitset::new();
        set.add_range(RangeInclusive {
            start: *range.start(),
            last: *range.end(),
        });
        Self::from_set(set, backend)
    }

    fn from_set(set: Bitset, backend: Backend) -> Self {
        // Enough for all items to share a nibble
        const MEMBERS_MAX: usize = 16;

        let family = backend.family();
        let mut members = [0; MEMBERS_MAX];
        if let Some(count) = set.members(&mut members) {
            let members = &members[..usize::from(count)];

            match *members {
                [] => Self::of_never(),
                [first] => Self::of_needles(family, [first]),
                [first, second] => Self::of_needles(family, [first, second]),
                [first, second, third] => Self::of_needles(family, [first, second, third]),
                // `members` is ascending and distinct, so the set is contiguous exactly when it
                // fills the span from its first to its last.
                [start, .., last] if usize::from(last - start) + 1 == members.len() => {
                    Self::of_range(family, RangeInclusive { start, last })
                }
                _ => Self::of_small_set(family, members)
                    .or_else(|| Self::of_single_nibble(family, members))
                    .unwrap_or_else(|| Self::of_any_byte(family, set)),
            }
        } else {
            // Too many members for any kind that names them. A range is still worth
            // recognizing: it is two comparisons per byte however wide it is.
            match set.extract_range() {
                Some(range) => Self::of_range(family, range),
                None => Self::of_any_byte(family, set),
            }
        }
    }

    fn of_needles<const N: usize>(family: Family, needles: [u8; N]) -> Self {
        let kind = match N {
            1 => Kind::OneByte,
            2 => Kind::TwoBytes,
            3 => Kind::ThreeBytes,
            _ => unreachable!(),
        };
        let mut splatted_needles = [[0; _]; 3];
        for (dst, needle) in splatted_needles.iter_mut().zip(needles) {
            *dst = [needle; _];
        }
        let data = KernelData { splatted_needles };
        match family {
            Family::Vector(level) => Self {
                family,
                kind,
                data,
                scan: dispatch!(level, simd => vector::scan::<_, vector::kernels::AnyOf<_, N>>(simd)),
            },
            Family::Scalar => Self {
                family,
                kind,
                data,
                scan: swar::scan::<swar::kernels::AnyOf<N>>(),
            },
        }
    }

    fn of_range(family: Family, range: RangeInclusive<u8>) -> Self {
        let kind = Kind::OneRange;
        match family {
            Family::Vector(level) => Self {
                family,
                kind,
                data: KernelData {
                    splatted_range: [[range.start; _], [range.last; _]],
                },
                scan: dispatch!(level, simd => vector::scan::<_, vector::kernels::OneRange<_>>(simd)),
            },
            Family::Scalar => Self {
                family,
                kind,
                data: KernelData {
                    range_masks: swar::kernels::OneRange::new(range),
                },
                scan: swar::scan::<swar::kernels::OneRange>(),
            },
        }
    }

    fn of_small_set(family: Family, possible_set: &[u8]) -> Option<Self> {
        if possible_set.len() > 8 {
            return None;
        }
        let Family::Vector(level) = family else {
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
            family,
            kind: Kind::SmallSet,
            data: KernelData {
                nibble_lookups: [lo_lookup, hi_lookup],
            },
            scan: dispatch!(level, simd => vector::scan::<_, vector::kernels::SmallSet>(simd)),
        })
    }

    fn of_single_nibble(family: Family, possible_set: &[u8]) -> Option<Self> {
        let Family::Vector(level) = family else {
            return None;
        };
        if !vector::has_byte_shuffle(level) {
            return None;
        }
        let nibble_table = extract_constant_nibble(possible_set)?;
        Some(Self {
            family,
            kind: Kind::ConstantNibble,
            data: KernelData { nibble_table },
            scan: dispatch!(level, simd => vector::scan::<_, vector::kernels::SingleNibble>(simd)),
        })
    }

    fn of_any_byte(family: Family, bitset: Bitset) -> Self {
        let family = family.for_shuffle();
        let (kind, data) = (Kind::AnyByte, KernelData { bitset });
        match family {
            Family::Vector(level) => Self {
                family,
                kind,
                data,
                scan: dispatch!(level, simd => vector::scan::<_, vector::kernels::AnyByte>(simd)),
            },
            Family::Scalar => Self {
                family,
                kind,
                data,
                scan: bytewise::scan::<bytewise::kernels::AnyByte>(),
            },
        }
    }

    fn of_never() -> Self {
        Self {
            family: Family::Scalar,
            kind: Kind::Never,
            data: KernelData { never: () },
            scan: never_scan(),
        }
    }

    /// Returns the offset of the first matching byte in `haystack`.
    ///
    /// Prefer [`iter`](Self::iter) rather than calling this function repeatedly
    /// to iterate over the instances of matching bytes.
    #[inline]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        // SAFETY: as in `Iter::refill`.
        unsafe { (self.scan.find_first)(&self.data, haystack) }
    }

    /// Returns an iterator over the offsets of every matching byte in `haystack`.
    ///
    /// If only iterating a single next instance, prefer [`find`](Self::find)
    #[inline]
    pub fn iter<'a>(&'a self, haystack: &'a [u8]) -> Iter<'a> {
        Iter {
            memchr_n: self,
            state: IterState {
                haystack,
                pos: 0,
                bits_offset: 0,
            },
            bits: 0,
        }
    }
}

impl FromIterator<u8> for MemchrN {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        Self::from_set(Bitset::from_iter(iter), Backend::Auto)
    }
}

// If `items` contains only items which share a constant lo/hi nibble, extract it, and
// a lookup table for the other nibble
fn extract_constant_nibble(items: &[u8]) -> Option<NibbleTable> {
    let first = *items.first()?;
    let (lo_nibble, hi_nibble) = (first & 0x0F, first >> 4);

    let (mut lo_constant, mut hi_constant) = (true, true);
    for &item in items {
        lo_constant &= item & 0x0F == lo_nibble;
        hi_constant &= item >> 4 == hi_nibble;
    }

    // Unfilled slots need a sentinel that can never match: slot `i` is only ever compared
    // against bytes whose variable nibble is `i`, so the sentinel's own variable nibble must
    // differ from its index. 0x00 satisfies that everywhere except slot 0, hence the one
    // filled slot below — the variable nibble is the high one for a constant low nibble, and
    // the low one for a constant high one.
    if lo_constant {
        let mut table = [0; 16];
        table[0] = 0x10;
        for &item in items {
            table[usize::from(item >> 4)] = item;
        }
        Some(NibbleTable {
            which: ConstantNibble::Lo,
            table,
        })
    } else if hi_constant {
        let mut table = [0; 16];
        table[0] = 0x01;
        for &item in items {
            table[usize::from(item & 0x0F)] = item;
        }
        Some(NibbleTable {
            which: ConstantNibble::Hi,
            table,
        })
    } else {
        None
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum ConstantNibble {
    Lo,
    Hi,
}

#[derive(Debug, Default, Copy, Clone)]
pub(crate) struct NibbleLookup(pub(crate) [u8; 16]);

impl NibbleLookup {
    #[inline]
    fn set(&mut self, nibble: u8, bit: u8) {
        debug_assert!(nibble < 16);
        debug_assert!(bit < 8);
        self.0[usize::from(nibble)] |= 1 << bit;
    }
}

pub struct Iter<'a> {
    memchr_n: &'a MemchrN,
    state: IterState<'a>,
    // Matches of the most recently scanned run that have not been yielded yet.
    bits: MatchedBitset,
}

// What a scan reads and writes, which is everything about the search but its matches.
struct IterState<'a> {
    haystack: &'a [u8],
    // Offset of the first byte that has not been scanned yet.
    pos: usize,
    // Offset of the first byte the most recent scan's bits describe.
    bits_offset: usize,
}

/// Everything a kernel needs, in the shape that kernel reads it, built once when the
/// [`MemchrN`] is.
///
/// This is effectively the data half of a manual implementation of a dyn trait,
/// but we don't need any dynamic allocations.
/// The data will be read in the [`Scan`] implementation.
#[derive(Copy, Clone)]
#[repr(align(16))]
union KernelData {
    /// [`vector::kernels::AnyOf`]: one to three needles, each splatted across a block.
    splatted_needles: [[u8; 16]; 3],
    /// [`vector::kernels::OneRange`]: the endpoints, each splatted across a block.
    splatted_range: [[u8; 16]; 2],
    /// [`vector::kernels::SmallSet`]: the low- and high-nibble tables.
    nibble_lookups: [NibbleLookup; 2],
    /// [`vector::kernels::SingleNibble`].
    nibble_table: NibbleTable,
    /// [`vector::kernels::AnyByte`] and [`bytewise::kernels::AnyByte`].
    bitset: Bitset,
    /// [`swar::kernels::OneRange`], whose masks are all derived up front.
    range_masks: swar::kernels::OneRange,
    /// never has no data
    never: (),
}

/// [`vector::kernels::SingleNibble`]'s table and the nibble it is indexed by.
#[derive(Copy, Clone)]
struct NibbleTable {
    which: ConstantNibble,
    table: [u8; 16],
}

#[derive(Copy, Clone)]
struct Scan {
    /// Search forward until the first non-zero matching bitset
    ///
    /// Modifies the passed [`IterState`] to the new pos/bits_offset.
    ///
    /// # Safety
    /// Callers must call with the matching KernelData that this [`Scan`] was created for
    find_next: unsafe fn(&KernelData, &mut IterState<'_>) -> MatchedBitset,
    /// Counts every match in what is left of a haystack.
    ///
    /// Takes that remainder rather than the [`IterState`] it comes from: counting reads the
    /// haystack once and never resumes, so a scan has nothing to write back, and the state
    /// need not go to memory across the call the way [`find_next`](Scan::find_next)'s does.
    count_all: unsafe fn(&KernelData, &[u8]) -> usize,
    /// [`MemchrN::find`]'s whole search, rather than the first refill of an iteration.
    ///
    /// Both of the above are shaped for an iterator that will call them again: they take the
    /// [`IterState`] by pointer, which forces it to memory across the call, and they answer
    /// in a [`MatchedBitset`] the caller has to unpack.
    find_first: unsafe fn(&KernelData, &[u8]) -> Option<usize>,
}

// The Scan for a byte set that nothing can match.
pub(crate) fn never_scan() -> &'static Scan {
    fn find_next(_data: &KernelData, state: &mut IterState<'_>) -> MatchedBitset {
        state.pos = state.haystack.len();
        0
    }

    fn count_all(_data: &KernelData, _haystack: &[u8]) -> usize {
        0
    }

    fn find_first(_data: &KernelData, _haystack: &[u8]) -> Option<usize> {
        None
    }

    &Scan {
        find_next,
        count_all,
        find_first,
    }
}

impl<'a> Iter<'a> {
    // Scans on from `pos` until a run that matched, leaving its bits in `bits`.
    //
    // `None` says the haystack is spent: a scan that finds nothing runs to the end of it,
    // so no bits and no error are the same answer.
    #[inline]
    fn refill(&mut self) -> Option<()> {
        if self.state.pos == self.state.haystack.len() {
            return None;
        }
        // SAFETY: each `build` installs a scan only for the kind whose `KernelData` field
        // its kernel reads, and the `Level` that chose it proves the target has its features.
        self.bits = unsafe { (self.memchr_n.scan.find_next)(&self.memchr_n.data, &mut self.state) };
        (self.bits != 0).then_some(())
    }

    // Takes the lowest match out of `bits`, which must hold one.
    #[inline]
    fn take_lowest(&mut self) -> usize {
        debug_assert!(self.bits != 0);

        let bit = self.bits.trailing_zeros() as usize;
        self.bits &= self.bits - 1;
        self.state.bits_offset + bit
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.bits == 0 {
            self.refill()?;
        }
        Some(self.take_lowest())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let min = self.bits.count_ones() as usize;
        let max = min.checked_add(self.state.haystack.len() - self.state.pos);
        (min, max)
    }

    fn count(self) -> usize {
        let mut total = self.bits.count_ones() as usize;
        // SAFETY: `pos` only ever moves to an offset a scan reached, so it is in bounds.
        let unscanned = unsafe { self.state.haystack.get_unchecked(self.state.pos..) };
        if !unscanned.is_empty() {
            // SAFETY: as in `refill`.
            total += unsafe { (self.memchr_n.scan.count_all)(&self.memchr_n.data, unscanned) };
        }
        total
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let mut remaining = n;
        loop {
            let held = self.bits.count_ones() as usize;
            if held > remaining {
                break;
            }
            remaining -= held;
            self.bits = 0;
            self.refill()?;
        }
        for _ in 0..remaining {
            self.bits &= self.bits - 1;
        }
        Some(self.take_lowest())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `MemchrN` is built per search often enough that its size is worth keeping honest,
    /// and one cache line is the round number to hold it to. `KernelData` is three quarters
    /// of that, and the rest fits in the padding its alignment leaves behind.
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
    fn add_range_matches_adding_each_byte() {
        for start in 0..=u8::MAX {
            for last in start..=u8::MAX {
                let mut ranged = Bitset::new();
                ranged.add_range(RangeInclusive { start, last });

                let mut one_at_a_time = Bitset::new();
                for byte in start..=last {
                    one_at_a_time.add(byte);
                }

                assert_eq!(ranged, one_at_a_time, "{start}..={last}");
            }
        }
    }

    #[test]
    fn add_range_of_empty_range_adds_nothing() {
        let mut set = Bitset::from_bytes(b"abc");
        let before = set;
        set.add_range(RangeInclusive { start: 10, last: 9 });
        assert_eq!(set, before);
    }

    /// Every byte in `set`, recovered by searching a haystack of all 256 byte values.
    fn members(set: &Bitset) -> Vec<u8> {
        let all: Vec<u8> = (0..=u8::MAX).collect();
        MemchrN::from_set(*set, Backend::Auto)
            .iter(&all)
            .map(|offset| all[offset])
            .collect()
    }

    /// Both halves must agree on the kernel, not just the set: a bitset that covers a span
    /// exactly still has to reach [`Kind::OneRange`], and few enough distinct bytes still
    /// have to reach the kinds that name their members.
    fn assert_same_set_and_kernel(bulk: &Bitset, one_at_a_time: &Bitset, case: &str) {
        assert_eq!(members(bulk), members(one_at_a_time), "{case}");
        for backend in [Backend::Auto, Backend::Scalar] {
            assert_eq!(
                format!("{:?}", MemchrN::from_set(*bulk, backend)),
                format!("{:?}", MemchrN::from_set(*one_at_a_time, backend)),
                "{case} on {backend:?}"
            );
        }
    }

    /// `from_bytes` collects a whole slice at once, and has to land where the byte-at-a-time
    /// insert would, at every size and shape of set.
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
            ("one past the members a kind can name", (0..17).collect()),
        ];

        for (case, bytes) in cases {
            let mut one_at_a_time = Bitset::new();
            for &byte in bytes {
                one_at_a_time.add(byte);
            }
            assert_same_set_and_kernel(&Bitset::from_bytes(bytes), &one_at_a_time, case);
        }
    }

    /// The same, for the range fast path taken when the array already holds something.
    #[test]
    fn add_wide_range_to_non_empty_matches_adding_each_byte() {
        let seeds: &[&[u8]] = &[b"", b"z", b"\x00", b"\x7f", b"az", b"\x00\xff", b"aeiouAEI"];
        for seed in seeds {
            for (start, last) in [(0u8, 255u8), (0x80, 0xFF), (10, 40), (100, 124), (60, 200)] {
                let mut ranged = Bitset::from_bytes(seed);
                ranged.add_range(RangeInclusive { start, last });

                let mut one_at_a_time = Bitset::from_bytes(seed);
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
        // Bounds that straddle every representation change: the array filling up, a bitset
        // word boundary, and the ends of the byte range.
        let bounds = [0u8, 1, 7, 23, 24, 25, 63, 64, 127, 128, 200, 254, 255];
        for &first_start in &bounds {
            for &first_last in bounds.iter().filter(|&&b| b >= first_start) {
                for &second_start in &bounds {
                    for &second_last in bounds.iter().filter(|&&b| b >= second_start) {
                        let mut ranged = Bitset::new();
                        ranged.add_range(RangeInclusive {
                            start: first_start,
                            last: first_last,
                        });
                        ranged.add_range(RangeInclusive {
                            start: second_start,
                            last: second_last,
                        });

                        let mut one_at_a_time = Bitset::new();
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
        let mut set = Bitset::new();
        set.add_range(RangeInclusive {
            start: 0,
            last: 100,
        });
        set.add(200);
        assert_eq!(members(&set), (0..=100).chain([200]).collect::<Vec<u8>>());
    }

    #[test]
    fn add_range_works_in_const_context() {
        const DIGITS: Bitset = {
            let mut set = Bitset::new();
            set.add_range(RangeInclusive {
                start: b'0',
                last: b'9',
            });
            set
        };
        assert_eq!(members(&DIGITS), b"0123456789");
    }

    fn build_word(bytes: &[u8]) -> MemchrN {
        MemchrN::new_with(bytes, Backend::Scalar)
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

    /// One byte set per [`Kind`], so every kernel is exercised.
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
        // A repeating byte pattern that hits every kernel's table entries, with the
        // period chosen so it does not line up with the 64-byte chunking.
        (0..len).map(|i| ((i * 37 + i / 7) % 251) as u8).collect()
    }

    #[test]
    fn matches_naive_across_lengths() {
        for set in sets() {
            for (name, searcher) in [("vector", build(&set)), ("word", build_word(&set))] {
                // Every length up to a chunk-and-change, so each way the tail can split
                // into whole words and a short remainder is covered, then the pair and
                // multi-chunk boundaries.
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
                // Every length up to a chunk-and-change, so each way the tail can split
                // into whole words and a short remainder is covered, then the pair and
                // multi-chunk boundaries.
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

    /// Both wide families scan the tail by re-reading the last whole unit they work in, so
    /// bits belonging to offsets the main loop already reported have to be shifted off, and
    /// the bytewise family instead has to advance past exactly the run it reported. A
    /// haystack that matches at every offset catches either going wrong.
    #[test]
    fn overlapping_tail_does_not_repeat_matches() {
        // The third set is large enough to reach `AnyByte`, and contains `x`.
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

    /// `find` covers a sub-chunk haystack with two reads that overlap in the middle, so the
    /// offset it reports depends on which of them saw the match and how far back the second
    /// one started. Every length up to a chunk-and-change, with the one match walked across
    /// every offset, is what pins that arithmetic down.
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

    /// Every kernel answers a short haystack a byte at a time now, through a scalar matcher
    /// written beside its wide one. The two are separate pieces of arithmetic over the same
    /// data — a table index against a shuffle, a compare against a splat — so they have to be
    /// held to every byte, not just the ones a shared test haystack happens to contain.
    #[test]
    fn per_byte_matcher_agrees_with_the_set() {
        for set in sets() {
            // Something to pad with that the set does not hold, so only the planted byte can
            // answer. Every set here leaves at least one byte over.
            let pad = (0..=u8::MAX)
                .find(|byte| !set.contains(byte))
                .expect("no set here holds every byte");

            for (name, searcher) in [("vector", build(&set)), ("word", build_word(&set))] {
                for byte in 0..=u8::MAX {
                    // Lengths either side of the scalar probe and of the staged pair's own
                    // ladder, with the byte walked across each so a probe that stops early
                    // and a stage that starts late both show up.
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

    /// `nth` has to consume the same matches `next` would, so mixing the two must walk the
    /// haystack exactly once.
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

    /// Both halves of a scanned pair are reported in one set of bits, the upper half
    /// shifted up by [`CHUNK_BYTES`]. A lone match walked across the pair boundary catches a
    /// half packed at the wrong end.
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
                    // `find` walks the pair itself rather than unpacking the iterator's
                    // bits, so it has to pick the right half of one on its own.
                    assert_eq!(
                        searcher.find(&haystack),
                        Some(offset),
                        "find, len {len} offset {offset}"
                    );
                }
            }
        }
    }

    /// `find` duplicates the walk `Iter` does rather than driving it, so the two have to
    /// agree everywhere — including on the tails, where each family reads bytes it has
    /// already reported on.
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

    /// The counting accumulator holds one byte per lane, so it must be drained before
    /// a lane can wrap.
    #[test]
    fn counts_do_not_overflow_the_accumulator() {
        let searcher = build(b"x");
        let len = 64 * (512 + 3);
        let haystack = vec![b'x'; len];
        assert_eq!(searcher.iter(&haystack).count(), len);
    }
}
