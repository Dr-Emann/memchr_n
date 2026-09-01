use crate::bitset::Bitset;
use crate::{ConstantNibble, Kind, NibbleLookup};
use core::range::RangeInclusive;

/// The most bytes any [`Kind`] names one at a time. Past this a set is only ever a range or
/// a bitset, neither of which needs its members listed.
const MEMBERS_MAX: usize = 16;

/// A set of bytes, on the way to becoming a [`Kind`].
///
/// Only a building block: [`MemchrN`](crate::MemchrN) collects a caller's bytes into one of
/// these, asks it for a [`Kind`], and drops it.
///
/// A bitset is the whole representation. It is what [`Kind::AnyByte`] wants as it stands,
/// what a range check reads, and what makes collection a probe per byte with no branch on
/// the answer — and the kinds that do want their bytes listed want at most sixteen of them,
/// which [`Bitset::members`] walks out in one pass over the set bits.
#[derive(Debug, Copy, Clone)]
pub(crate) struct ByteSet(Bitset);

impl ByteSet {
    pub(crate) const fn new() -> Self {
        Self(Bitset::new())
    }

    pub(crate) const fn from_bytes(bytes: &[u8]) -> Self {
        Self(Bitset::from_bytes(bytes))
    }

    // Intentionally take a ops::RangeInclusive rather than a range::RangeInclusive.
    // Its worth it to support `..=` syntax
    pub(crate) const fn from_range(range: core::ops::RangeInclusive<u8>) -> Self {
        let mut res = Self::new();
        res.add_range(RangeInclusive {
            start: *range.start(),
            last: *range.end(),
        });
        res
    }

    pub(crate) const fn add(&mut self, item: u8) {
        self.0.add(item);
    }

    /// Adds every byte from `range.start` through `range.last`, inclusive.
    ///
    /// An empty range (one whose `start` is past its `last`) adds nothing.
    pub(crate) const fn add_range(&mut self, range: RangeInclusive<u8>) {
        self.0.add_range(range);
    }

    /// Picks the kind that matches this set most cheaply.
    ///
    /// `fast_shuffles` gates the kinds a kernel can only scan by shuffling bytes within a
    /// vector: unset for a target whose shuffles are slow, and for the word kernels, which
    /// have none.
    ///
    /// The order is by preference, not by cost to test: the earlier arms match fewer sets
    /// but scan faster, so a set that reaches one has already been ruled out of everything
    /// above it.
    pub(crate) fn kind(&self, fast_shuffles: bool) -> Kind {
        let mut members = [0; MEMBERS_MAX];
        let Some(count) = self.0.members(&mut members) else {
            // Too many members for any kind that names them. A range is still worth
            // recognising: it is two comparisons per byte however wide it is.
            return match self.0.extract_range() {
                Some(range) => Kind::OneRange(range),
                None => Kind::AnyByte(self.0),
            };
        };
        let members = &members[..usize::from(count)];

        match *members {
            [] => Kind::Never,
            [item] => Kind::OneByte(item),
            [first, last] => Kind::TwoBytes([first, last]),
            [first, second, third] => Kind::ThreeBytes([first, second, third]),
            // `members` is ascending and distinct, so the set is contiguous exactly when it
            // fills the span from its first to its last.
            [first, .., last] if usize::from(last - first) + 1 == members.len() => {
                Kind::OneRange(RangeInclusive { start: first, last })
            }
            _ if fast_shuffles && members.len() <= 8 => {
                let mut lo_lookup = NibbleLookup::default();
                let mut hi_lookup = NibbleLookup::default();
                for (i, &item) in members.iter().enumerate() {
                    lo_lookup.set(item & 0x0F, i as u8);
                    hi_lookup.set(item >> 4, i as u8);
                }
                Kind::SmallSet {
                    lo_lookup,
                    hi_lookup,
                }
            }
            _ if fast_shuffles
                && let Some((nibble, lookup)) = constant_nibble(members) =>
            {
                Kind::ConstantNibble(nibble, lookup)
            }
            _ => Kind::AnyByte(self.0),
        }
    }
}

/// The table [`crate::vector::kernels::SingleNibble`] shuffles, if every byte of `items`
/// agrees on one of its nibbles.
///
/// Deciding costs one pass with no branch on the answer, and only the winning table is then
/// filled — testing and building together, as this used to, meant carrying two half-built
/// tables through the loop and a branch per item per table.
fn constant_nibble(items: &[u8]) -> Option<(ConstantNibble, [u8; 16])> {
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
        Some((ConstantNibble::Lo, table))
    } else if hi_constant {
        let mut table = [0; 16];
        table[0] = 0x01;
        for &item in items {
            table[usize::from(item & 0x0F)] = item;
        }
        Some((ConstantNibble::Hi, table))
    } else {
        None
    }
}

impl FromIterator<u8> for ByteSet {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        let mut res = Self::new();
        res.extend(iter);
        res
    }
}

impl Extend<u8> for ByteSet {
    fn extend<T: IntoIterator<Item = u8>>(&mut self, iter: T) {
        iter.into_iter().for_each(|item| self.add(item));
    }
}
