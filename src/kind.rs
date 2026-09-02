use crate::bitset::Bitset;
use core::range::RangeInclusive;

/// The most bytes any [`Kind`] names one at a time. Past this a set is only ever a range or
/// a bitset, neither of which needs its members listed.
const MEMBERS_MAX: usize = 16;

/// Which kernel a byte set resolves to, with what that kernel matches on.
///
/// Everything above the kernels turns on this: [`Kind::of`] is where a set of bytes stops
/// being a set and becomes a choice of scan.
#[derive(Copy, Clone, Debug)]
pub(crate) enum Kind {
    AnyByte(Bitset),
    SmallSet {
        lo_lookup: NibbleLookup,
        hi_lookup: NibbleLookup,
    },
    ConstantNibble(ConstantNibble, [u8; 16]),
    OneByte(u8),
    TwoBytes([u8; 2]),
    ThreeBytes([u8; 3]),
    OneRange(RangeInclusive<u8>),
    Never,
}

/// Which [`Kind`] a [`MemchrN`](crate::MemchrN) was built from, less the payload that
/// [`KernelData`](crate::KernelData) holds in the shape its kernel wants.
///
/// The payload cannot be recovered from that in every case — `swar`'s `OneRange` keeps only
/// the low seven bits of its start — so naming the kind is as much as [`Debug`] can offer
/// without carrying a second copy of it. This costs nothing: it fits in the padding the
/// alignment of [`KernelData`](crate::KernelData) leaves behind.
#[derive(Copy, Clone, Debug)]
pub(crate) enum KindTag {
    AnyByte,
    SmallSet,
    ConstantNibble,
    OneByte,
    TwoBytes,
    ThreeBytes,
    OneRange,
    Never,
}

impl KindTag {
    pub(crate) fn of(kind: Kind) -> Self {
        match kind {
            Kind::AnyByte(_) => Self::AnyByte,
            Kind::SmallSet { .. } => Self::SmallSet,
            Kind::ConstantNibble(..) => Self::ConstantNibble,
            Kind::OneByte(_) => Self::OneByte,
            Kind::TwoBytes(_) => Self::TwoBytes,
            Kind::ThreeBytes(_) => Self::ThreeBytes,
            Kind::OneRange(_) => Self::OneRange,
            Kind::Never => Self::Never,
        }
    }
}

impl Kind {
    /// Picks the kind that matches `set` most cheaply.
    ///
    /// `fast_shuffles` gates the kinds a kernel can only scan by shuffling bytes within a
    /// vector: unset for a target whose shuffles are slow, and for the word kernels, which
    /// have none.
    ///
    /// The order is by preference, not by cost to test: the earlier arms match fewer sets
    /// but scan faster, so a set that reaches one has already been ruled out of everything
    /// above it.
    pub(crate) fn of(set: &Bitset, fast_shuffles: bool) -> Self {
        let mut members = [0; MEMBERS_MAX];
        let Some(count) = set.members(&mut members) else {
            // Too many members for any kind that names them. A range is still worth
            // recognising: it is two comparisons per byte however wide it is.
            return match set.extract_range() {
                Some(range) => Self::OneRange(range),
                None => Self::AnyByte(*set),
            };
        };
        let members = &members[..usize::from(count)];

        match *members {
            [] => Self::Never,
            [item] => Self::OneByte(item),
            [first, last] => Self::TwoBytes([first, last]),
            [first, second, third] => Self::ThreeBytes([first, second, third]),
            // `members` is ascending and distinct, so the set is contiguous exactly when it
            // fills the span from its first to its last.
            [first, .., last] if usize::from(last - first) + 1 == members.len() => {
                Self::OneRange(RangeInclusive { start: first, last })
            }
            _ if fast_shuffles && members.len() <= 8 => {
                let mut lo_lookup = NibbleLookup::default();
                let mut hi_lookup = NibbleLookup::default();
                for (i, &item) in members.iter().enumerate() {
                    lo_lookup.set(item & 0x0F, i as u8);
                    hi_lookup.set(item >> 4, i as u8);
                }
                Self::SmallSet {
                    lo_lookup,
                    hi_lookup,
                }
            }
            _ if fast_shuffles
                && let Some((nibble, lookup)) = constant_nibble(members) =>
            {
                Self::ConstantNibble(nibble, lookup)
            }
            _ => Self::AnyByte(*set),
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
