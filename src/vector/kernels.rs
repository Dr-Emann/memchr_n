use super::Kernel;
use crate::bitset::Bitset;
use crate::{ConstantNibble, FinderKind, NibbleLookup};
use fearless_simd::prelude::*;
use fearless_simd::{mask8x64, u8x16, u8x32, u8x64};
use std::range::RangeInclusive;

/// Compares against each needle in turn, which beats a table lookup while the set is
/// small enough that the comparisons stay cheaper than the lookup they replace.
#[derive(Copy, Clone)]
pub(crate) struct AnyOf<const N: usize> {
    needles: [u8; N],
}

#[inline(always)]
fn any_of_matches<S: Simd, const N: usize>(chunk: u8x64<S>, needles: [u8; N]) -> mask8x64<S> {
    let mut matched = mask8x64::splat(chunk.simd, false);
    for &needle in &needles {
        matched |= chunk.simd_eq(needle);
    }
    matched
}

impl Kernel for AnyOf<1> {
    fn from_kind(kind: &FinderKind) -> Option<Self> {
        let FinderKind::OneByte(needle) = *kind else {
            return None;
        };
        Some(Self { needles: [needle] })
    }

    #[inline(always)]
    fn matches<S: Simd>(&self, chunk: u8x64<S>) -> mask8x64<S> {
        any_of_matches(chunk, self.needles)
    }
}

impl Kernel for AnyOf<2> {
    fn from_kind(kind: &FinderKind) -> Option<Self> {
        let FinderKind::TwoBytes(needles) = *kind else {
            return None;
        };
        Some(Self { needles })
    }

    #[inline(always)]
    fn matches<S: Simd>(&self, chunk: u8x64<S>) -> mask8x64<S> {
        any_of_matches(chunk, self.needles)
    }
}

impl Kernel for AnyOf<3> {
    fn from_kind(kind: &FinderKind) -> Option<Self> {
        let FinderKind::ThreeBytes(needles) = *kind else {
            return None;
        };
        Some(Self { needles })
    }

    #[inline(always)]
    fn matches<S: Simd>(&self, chunk: u8x64<S>) -> mask8x64<S> {
        any_of_matches(chunk, self.needles)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct OneRange {
    range: RangeInclusive<u8>,
}

impl Kernel for OneRange {
    fn from_kind(kind: &FinderKind) -> Option<Self> {
        let FinderKind::OneRange(range) = *kind else {
            return None;
        };
        Some(Self { range })
    }
    #[inline(always)]
    fn matches<S: Simd>(&self, chunk: u8x64<S>) -> mask8x64<S> {
        let RangeInclusive { start, last } = self.range;
        chunk.simd_ge(start) & chunk.simd_le(last)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct SmallSet {
    lo_lookup: NibbleLookup,
    hi_lookup: NibbleLookup,
}

impl Kernel for SmallSet {
    fn from_kind(kind: &FinderKind) -> Option<Self> {
        let FinderKind::SmallSet {
            lo_lookup,
            hi_lookup,
        } = *kind
        else {
            return None;
        };
        Some(Self {
            lo_lookup,
            hi_lookup,
        })
    }

    #[inline(always)]
    fn matches<S: Simd>(&self, chunk: u8x64<S>) -> mask8x64<S> {
        let simd = chunk.simd;
        let lo_lookup = u8x64::block_splat(u8x16::load_array(simd, self.lo_lookup.0));
        let hi_lookup = u8x64::block_splat(u8x16::load_array(simd, self.hi_lookup.0));

        let lo = lo_lookup.swizzle_dyn_within_blocks(chunk & 0x0F);
        let hi = hi_lookup.swizzle_dyn_within_blocks(chunk >> 4);

        !(lo & hi).simd_eq(0)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct SingleNibble {
    which: ConstantNibble,
    table: [u8; 16],
}

impl Kernel for SingleNibble {
    fn from_kind(kind: &FinderKind) -> Option<Self> {
        let FinderKind::ConstantNibble(which, table) = *kind else {
            return None;
        };
        Some(Self { which, table })
    }

    #[inline(always)]
    fn matches<S: Simd>(&self, chunk: u8x64<S>) -> mask8x64<S> {
        let table = u8x64::block_splat(u8x16::simd_from(chunk.simd, self.table));
        let non_const_nibbles = match self.which {
            ConstantNibble::Lo => chunk >> 4,
            ConstantNibble::Hi => chunk & 0x0F,
        };
        let should_match = table.swizzle_dyn_within_blocks(non_const_nibbles);
        chunk.simd_eq(should_match)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct AnyByte {
    bitset: Bitset,
}

impl Kernel for AnyByte {
    fn from_kind(kind: &FinderKind) -> Option<Self> {
        let FinderKind::AnyByte(bitset) = *kind else {
            return None;
        };
        Some(Self { bitset })
    }

    #[inline(always)]
    fn matches<S: Simd>(&self, chunk: u8x64<S>) -> mask8x64<S> {
        let bits = u8x64::block_splat(u8x16::from_fn(chunk.simd, |i| 1 << (i % 8)));
        let bit = bits.swizzle_dyn_within_blocks(chunk & 0b0111);
        !(bit & membership_bits(&self.bitset, chunk >> 3)).simd_eq(0)
    }
}

/// Looks each byte's high five bits up in the 256-bit table, giving the table byte
/// that holds its membership bit.
#[inline(always)]
fn membership_bits<S: Simd>(bitset: &Bitset, indices: u8x64<S>) -> u8x64<S> {
    let table = u8x32::load_array_ref(indices.simd, bitset.as_array());
    // The table spans 32 lanes, so the lookup runs at whatever width the target
    // implements natively rather than at `u8x64` always.
    if S::u8s::N >= 64 {
        let table = table.combine(table);
        table.swizzle_dyn(indices)
    } else {
        let (lo, hi) = indices.split();
        table.swizzle_dyn(lo).combine(table.swizzle_dyn(hi))
    }
}
