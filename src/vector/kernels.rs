use super::Kernel;
use crate::bitset::Bitset;
use crate::{ConstantNibble, FinderKind, NibbleLookup};
use core::range::RangeInclusive;
use fearless_simd::prelude::*;
use fearless_simd::{u8x16, u8x32, u8x64};

/// Compares against each needle in turn, which beats a table lookup while the set is
/// small enough that the comparisons stay cheaper than the lookup they replace.
#[derive(Copy, Clone)]
pub(crate) struct AnyOf<S: Simd, const N: usize> {
    needles: [u8x16<S>; N],
}

#[inline(always)]
fn any_of_matches<S: Simd, V: SimdInt<S, Element = u8, Block = u8x16<S>>, const N: usize>(
    chunk: V,
    needles: [u8x16<S>; N],
) -> V::Mask {
    let mut matched = V::Mask::splat(chunk.witness(), false);
    for &needle in &needles {
        matched |= chunk.simd_eq(V::block_splat(needle));
    }
    matched
}

impl<S: Simd> Kernel<S> for AnyOf<S, 1> {
    fn from_kind(simd: S, kind: &FinderKind) -> Option<Self> {
        let FinderKind::OneByte(needle) = *kind else {
            return None;
        };
        Some(Self {
            needles: [u8x16::splat(simd, needle)],
        })
    }

    #[inline(always)]
    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>>>(&self, chunk: V) -> V::Mask {
        any_of_matches(chunk, self.needles)
    }
}

impl<S: Simd> Kernel<S> for AnyOf<S, 2> {
    fn from_kind(simd: S, kind: &FinderKind) -> Option<Self> {
        let FinderKind::TwoBytes(needles) = *kind else {
            return None;
        };
        Some(Self {
            needles: needles.map(|n| u8x16::splat(simd, n)),
        })
    }

    #[inline(always)]
    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>>>(&self, chunk: V) -> V::Mask {
        any_of_matches(chunk, self.needles)
    }
}

impl<S: Simd> Kernel<S> for AnyOf<S, 3> {
    fn from_kind(simd: S, kind: &FinderKind) -> Option<Self> {
        let FinderKind::ThreeBytes(needles) = *kind else {
            return None;
        };
        Some(Self {
            needles: needles.map(|n| u8x16::splat(simd, n)),
        })
    }

    #[inline(always)]
    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>>>(&self, chunk: V) -> V::Mask {
        any_of_matches(chunk, self.needles)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct OneRange<S: Simd> {
    start: u8x16<S>,
    last: u8x16<S>,
}

impl<S: Simd> Kernel<S> for OneRange<S> {
    fn from_kind(simd: S, kind: &FinderKind) -> Option<Self> {
        let FinderKind::OneRange(RangeInclusive { start, last }) = *kind else {
            return None;
        };
        Some(Self {
            start: start.simd_into(simd),
            last: last.simd_into(simd),
        })
    }
    #[inline(always)]
    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>>>(&self, chunk: V) -> V::Mask {
        chunk.simd_ge(V::block_splat(self.start)) & chunk.simd_le(V::block_splat(self.last))
    }
}

#[derive(Copy, Clone)]
pub(crate) struct SmallSet {
    lo_lookup: NibbleLookup,
    hi_lookup: NibbleLookup,
}

impl<S: Simd> Kernel<S> for SmallSet {
    fn from_kind(_simd: S, kind: &FinderKind) -> Option<Self> {
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
    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>, ByteVector = V>>(
        &self,
        chunk: V,
    ) -> V::Mask {
        let simd = chunk.witness();
        let lo_lookup = V::block_splat(u8x16::load_array(simd, self.lo_lookup.0));
        let hi_lookup = V::block_splat(u8x16::load_array(simd, self.hi_lookup.0));

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

impl<S: Simd> Kernel<S> for SingleNibble {
    fn from_kind(_simd: S, kind: &FinderKind) -> Option<Self> {
        let FinderKind::ConstantNibble(which, table) = *kind else {
            return None;
        };
        Some(Self { which, table })
    }

    #[inline(always)]
    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>, ByteVector = V>>(
        &self,
        chunk: V,
    ) -> V::Mask {
        let table = V::block_splat(u8x16::simd_from(chunk.witness(), self.table));
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

impl<S: Simd> Kernel<S> for AnyByte {
    fn from_kind(_simd: S, kind: &FinderKind) -> Option<Self> {
        let FinderKind::AnyByte(bitset) = *kind else {
            return None;
        };
        Some(Self { bitset })
    }

    #[inline(always)]
    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>, ByteVector = V>>(
        &self,
        chunk: V,
    ) -> V::Mask {
        let bits = V::block_splat(u8x16::from_fn(chunk.witness(), |i| 1 << (i % 8)));
        let bit = bits.swizzle_dyn_within_blocks(chunk & 0b0111);
        !(bit & membership_bits(&self.bitset, chunk >> 3)).simd_eq(0)
    }
}

/// Looks each byte's high five bits up in the 256-bit table, giving the table byte
/// that holds its membership bit.
#[inline(always)]
fn membership_bits<S: Simd, V: SimdInt<S, Element = u8, ByteVector = V>>(
    bitset: &Bitset,
    indices: V,
) -> V {
    let simd = indices.witness();
    let table = u8x32::load_array_ref(simd, bitset.as_array());

    const { assert!(V::N == 16 || V::N == 32 || V::N == 64) }
    match V::N {
        16 => {
            let indices = u8x16::from_slice(simd, indices.as_slice());
            // TODO: When concat_swizzle_dyn is available, use it
            let res = table.swizzle_dyn(indices.combine(indices)).split().0;
            V::from_slice(simd, res.as_slice())
        }
        32 => {
            let indices = u8x32::from_slice(simd, indices.as_slice());
            let res = table.swizzle_dyn(indices);
            V::from_slice(simd, res.as_slice())
        }
        64 => {
            let indices = u8x64::from_slice(simd, indices.as_slice());
            let res = if S::u8s::N >= 64 {
                table.combine(table).swizzle_dyn(indices)
            } else {
                let (lo_indices, hi_indices) = indices.split();
                let (lo, hi) = (table.swizzle_dyn(lo_indices), table.swizzle_dyn(hi_indices));
                lo.combine(hi)
            };
            V::from_slice(simd, res.as_slice())
        }
        _ => unreachable!(),
    }
}
