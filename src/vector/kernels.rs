use super::Kernel;
use crate::bitset::{ByteRange, ByteSet};
use crate::search::StoredKernel;
use crate::{BitsetLookup, FixedNibble, FixedNibbleTable, NibbleLookup};
use fearless_simd::prelude::*;
use fearless_simd::{u8x16, u8x32, u8x64};

/// Compares each byte with up to three needles.
#[derive(Copy, Clone)]
pub(crate) struct AnyOf<const N: usize> {
    splatted_needles: [[u8; 16]; N],
}

impl<const N: usize> AnyOf<N> {
    pub(crate) fn new(needles: [u8; N]) -> Self {
        let mut splatted_needles = [[0; 16]; N];
        for (dst, needle) in splatted_needles.iter_mut().zip(needles) {
            *dst = [needle; 16];
        }
        Self { splatted_needles }
    }
}

impl<const N: usize> Kernel for AnyOf<N>
where
    Self: StoredKernel,
{
    #[inline(always)]
    fn matches<S: Simd, V: SimdInt<S, Element = u8, Block = u8x16<S>>>(&self, chunk: V) -> V::Mask {
        let simd = chunk.token();
        let mut matched = V::Mask::splat(simd, false);
        for &needle in &self.splatted_needles {
            matched |= chunk.simd_eq(V::block_splat(u8x16::load_array(simd, needle)));
        }
        matched
    }

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        for needle in &self.splatted_needles {
            if needle[0] == byte {
                return true;
            }
        }
        false
    }
}

#[derive(Copy, Clone)]
pub(crate) struct NotByte {
    splatted_byte: [u8; 16],
}

impl NotByte {
    pub(crate) fn new(byte: u8) -> Self {
        Self {
            splatted_byte: [byte; 16],
        }
    }
}

impl Kernel for NotByte {
    #[inline(always)]
    fn matches<S: Simd, V: SimdInt<S, Element = u8, Block = u8x16<S>>>(&self, chunk: V) -> V::Mask {
        !chunk.simd_eq(V::block_splat(u8x16::load_array(
            chunk.token(),
            self.splatted_byte,
        )))
    }

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        byte != self.splatted_byte[0]
    }
}

#[derive(Copy, Clone)]
pub(crate) struct OneRange {
    splatted_start: [u8; 16],
    splatted_last: [u8; 16],
}

impl OneRange {
    pub(crate) fn new(ByteRange { start, last }: ByteRange) -> Self {
        Self {
            splatted_start: [start; 16],
            splatted_last: [last; 16],
        }
    }
}

impl Kernel for OneRange {
    #[inline(always)]
    fn matches<S: Simd, V: SimdInt<S, Element = u8, Block = u8x16<S>>>(&self, chunk: V) -> V::Mask {
        let simd = chunk.token();
        chunk.simd_ge(V::block_splat(u8x16::load_array(simd, self.splatted_start)))
            & chunk.simd_le(V::block_splat(u8x16::load_array(simd, self.splatted_last)))
    }

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        let (start, last) = (self.splatted_start[0], self.splatted_last[0]);
        start <= byte && byte <= last
    }
}

#[derive(Copy, Clone)]
pub(crate) struct SmallSet {
    lo_lookup: NibbleLookup,
    hi_lookup: NibbleLookup,
}

impl SmallSet {
    pub(crate) fn new(lo_lookup: NibbleLookup, hi_lookup: NibbleLookup) -> Self {
        Self {
            lo_lookup,
            hi_lookup,
        }
    }
}

impl Kernel for SmallSet {
    #[inline(always)]
    fn matches<S: Simd, V: SimdInt<S, Element = u8, Block = u8x16<S>, ByteVector = V>>(
        &self,
        chunk: V,
    ) -> V::Mask {
        let simd = chunk.token();
        let lo_lookup = V::block_splat(u8x16::load_array(simd, self.lo_lookup.0));
        let hi_lookup = V::block_splat(u8x16::load_array(simd, self.hi_lookup.0));

        let lo = lo_lookup.swizzle_dyn_within_blocks(chunk & 0x0F);
        let hi = hi_lookup.swizzle_dyn_within_blocks(chunk >> 4);

        !(lo & hi).simd_eq(0)
    }

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        // Each bit represents a group containing every combination of its high and
        // low nibbles, so a shared bit proves this byte belongs to the set.
        let lo = self.lo_lookup.0[usize::from(byte & 0x0F)];
        let hi = self.hi_lookup.0[usize::from(byte >> 4)];
        lo & hi != 0
    }
}

#[derive(Copy, Clone)]
pub(crate) struct FixedNibbleSet {
    fixed_nibble: FixedNibble,
    table: [u8; 16],
}

impl FixedNibbleSet {
    pub(crate) fn new(table: FixedNibbleTable) -> Self {
        let FixedNibbleTable {
            fixed_nibble,
            table,
        } = table;
        Self {
            fixed_nibble,
            table,
        }
    }
}

impl Kernel for FixedNibbleSet {
    #[inline(always)]
    fn matches<S: Simd, V: SimdInt<S, Element = u8, Block = u8x16<S>, ByteVector = V>>(
        &self,
        chunk: V,
    ) -> V::Mask {
        let table = V::block_splat(u8x16::simd_from(chunk.token(), self.table));
        let non_const_nibbles = match self.fixed_nibble {
            FixedNibble::Low => chunk >> 4,
            FixedNibble::High => chunk & 0x0F,
        };
        let should_match = table.swizzle_dyn_within_blocks(non_const_nibbles);
        chunk.simd_eq(should_match)
    }

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        // Empty slots contain a sentinel whose variable nibble differs from its index.
        let variable_nibble = match self.fixed_nibble {
            FixedNibble::Low => byte >> 4,
            FixedNibble::High => byte & 0x0F,
        };
        self.table[usize::from(variable_nibble)] == byte
    }
}

impl Kernel for BitsetLookup {
    #[inline(always)]
    fn matches<S: Simd, V: SimdInt<S, Element = u8, Block = u8x16<S>, ByteVector = V>>(
        &self,
        chunk: V,
    ) -> V::Mask {
        let bits = V::block_splat(u8x16::from_fn(chunk.token(), |i| 1 << (i % 8)));
        let bit = bits.swizzle_dyn_within_blocks(chunk & 0b0111);
        !(bit & lookup_membership_bytes(self.byte_set(), chunk >> 3)).simd_eq(0)
    }

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        self.contains(byte)
    }
}

/// Looks each byte's high five bits up in the 256-bit table, giving the table byte
/// that holds its membership bit.
#[inline(always)]
fn lookup_membership_bytes<S: Simd, V: SimdInt<S, Element = u8, ByteVector = V>>(
    byte_set: &ByteSet,
    indices: V,
) -> V {
    let simd = indices.token();
    let table = u8x32::load_array_ref(simd, byte_set.as_array());

    const { assert!(V::LEN == 16 || V::LEN == 32 || V::LEN == 64) }
    match V::LEN {
        16 => {
            let indices = u8x16::from_slice(simd, indices.as_slice());
            let (lo, hi) = table.split();
            let res = lo.concat_swizzle_dyn(hi, indices);
            V::from_slice(simd, res.as_slice())
        }
        32 => {
            let indices = u8x32::from_slice(simd, indices.as_slice());
            let res = table.swizzle_dyn(indices);
            V::from_slice(simd, res.as_slice())
        }
        64 => {
            let indices = u8x64::from_slice(simd, indices.as_slice());
            let res = if S::u8s::LEN >= 64 {
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
