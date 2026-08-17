use super::Kernel;
use crate::bitset::ByteSet;
use crate::{BitsetLookup, FixedNibble, KernelData, NibbleLookup};
use fearless_simd::prelude::*;
use fearless_simd::{u8x16, u8x32, u8x64};

/// Compares each byte with up to three needles.
#[derive(Copy, Clone)]
pub(crate) struct AnyOf<S: Simd, const N: usize> {
    needle_vectors: [u8x16<S>; N],
}

impl<S: Simd, const N: usize> Kernel<S> for AnyOf<S, N> {
    unsafe fn from_data(simd: S, kernel_data: &KernelData) -> Self {
        const { assert!(N <= 3, "`splatted_needles` holds three") }
        // SAFETY: the caller guarantees `splatted_needles` is live; `N <= 3` bounds the reads.
        let splatted = unsafe { &kernel_data.splatted_needles };
        Self {
            needle_vectors: core::array::from_fn(|i| u8x16::load_array_ref(simd, &splatted[i])),
        }
    }

    #[inline(always)]
    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>>>(&self, chunk: V) -> V::Mask {
        let mut matched = V::Mask::splat(chunk.witness(), false);
        for &needle in &self.needle_vectors {
            matched |= chunk.simd_eq(V::block_splat(needle));
        }
        matched
    }

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        self.needle_vectors.iter().map(|x| x[0]).any(|x| x == byte)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct OneRange<S: Simd> {
    start_vector: u8x16<S>,
    last_vector: u8x16<S>,
    scalar_bounds: (u8, u8),
}

impl<S: Simd> Kernel<S> for OneRange<S> {
    unsafe fn from_data(simd: S, kernel_data: &KernelData) -> Self {
        // SAFETY: the caller guarantees `splatted_bounds` is live.
        let [start, last] = unsafe { &kernel_data.splatted_bounds };
        Self {
            start_vector: u8x16::load_array_ref(simd, start),
            last_vector: u8x16::load_array_ref(simd, last),
            scalar_bounds: (splatted_byte(start), splatted_byte(last)),
        }
    }
    #[inline(always)]
    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>>>(&self, chunk: V) -> V::Mask {
        chunk.simd_ge(V::block_splat(self.start_vector))
            & chunk.simd_le(V::block_splat(self.last_vector))
    }

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        let (start, last) = self.scalar_bounds;
        start <= byte && byte <= last
    }
}

#[derive(Copy, Clone)]
pub(crate) struct SmallSet {
    lo_lookup: NibbleLookup,
    hi_lookup: NibbleLookup,
}

impl<S: Simd> Kernel<S> for SmallSet {
    unsafe fn from_data(_simd: S, kernel_data: &KernelData) -> Self {
        // SAFETY: the caller guarantees `nibble_lookups` is live.
        let [lo_lookup, hi_lookup] = unsafe { kernel_data.nibble_lookups };
        Self {
            lo_lookup,
            hi_lookup,
        }
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

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        // A shared bit means both nibbles came from the same set member.
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

impl<S: Simd> Kernel<S> for FixedNibbleSet {
    unsafe fn from_data(_simd: S, kernel_data: &KernelData) -> Self {
        // SAFETY: the caller guarantees `fixed_nibble_table` is live.
        let table = unsafe { kernel_data.fixed_nibble_table };
        Self {
            fixed_nibble: table.fixed_nibble,
            table: table.table,
        }
    }

    #[inline(always)]
    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>, ByteVector = V>>(
        &self,
        chunk: V,
    ) -> V::Mask {
        let table = V::block_splat(u8x16::simd_from(chunk.witness(), self.table));
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

impl<S: Simd> Kernel<S> for BitsetLookup {
    unsafe fn from_data(_simd: S, kernel_data: &KernelData) -> Self {
        unsafe { BitsetLookup::from_data(kernel_data) }
    }

    #[inline(always)]
    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>, ByteVector = V>>(
        &self,
        chunk: V,
    ) -> V::Mask {
        let bits = V::block_splat(u8x16::from_fn(chunk.witness(), |i| 1 << (i % 8)));
        let bit = bits.swizzle_dyn_within_blocks(chunk & 0b0111);
        !(bit & lookup_membership_bytes(self.byte_set(), chunk >> 3)).simd_eq(0)
    }

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        self.contains(byte)
    }
}

/// The byte a block of [`KernelData`] holds splatted, for [`Kernel::matches_byte`].
///
/// Borrowing preserves an aligned vector load; copying generates scalar inserts.
#[inline(always)]
fn splatted_byte(splatted: &[u8; 16]) -> u8 {
    splatted[0]
}

/// Looks each byte's high five bits up in the 256-bit table, giving the table byte
/// that holds its membership bit.
#[inline(always)]
fn lookup_membership_bytes<S: Simd, V: SimdInt<S, Element = u8, ByteVector = V>>(
    byte_set: &ByteSet,
    indices: V,
) -> V {
    let simd = indices.witness();
    let table = u8x32::load_array_ref(simd, byte_set.as_array());

    const { assert!(V::LEN == 16 || V::LEN == 32 || V::LEN == 64) }
    match V::LEN {
        16 => {
            let indices = u8x16::from_slice(simd, indices.as_slice());
            #[cfg(target_arch = "aarch64")]
            if let Some(neon) = simd.level().as_neon() {
                let res = super::aarch64_swizzle_32_to_16(neon, table.into(), indices.into());
                return V::from_slice(simd, &res);
            }
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
