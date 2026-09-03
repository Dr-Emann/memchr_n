use super::Kernel;
use crate::bitset::Bitset;
use crate::{ConstantNibble, KernelData, NibbleLookup};
use fearless_simd::prelude::*;
use fearless_simd::{u8x16, u8x32, u8x64};

/// Compares against each needle in turn, which beats a table lookup while the set is
/// small enough that the comparisons stay cheaper than the lookup they replace.
#[derive(Copy, Clone)]
pub(crate) struct AnyOf<S: Simd, const N: usize> {
    needles: [u8x16<S>; N],
    /// The same needles, unsplatted, for [`Kernel::matches_byte`]. See [`plain`].
    bytes: [u8; N],
}

impl<S: Simd, const N: usize> Kernel<S> for AnyOf<S, N> {
    unsafe fn from_data(simd: S, data: &KernelData) -> Self {
        const { assert!(N <= 3, "`splatted_needles` holds three") }
        // SAFETY: the caller guarantees `splatted_needles` is live, and the assertion above
        // keeps the reads below inside it.
        let splatted = unsafe { &data.splatted_needles };
        Self {
            needles: core::array::from_fn(|i| u8x16::load_array_ref(simd, &splatted[i])),
            bytes: core::array::from_fn(|i| plain(&splatted[i])),
        }
    }

    #[inline(always)]
    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>>>(&self, chunk: V) -> V::Mask {
        let mut matched = V::Mask::splat(chunk.witness(), false);
        for &needle in &self.needles {
            matched |= chunk.simd_eq(V::block_splat(needle));
        }
        matched
    }

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        self.bytes.contains(&byte)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct OneRange<S: Simd> {
    start: u8x16<S>,
    last: u8x16<S>,
    /// The same endpoints, unsplatted, for [`Kernel::matches_byte`]. See [`plain`].
    bounds: (u8, u8),
}

impl<S: Simd> Kernel<S> for OneRange<S> {
    unsafe fn from_data(simd: S, data: &KernelData) -> Self {
        // SAFETY: the caller guarantees `splatted_range` is live.
        let [start, last] = unsafe { &data.splatted_range };
        Self {
            start: u8x16::load_array_ref(simd, start),
            last: u8x16::load_array_ref(simd, last),
            bounds: (plain(start), plain(last)),
        }
    }
    #[inline(always)]
    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>>>(&self, chunk: V) -> V::Mask {
        chunk.simd_ge(V::block_splat(self.start)) & chunk.simd_le(V::block_splat(self.last))
    }

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        // One subtraction would do instead of two compares, but the data here is a pair of
        // bounds and not a start and a span, so spelling it as the pair keeps this reading
        // like the vector kernel above.
        let (start, last) = self.bounds;
        start <= byte && byte <= last
    }
}

#[derive(Copy, Clone)]
pub(crate) struct SmallSet {
    lo_lookup: NibbleLookup,
    hi_lookup: NibbleLookup,
}

impl<S: Simd> Kernel<S> for SmallSet {
    unsafe fn from_data(_simd: S, data: &KernelData) -> Self {
        // SAFETY: the caller guarantees `nibble_lookups` is live.
        let [lo_lookup, hi_lookup] = unsafe { data.nibble_lookups };
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
        // The shuffle above is a table index, so a scalar one is the same index spelled with
        // brackets. A member sets the same bit in both tables, so a shared bit means the two
        // nibbles came from one member rather than from two different ones.
        let lo = self.lo_lookup.0[usize::from(byte & 0x0F)];
        let hi = self.hi_lookup.0[usize::from(byte >> 4)];
        lo & hi != 0
    }
}

#[derive(Copy, Clone)]
pub(crate) struct SingleNibble {
    which: ConstantNibble,
    table: [u8; 16],
}

impl<S: Simd> Kernel<S> for SingleNibble {
    unsafe fn from_data(_simd: S, data: &KernelData) -> Self {
        // SAFETY: the caller guarantees `nibble_table` is live.
        let table = unsafe { data.nibble_table };
        Self {
            which: table.which,
            table: table.table,
        }
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

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        // As in the vector kernel: the variable nibble picks the one member it could be, and
        // the byte matches only by being that member. An unfilled slot holds a sentinel whose
        // own variable nibble is not its index, so it cannot be picked by the byte it holds.
        let variable_nibble = match self.which {
            ConstantNibble::Lo => byte >> 4,
            ConstantNibble::Hi => byte & 0x0F,
        };
        self.table[usize::from(variable_nibble)] == byte
    }
}

#[derive(Copy, Clone)]
pub(crate) struct AnyByte {
    bitset: Bitset,
}

impl<S: Simd> Kernel<S> for AnyByte {
    unsafe fn from_data(_simd: S, data: &KernelData) -> Self {
        // SAFETY: the caller guarantees `bitset` is live.
        Self {
            bitset: unsafe { data.bitset },
        }
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

    #[inline(always)]
    fn matches_byte(&self, byte: u8) -> bool {
        // The gather above is a bit test through two shuffles because a vector has no
        // addressable table. One byte can just index the table it is a bit of.
        self.bitset.contains(byte)
    }
}

/// The byte a block of [`KernelData`] holds splatted, for [`Kernel::matches_byte`].
///
/// # Why this borrows
///
/// The block has to stay in memory, which is why every caller reaches it through
/// `&data.<field>` and loads its vector with `load_array_ref`. Copy the block out of the union
/// first and it becomes a value LLVM owns, and one scalar read of it is then enough for LLVM
/// to split the whole thing into sixteen: the single aligned load that fed the broadcast turns
/// into a byte load and fifteen `vpinsrb`s to put the vector back together, on every call at
/// every haystack length. That is worth 3.3ns against 8.2 on a 32-byte `OneRange` search — far
/// more than the short-haystack probe this exists for can win back.
#[inline(always)]
fn plain(splatted: &[u8; 16]) -> u8 {
    splatted[0]
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
