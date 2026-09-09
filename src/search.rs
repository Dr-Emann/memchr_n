use crate::bitset::ByteSet;
use crate::swar;
use crate::{IterState, MatchedBitset};

#[derive(Copy, Clone)]
pub(crate) struct SearchPlan {
    pub(crate) kernel_data: KernelData,
    pub(crate) scan_ops: &'static ScanOps,
    pub(crate) engine: crate::Engine,
    pub(crate) kernel_kind: KernelKind,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum KernelKind {
    BitsetLookup,
    SmallSet,
    FixedNibble,
    OneByte,
    TwoBytes,
    ThreeBytes,
    OneRange,
    Never,
}

#[derive(Copy, Clone)]
#[repr(align(16))]
pub(crate) union KernelData {
    pub(crate) splatted_needles: [[u8; 16]; 3],
    pub(crate) splatted_bounds: [[u8; 16]; 2],
    pub(crate) nibble_lookups: [NibbleLookup; 2],
    pub(crate) fixed_nibble_table: FixedNibbleTable,
    pub(crate) byte_set: ByteSet,
    pub(crate) range_masks: swar::kernels::OneRange,
    pub(crate) no_data: (),
}

#[derive(Copy, Clone)]
pub(crate) struct FixedNibbleTable {
    pub(crate) fixed_nibble: FixedNibble,
    pub(crate) table: [u8; 16],
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum FixedNibble {
    Low,
    High,
}

#[derive(Debug, Default, Copy, Clone)]
pub(crate) struct NibbleLookup(pub(crate) [u8; 16]);

impl NibbleLookup {
    #[inline]
    pub(crate) fn set(&mut self, nibble: u8, bit: u8) {
        debug_assert!(nibble < 16);
        debug_assert!(bit < 8);
        self.0[usize::from(nibble)] |= 1 << bit;
    }
}

#[derive(Copy, Clone)]
pub(crate) struct ScanOps {
    pub(crate) find_next: unsafe fn(&KernelData, &mut IterState<'_>) -> MatchedBitset,
    pub(crate) count_all: unsafe fn(&KernelData, &[u8]) -> usize,
    pub(crate) find_first: unsafe fn(&KernelData, &[u8]) -> Option<usize>,
}

pub(crate) fn never_scan_ops() -> &'static ScanOps {
    fn find_next(_data: &KernelData, state: &mut IterState<'_>) -> MatchedBitset {
        state.scan_offset = state.haystack.len();
        0
    }

    fn count_all(_data: &KernelData, _haystack: &[u8]) -> usize {
        0
    }

    fn find_first(_data: &KernelData, _haystack: &[u8]) -> Option<usize> {
        None
    }

    &ScanOps {
        find_next,
        count_all,
        find_first,
    }
}
