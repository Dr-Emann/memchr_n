use crate::bitset::ByteSet;
use crate::{IterState, MatchedBitset};

#[derive(Copy, Clone)]
pub(crate) struct SearchPlan {
    pub(crate) kernel_storage: KernelStorage,
    pub(crate) scan_ops: &'static ScanOps,
    pub(crate) engine: crate::Engine,
    pub(crate) kernel_kind: KernelKind,
}

#[derive(Copy, Clone)]
pub(crate) struct BitsetLookup {
    byte_set: ByteSet,
}

impl BitsetLookup {
    pub(crate) const fn new(byte_set: ByteSet) -> Self {
        Self { byte_set }
    }

    #[inline]
    pub(crate) fn contains(&self, byte: u8) -> bool {
        self.byte_set.contains(byte)
    }

    pub(crate) fn byte_set(&self) -> &ByteSet {
        &self.byte_set
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum KernelKind {
    BitsetLookup,
    SmallSet,
    FixedNibble,
    OneByte,
    NotByte,
    TwoBytes,
    ThreeBytes,
    OneRange,
    Never,
}

pub(crate) trait StoredKernel: Copy + Into<KernelStorage> {
    /// Borrows the stored kernel.
    ///
    /// # Safety
    ///
    /// The live field of `storage` must contain `Self`.
    unsafe fn from_storage(storage: &KernelStorage) -> &Self;
}

macro_rules! define_kernel_storage {
    (
        $($field:ident: $kernel:ty),* $(,)?
    ) => {
        #[derive(Copy, Clone)]
        #[repr(align(16))]
        pub(crate) union KernelStorage {
            $(pub(crate) $field: $kernel),*
        }

        $(
        impl From<$kernel> for KernelStorage {
            fn from(kernel: $kernel) -> Self {
                Self { $field: kernel }
            }
        }

        impl StoredKernel for $kernel {
            unsafe fn from_storage(storage: &KernelStorage) -> &Self {
                unsafe { &storage.$field }
            }
        }
        )*
    };
}

define_kernel_storage! {
    swar_any1: crate::swar::kernels::AnyOf<1>,
    swar_any2: crate::swar::kernels::AnyOf<2>,
    swar_any3: crate::swar::kernels::AnyOf<3>,
    swar_range: crate::swar::kernels::OneRange,
    swar_not_byte: crate::swar::kernels::NotByte,

    vector_any1: crate::vector::kernels::AnyOf<1>,
    vector_any2: crate::vector::kernels::AnyOf<2>,
    vector_any3: crate::vector::kernels::AnyOf<3>,
    vector_range: crate::vector::kernels::OneRange,
    vector_small_set: crate::vector::kernels::SmallSet,
    vector_fixed_nibble: crate::vector::kernels::FixedNibbleSet,
    vector_not_byte: crate::vector::kernels::NotByte,

    bitset: BitsetLookup,

    never: (),
}

impl KernelStorage {
    /// Borrows the stored kernel as `T`.
    ///
    /// # Safety
    ///
    /// The live field must contain `T`.
    pub(crate) unsafe fn get_unchecked<T: StoredKernel>(&self) -> &T {
        unsafe { T::from_storage(self) }
    }
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
    pub(crate) next_match_batch: unsafe fn(&KernelStorage, &mut IterState<'_>) -> MatchedBitset,
    pub(crate) count_all: unsafe fn(&KernelStorage, &[u8]) -> usize,
    pub(crate) first_match: unsafe fn(&KernelStorage, &[u8]) -> Option<usize>,
}

pub(crate) fn never_scan_ops() -> &'static ScanOps {
    fn next_match_batch(_data: &KernelStorage, state: &mut IterState<'_>) -> MatchedBitset {
        state.scan_offset = state.haystack.len();
        0
    }

    fn count_all(_data: &KernelStorage, _haystack: &[u8]) -> usize {
        0
    }

    fn first_match(_data: &KernelStorage, _haystack: &[u8]) -> Option<usize> {
        None
    }

    &ScanOps {
        next_match_batch,
        count_all,
        first_match,
    }
}
