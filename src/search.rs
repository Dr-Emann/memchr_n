use crate::bitset::ByteSet;
use crate::{Engine, IterState, MatchedBitset};
use fearless_simd::{Level, dispatch};

/// A kernel paired with the scan operations and SIMD level that may run it.
///
/// Constructing a plan is the only way to fill in [`KernelStorage`] and pick a [`ScanOps`], so the
/// three always agree: the live union field holds the kernel the table's entry points read, and a
/// vector table is only ever built for a [`Level`] the caller proved supported.
#[derive(Copy, Clone)]
pub(crate) struct SearchPlan {
    kernel_storage: KernelStorage,
    scan_ops: &'static ScanOps,
    engine: Engine,
    kernel_kind: KernelKind,
}

impl SearchPlan {
    /// Plans word-at-a-time scanning with `kernel`.
    pub(crate) fn swar<K: crate::swar::Kernel>(kernel: K) -> Self {
        Self {
            kernel_storage: kernel.into(),
            scan_ops: crate::swar::scan_ops::<K>(),
            engine: Engine::Swar,
            kernel_kind: K::KIND,
        }
    }

    /// Plans vector scanning with `kernel` on the kernels `level` supports.
    pub(crate) fn vector<K: crate::vector::Kernel>(level: Level, kernel: K) -> Self {
        Self {
            kernel_storage: kernel.into(),
            scan_ops: dispatch!(level, simd => crate::vector::scan_ops::<_, K>(simd)),
            engine: Engine::Vector(level),
            kernel_kind: K::KIND,
        }
    }

    /// Plans a search that never matches.
    pub(crate) fn never() -> Self {
        Self {
            kernel_storage: ().into(),
            scan_ops: never_scan_ops(),
            engine: Engine::Swar,
            kernel_kind: KernelKind::Never,
        }
    }

    pub(crate) fn engine(&self) -> Engine {
        self.engine
    }

    pub(crate) fn kernel_kind(&self) -> KernelKind {
        self.kernel_kind
    }

    /// Returns the offset of the first matching byte in `haystack`.
    #[inline]
    pub(crate) fn first_match(&self, haystack: &[u8]) -> Option<usize> {
        // SAFETY: the constructors pair `scan_ops` with the kernel in `kernel_storage` and with a
        // SIMD level this target supports.
        unsafe { (self.scan_ops.first_match)(&self.kernel_storage, haystack) }
    }

    /// Counts the matching bytes in `haystack`.
    #[inline]
    pub(crate) fn count_all(&self, haystack: &[u8]) -> usize {
        // SAFETY: the constructors pair `scan_ops` with the kernel in `kernel_storage` and with a
        // SIMD level this target supports.
        unsafe { (self.scan_ops.count_all)(&self.kernel_storage, haystack) }
    }

    /// Scans from `state.scan_offset` until it finds matches or reaches the end.
    ///
    /// # Safety
    ///
    /// `state.scan_offset` must not exceed `state.haystack.len()`.
    #[inline]
    pub(crate) unsafe fn next_match_batch(&self, state: &mut IterState<'_>) -> MatchedBitset {
        debug_assert!(state.scan_offset <= state.haystack.len());
        // SAFETY: the constructors pair `scan_ops` with the kernel in `kernel_storage` and with a
        // SIMD level this target supports; the caller guarantees the scan offset is in bounds.
        unsafe { (self.scan_ops.next_match_batch)(&self.kernel_storage, state) }
    }
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

/// A kernel that [`KernelStorage`] can hold, along with the [`KernelKind`] naming it.
///
/// # Safety
///
/// Converting `Self` into a [`KernelStorage`] and then calling [`from_storage`] on that storage
/// must yield the same kernel: both must name the one union field reserved for `Self`.
///
/// [`from_storage`]: StoredKernel::from_storage
pub(crate) unsafe trait StoredKernel: Copy + Into<KernelStorage> {
    /// Names this kernel in diagnostics.
    const KIND: KernelKind;

    /// Borrows the stored kernel.
    ///
    /// # Safety
    ///
    /// The live field of `storage` must contain `Self`.
    unsafe fn from_storage(storage: &KernelStorage) -> &Self;
}

macro_rules! define_kernel_storage {
    (
        $($field:ident: $kernel:ty => $kind:ident),* $(,)?
    ) => {
        #[derive(Copy, Clone)]
        #[repr(align(16))]
        pub(crate) union KernelStorage {
            $($field: $kernel),*
        }

        $(
        impl From<$kernel> for KernelStorage {
            fn from(kernel: $kernel) -> Self {
                Self { $field: kernel }
            }
        }

        // SAFETY: both halves name `$field`, the field no other kernel type is given.
        unsafe impl StoredKernel for $kernel {
            const KIND: KernelKind = KernelKind::$kind;

            unsafe fn from_storage(storage: &KernelStorage) -> &Self {
                unsafe { &storage.$field }
            }
        }
        )*
    };
}

define_kernel_storage! {
    swar_any1: crate::swar::kernels::AnyOf<1> => OneByte,
    swar_any2: crate::swar::kernels::AnyOf<2> => TwoBytes,
    swar_any3: crate::swar::kernels::AnyOf<3> => ThreeBytes,
    swar_range: crate::swar::kernels::OneRange => OneRange,
    swar_not_byte: crate::swar::kernels::NotByte => NotByte,

    vector_any1: crate::vector::kernels::AnyOf<1> => OneByte,
    vector_any2: crate::vector::kernels::AnyOf<2> => TwoBytes,
    vector_any3: crate::vector::kernels::AnyOf<3> => ThreeBytes,
    vector_range: crate::vector::kernels::OneRange => OneRange,
    vector_small_set: crate::vector::kernels::SmallSet => SmallSet,
    vector_fixed_nibble: crate::vector::kernels::FixedNibbleSet => FixedNibble,
    vector_not_byte: crate::vector::kernels::NotByte => NotByte,

    bitset: BitsetLookup => BitsetLookup,

    never: () => Never,
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

fn never_scan_ops() -> &'static ScanOps {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitset::ByteRange;

    fn haystack() -> Vec<u8> {
        let mut haystack = Vec::with_capacity(300);
        for i in 0..300usize {
            haystack.push((i * 37 % 256) as u8);
        }
        haystack
    }

    fn scalar_offsets(haystack: &[u8], member: &dyn Fn(u8) -> bool) -> Vec<usize> {
        let mut offsets = Vec::new();
        for (offset, &byte) in haystack.iter().enumerate() {
            if member(byte) {
                offsets.push(offset);
            }
        }
        offsets
    }

    fn batched_offsets(plan: &SearchPlan, haystack: &[u8]) -> Vec<usize> {
        let mut state = IterState {
            haystack,
            scan_offset: 0,
            match_base: 0,
        };
        let mut offsets = Vec::new();
        while state.scan_offset != haystack.len() {
            let scan_offset = state.scan_offset;
            // SAFETY: the initial offset is zero, and each batch's bounds are checked below.
            let mut bits = unsafe { plan.next_match_batch(&mut state) };
            assert!(state.scan_offset > scan_offset, "batch did not advance");
            assert!(
                state.scan_offset <= haystack.len(),
                "batch advanced beyond the haystack"
            );
            while bits != 0 {
                offsets.push(state.match_base + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        }
        offsets
    }

    #[track_caller]
    fn assert_operations(plan: &SearchPlan, case: &str, member: &dyn Fn(u8) -> bool) {
        let haystack = haystack();
        let expected = scalar_offsets(&haystack, member);

        assert_eq!(
            plan.first_match(&haystack),
            expected.first().copied(),
            "{case}"
        );
        assert_eq!(plan.count_all(&haystack), expected.len(), "{case}");
        assert_eq!(batched_offsets(plan, &haystack), expected, "{case}");

        assert_eq!(plan.first_match(&[]), None, "{case} on empty input");
        assert_eq!(plan.count_all(&[]), 0, "{case} on empty input");
        let mut state = IterState {
            haystack: &[],
            scan_offset: 0,
            match_base: 0,
        };
        // SAFETY: the scan offset equals the empty haystack's length.
        let bits = unsafe { plan.next_match_batch(&mut state) };
        assert_eq!(bits, 0, "{case} on empty input");
        assert_eq!(state.scan_offset, 0, "{case} on empty input");
    }

    #[track_caller]
    fn assert_plan(plan: SearchPlan, case: &str, kind: KernelKind, member: &dyn Fn(u8) -> bool) {
        assert_eq!(
            format!("{:?}", plan.kernel_kind()),
            format!("{kind:?}"),
            "{case}"
        );
        assert_operations(&plan, case, member);
    }

    #[track_caller]
    fn assert_swar_plan(
        plan: SearchPlan,
        case: &str,
        kind: KernelKind,
        member: &dyn Fn(u8) -> bool,
    ) {
        match plan.engine() {
            Engine::Swar => {}
            Engine::Vector(_) => panic!("{case} reported a vector engine"),
        }
        assert_plan(plan, case, kind, member);
    }

    #[track_caller]
    fn assert_vector_plan(
        plan: SearchPlan,
        case: &str,
        level: Level,
        kind: KernelKind,
        member: &dyn Fn(u8) -> bool,
    ) {
        match plan.engine() {
            Engine::Vector(selected) => {
                assert_eq!(format!("{selected:?}"), format!("{level:?}"), "{case}")
            }
            Engine::Swar => panic!("{case} reported a SWAR engine"),
        }
        assert_plan(plan, case, kind, member);
    }

    const NEEDLES: [u8; 3] = [b'a', 7, 200];
    const RANGE: ByteRange = ByteRange {
        start: 40,
        last: 90,
    };
    const EXCLUDED: u8 = 17;

    fn any_of(count: usize) -> impl Fn(u8) -> bool {
        move |byte| {
            let mut found = false;
            for &needle in &NEEDLES[..count] {
                found |= byte == needle;
            }
            found
        }
    }

    fn in_range(byte: u8) -> bool {
        (RANGE.start..=RANGE.last).contains(&byte)
    }

    fn not_excluded(byte: u8) -> bool {
        byte != EXCLUDED
    }

    fn scattered_set() -> ByteSet {
        let mut set = ByteSet::new();
        for byte in [0u8, 3, 31, 64, 129, 200, 251, 255] {
            set.add(byte);
        }
        set
    }

    /// The eight members of a [`vector::kernels::SmallSet`], which needs no shared nibble.
    const SMALL_SET: [u8; 8] = [0x01, 0x13, 0x25, 0x37, 0x49, 0x5B, 0x6D, 0x7F];

    /// Members sharing a low nibble, which [`vector::kernels::FixedNibbleSet`] indexes by the high one.
    const FIXED_NIBBLE_SET: [u8; 5] = [0x03, 0x23, 0x53, 0xA3, 0xF3];

    fn contains(set: &[u8], byte: u8) -> bool {
        let mut found = false;
        for &member in set {
            found |= member == byte;
        }
        found
    }

    fn small_set_kernel() -> crate::vector::kernels::SmallSet {
        let mut lo_lookup = NibbleLookup::default();
        let mut hi_lookup = NibbleLookup::default();
        for (i, &item) in SMALL_SET.iter().enumerate() {
            lo_lookup.set(item & 0x0F, i as u8);
            hi_lookup.set(item >> 4, i as u8);
        }
        crate::vector::kernels::SmallSet::new(lo_lookup, hi_lookup)
    }

    fn fixed_nibble_kernel() -> crate::vector::kernels::FixedNibbleSet {
        let table = crate::extract_fixed_nibble_table(&FIXED_NIBBLE_SET)
            .expect("the members share their low nibble");
        crate::vector::kernels::FixedNibbleSet::new(table)
    }

    #[test]
    fn swar_constructor_pairs_every_stored_kernel_with_its_operations() {
        use crate::swar::kernels;

        assert_swar_plan(
            SearchPlan::swar(kernels::AnyOf::new([NEEDLES[0]])),
            "swar AnyOf<1>",
            KernelKind::OneByte,
            &any_of(1),
        );
        assert_swar_plan(
            SearchPlan::swar(kernels::AnyOf::new([NEEDLES[0], NEEDLES[1]])),
            "swar AnyOf<2>",
            KernelKind::TwoBytes,
            &any_of(2),
        );
        assert_swar_plan(
            SearchPlan::swar(kernels::AnyOf::new(NEEDLES)),
            "swar AnyOf<3>",
            KernelKind::ThreeBytes,
            &any_of(3),
        );
        assert_swar_plan(
            SearchPlan::swar(kernels::OneRange::new(RANGE)),
            "swar OneRange",
            KernelKind::OneRange,
            &in_range,
        );
        assert_swar_plan(
            SearchPlan::swar(kernels::NotByte::new(EXCLUDED)),
            "swar NotByte",
            KernelKind::NotByte,
            &not_excluded,
        );
        let set = scattered_set();
        assert_swar_plan(
            SearchPlan::swar(BitsetLookup::new(set)),
            "swar BitsetLookup",
            KernelKind::BitsetLookup,
            &move |byte| set.contains(byte),
        );
    }

    #[test]
    fn vector_constructor_pairs_every_stored_kernel_with_its_operations() {
        use crate::vector::kernels;

        for level in [Level::baseline(), Level::new()] {
            let at = format!("at {level:?}");
            assert_vector_plan(
                SearchPlan::vector(level, kernels::AnyOf::new([NEEDLES[0]])),
                &format!("vector AnyOf<1> {at}"),
                level,
                KernelKind::OneByte,
                &any_of(1),
            );
            assert_vector_plan(
                SearchPlan::vector(level, kernels::AnyOf::new([NEEDLES[0], NEEDLES[1]])),
                &format!("vector AnyOf<2> {at}"),
                level,
                KernelKind::TwoBytes,
                &any_of(2),
            );
            assert_vector_plan(
                SearchPlan::vector(level, kernels::AnyOf::new(NEEDLES)),
                &format!("vector AnyOf<3> {at}"),
                level,
                KernelKind::ThreeBytes,
                &any_of(3),
            );
            assert_vector_plan(
                SearchPlan::vector(level, kernels::OneRange::new(RANGE)),
                &format!("vector OneRange {at}"),
                level,
                KernelKind::OneRange,
                &in_range,
            );
            assert_vector_plan(
                SearchPlan::vector(level, kernels::NotByte::new(EXCLUDED)),
                &format!("vector NotByte {at}"),
                level,
                KernelKind::NotByte,
                &not_excluded,
            );
            let set = scattered_set();
            assert_vector_plan(
                SearchPlan::vector(level, BitsetLookup::new(set)),
                &format!("vector BitsetLookup {at}"),
                level,
                KernelKind::BitsetLookup,
                &move |byte| set.contains(byte),
            );

            // The shuffle kernels are only ever planned for a level that shuffles bytes.
            if !crate::vector::has_byte_shuffle(level) {
                continue;
            }
            assert_vector_plan(
                SearchPlan::vector(level, small_set_kernel()),
                &format!("vector SmallSet {at}"),
                level,
                KernelKind::SmallSet,
                &|byte| contains(&SMALL_SET, byte),
            );
            assert_vector_plan(
                SearchPlan::vector(level, fixed_nibble_kernel()),
                &format!("vector FixedNibbleSet {at}"),
                level,
                KernelKind::FixedNibble,
                &|byte| contains(&FIXED_NIBBLE_SET, byte),
            );
        }
    }

    #[test]
    fn never_constructor_matches_nothing() {
        assert_swar_plan(SearchPlan::never(), "never", KernelKind::Never, &|_| false);
    }
}
