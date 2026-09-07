/// Which [`Kind`] a [`MemchrN`](crate::MemchrN) was built from, less the payload that
/// [`KernelData`](crate::KernelData) holds in the shape its kernel wants.
///
/// The payload cannot be recovered from that in every case — `swar`'s `OneRange` keeps only
/// the low seven bits of its start — so naming the kind is as much as [`Debug`] can offer
/// without carrying a second copy of it. This costs nothing: it fits in the padding the
/// alignment of [`KernelData`](crate::KernelData) leaves behind.
#[derive(Copy, Clone, Debug)]
pub(crate) enum Kind {
    AnyByte,
    SmallSet,
    ConstantNibble,
    OneByte,
    TwoBytes,
    ThreeBytes,
    OneRange,
    Never,
}
