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
