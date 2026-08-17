use crate::bitset::Bitset;
use crate::vector::{ConstantNibble, NibbleLookup};
use crate::{Backend, Family, Finder, FinderKind, vector};
use core::range::RangeInclusive;
use fearless_simd::Level;

const ARRAY_MAX: usize = 24;

#[derive(Debug, Copy, Clone)]
pub struct Bytes(pub(crate) BytesRepr);

#[derive(Debug, Copy, Clone)]
pub(crate) enum BytesRepr {
    Array { count: u8, arr: [u8; ARRAY_MAX] },
    Range(RangeInclusive<u8>),
    Bitset(Bitset),
}

impl Default for Bytes {
    fn default() -> Self {
        Self::new()
    }
}

impl Bytes {
    pub const fn new() -> Self {
        Self(BytesRepr::Array {
            count: 0,
            arr: [0; ARRAY_MAX],
        })
    }

    pub const fn from_bytes(bytes: &[u8]) -> Self {
        let mut res = Self::new();
        let mut i = 0;
        while i < bytes.len() {
            res.add(bytes[i]);
            i += 1;
        }
        res
    }

    // Intentionally take a ops::RangeInclusive rather than a range::RangeInclusive.
    // Its worth it to support `..=` syntax
    pub const fn from_range(range: core::ops::RangeInclusive<u8>) -> Self {
        let mut res = Self::new();
        res.add_range(RangeInclusive {
            start: *range.start(),
            last: *range.end(),
        });
        res
    }

    pub const fn add(&mut self, item: u8) {
        match &mut self.0 {
            BytesRepr::Array { count, arr } => {
                let len = *count as usize;
                // Not worth keeping sorted/binary searching vs linear search on 24 items
                let mut i = 0;
                while i < len {
                    if arr[i] == item {
                        return;
                    }
                    i += 1;
                }
                if len < ARRAY_MAX {
                    arr[len] = item;
                    *count += 1;
                } else {
                    self.0 = if let Some(range) = disjoint_items_to_range(item, arr) {
                        BytesRepr::Range(range)
                    } else {
                        let mut bitset = Bitset::from_bytes(arr);
                        bitset.add(item);
                        BytesRepr::Bitset(bitset)
                    };
                }
            }
            BytesRepr::Range(range) => {
                if let Some(x) = range.start.checked_sub(1)
                    && x == item
                {
                    range.start = item;
                } else if let Some(x) = range.last.checked_add(1)
                    && x == item
                {
                    range.last = item;
                } else if item < range.start || item > range.last {
                    let mut bitset = Bitset::new();
                    bitset.add_range(*range);
                    bitset.add(item);
                    self.0 = BytesRepr::Bitset(bitset);
                }
            }
            BytesRepr::Bitset(bitset) => {
                bitset.add(item);
            }
        }
    }

    /// Adds every byte from `range.start` through `range.last`, inclusive.
    ///
    /// An empty range (one whose `start` is past its `last`) adds nothing.
    pub const fn add_range(&mut self, range: RangeInclusive<u8>) {
        let RangeInclusive { start, last } = range;
        if start > last {
            return;
        }
        match self.0 {
            // One at a time, so a range small enough to stay in the array keeps the
            // representations only the array can reach.
            BytesRepr::Array { .. } => {
                let mut item = start;
                loop {
                    self.add(item);
                    if item == last {
                        return;
                    }
                    item += 1;
                }
            }
            BytesRepr::Range(existing) => {
                // Two ranges stay one range only if they overlap or touch.
                if start <= existing.last.saturating_add(1)
                    && existing.start <= last.saturating_add(1)
                {
                    self.0 = BytesRepr::Range(RangeInclusive {
                        start: if start < existing.start {
                            start
                        } else {
                            existing.start
                        },
                        last: if last > existing.last {
                            last
                        } else {
                            existing.last
                        },
                    });
                } else {
                    let mut bitset = Bitset::new();
                    bitset.add_range(existing);
                    bitset.add_range(range);
                    self.0 = BytesRepr::Bitset(bitset);
                }
            }
            BytesRepr::Bitset(mut bitset) => {
                bitset.add_range(range);
                self.0 = BytesRepr::Bitset(bitset);
            }
        }
    }

    pub fn finder(&self) -> Finder {
        self.finder_with(Backend::Auto)
    }

    pub fn finder_with(&self, backend: Backend) -> Finder {
        let level = Level::new();
        let family = match backend {
            Backend::Scalar => Family::Word,
            Backend::Auto if level.is_fallback() => Family::Word,
            Backend::Auto => Family::Vector,
        };
        // A word kernel has no shuffle to reach for, so it classifies as a vector target
        // without fast ones does.
        let fast_shuffles = matches!(family, Family::Vector) && vector::has_byte_shuffle(level);
        Finder::new(level, family, self.kind(fast_shuffles))
    }

    /// Picks the kind that matches this set most cheaply.
    ///
    /// `fast_shuffles` gates the kinds a kernel can only scan by shuffling bytes within a
    /// vector: unset for a target whose shuffles are slow, and for the word kernels, which
    /// have none.
    fn kind(&self, fast_shuffles: bool) -> FinderKind {
        match self.0 {
            BytesRepr::Array { count: 0, arr: _ } => FinderKind::Never,

            // 1-3 bytes
            BytesRepr::Array {
                count: 1,
                arr: [item1, ..],
            } => FinderKind::OneByte(item1),
            BytesRepr::Array {
                count: 2,
                arr: [item1, item2, ..],
            } => FinderKind::TwoBytes([item1, item2]),
            BytesRepr::Array {
                count: 3,
                arr: [item1, item2, item3, ..],
            } => FinderKind::ThreeBytes([item1, item2, item3]),

            // Ranges
            BytesRepr::Array { count, ref arr }
                if let Some(range) = disjoint_slice_to_range(&arr[..usize::from(count)]) =>
            {
                FinderKind::OneRange(range)
            }
            BytesRepr::Range(range) => FinderKind::OneRange(range),
            BytesRepr::Bitset(bitset) if let Some(range) = bitset.extract_range() => {
                FinderKind::OneRange(range)
            }

            // Small set
            BytesRepr::Array {
                count: count @ ..=8,
                ref arr,
            } if fast_shuffles => {
                let mut lo_lookup = NibbleLookup::default();
                let mut hi_lookup = NibbleLookup::default();
                for (i, &item) in arr[..usize::from(count)].iter().enumerate() {
                    lo_lookup.set(item & 0x0F, i as u8);
                    hi_lookup.set(item >> 4, i as u8);
                }
                FinderKind::SmallSet {
                    lo_lookup,
                    hi_lookup,
                }
            }

            // Constant Nibble
            BytesRepr::Array {
                count: count @ ..=16,
                ref arr,
            } if fast_shuffles
                && let Some((nibble, lookup)) = constant_nibble(&arr[..usize::from(count)]) =>
            {
                FinderKind::ConstantNibble(nibble, lookup)
            }

            // Bitset-based any byte
            BytesRepr::Array { count, ref arr } => {
                FinderKind::AnyByte(Bitset::from_bytes(&arr[..usize::from(count)]))
            }
            BytesRepr::Bitset(bitset) => FinderKind::AnyByte(bitset),
        }
    }
}

const fn disjoint_slice_to_range(arr: &[u8]) -> Option<RangeInclusive<u8>> {
    let Some((&first, rest)) = arr.split_first() else {
        return None;
    };
    disjoint_items_to_range(first, rest)
}

const fn disjoint_items_to_range(first_item: u8, arr: &[u8]) -> Option<RangeInclusive<u8>> {
    // All items must be unique.
    if cfg!(debug_assertions) {
        let mut i = 0;
        while i < arr.len() {
            assert!(arr[i] != first_item, "items must be disjoint");
            i += 1;
        }

        let mut i = 0;

        while i < arr.len() {
            let mut j = i + 1;
            while j < arr.len() {
                assert!(arr[i] != arr[j], "items must be disjoint");
                j += 1;
            }
            i += 1;
        }
    }
    let mut min = first_item;
    let mut max = first_item;

    let mut i = 0;
    while i < arr.len() {
        let item = arr[i];
        if item < min {
            min = item;
        }
        if item > max {
            max = item;
        }
        i += 1;
    }
    // Since all items are unique, this is a contiguous range if the distance between min and max
    // is equal to the number of items - 1
    if (max - min) as usize == arr.len() {
        Some(RangeInclusive {
            start: min,
            last: max,
        })
    } else {
        None
    }
}

fn constant_nibble(items: &[u8]) -> Option<(ConstantNibble, [u8; 16])> {
    let first = *items.first()?;
    let lo_nibble = first & 0x0F;
    let hi_nibble = first >> 4;
    // Unfilled slots need a sentinel that can never match: slot `i` is only ever
    // compared against bytes whose variable nibble is `i`, so the sentinel's own
    // variable nibble must differ from its index. 0x00 satisfies that everywhere
    // except slot 0, hence the special case — the variable nibble is the high one
    // for lo_table and the low one for hi_table.
    let mut lo_table: Option<[u8; 16]> = Some(core::array::from_fn(|i| u8::from(i == 0) << 4));
    let mut hi_table: Option<[u8; 16]> = Some(core::array::from_fn(|i| u8::from(i == 0)));

    for &item in items {
        let lo = item & 0x0F;
        let hi = item >> 4;
        if let Some(table) = &mut lo_table
            && lo == lo_nibble
        {
            table[hi as usize] = item;
        } else {
            lo_table = None;
        }
        if let Some(table) = &mut hi_table
            && hi == hi_nibble
        {
            table[lo as usize] = item;
        } else {
            hi_table = None;
        }
        if lo_table.is_none() && hi_table.is_none() {
            return None;
        }
    }
    if let Some(table) = lo_table {
        Some((ConstantNibble::Lo, table))
    } else {
        hi_table.map(|table| (ConstantNibble::Hi, table))
    }
}

impl FromIterator<u8> for Bytes {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        let mut res = Self::new();
        res.extend(iter);
        res
    }
}

impl Extend<u8> for Bytes {
    fn extend<T: IntoIterator<Item = u8>>(&mut self, iter: T) {
        iter.into_iter().for_each(|item| self.add(item));
    }
}
