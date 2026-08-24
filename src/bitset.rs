use core::range::RangeInclusive;

const TABLE_BITS: usize = 256;
const TABLE_BYTES: usize = TABLE_BITS / u8::BITS as usize;

#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub(crate) struct Bitset([u8; TABLE_BYTES]);

impl Bitset {
    pub(crate) const fn new() -> Self {
        Self([0; TABLE_BYTES])
    }

    pub(crate) fn as_array(&self) -> &[u8; 32] {
        &self.0
    }

    pub(crate) const fn from_bytes(bytes: &[u8]) -> Self {
        let mut set = Self::new();
        let mut i = 0;
        while i < bytes.len() {
            set.add(bytes[i]);
            i += 1;
        }
        set
    }

    pub(crate) const fn add(&mut self, byte: u8) {
        let word_idx = (byte / 8) as usize;
        let bit_idx = byte % 8;
        let mask = 1 << bit_idx;
        self.0[word_idx] |= mask;
    }

    pub(crate) const fn add_range(&mut self, range: RangeInclusive<u8>) {
        let RangeInclusive { start, last } = range;
        if start > last {
            return;
        }
        let first_word = (start / 8) as usize;
        let last_word = (last / 8) as usize;
        let from_start = u8::MAX << (start % 8);
        let through_last = u8::MAX >> (7 - last % 8);

        if first_word == last_word {
            self.0[first_word] |= from_start & through_last;
            return;
        }

        self.0[first_word] |= from_start;
        let mut i = first_word + 1;
        while i < last_word {
            self.0[i] = u8::MAX;
            i += 1;
        }
        self.0[last_word] |= through_last;
    }

    pub(crate) const fn contains(&self, byte: u8) -> bool {
        let word_idx = (byte / 8) as usize;
        let bit_idx = byte % 8;
        self.0[word_idx] & (1 << bit_idx) != 0
    }

    pub(crate) fn extract_range(&self) -> Option<RangeInclusive<u8>> {
        let mut first = None;
        let mut last = 0;
        let mut count = 0;
        for (i, word) in self.as_u64s().enumerate() {
            if word == 0 {
                continue;
            }
            let base = (i * 64) as u32;
            if first.is_none() {
                first = Some(base + word.trailing_zeros());
            }
            last = base + 64 - 1 - word.leading_zeros();
            count += word.count_ones();
        }

        let first = first?;
        // The set is a range exactly when its members fill the whole span they cover.
        if count != last - first + 1 {
            return None;
        }
        Some(RangeInclusive {
            start: first as u8,
            last: last as u8,
        })
    }

    fn as_u64s(&self) -> impl Iterator<Item = u64> + use<'_> {
        let (words, []) = self.0.as_chunks::<8>() else {
            unreachable!()
        };
        words.iter().map(move |word| u64::from_le_bytes(*word))
    }
}

impl Extend<u8> for Bitset {
    fn extend<T: IntoIterator<Item = u8>>(&mut self, iter: T) {
        for byte in iter {
            self.add(byte);
        }
    }
}

impl FromIterator<u8> for Bitset {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        let mut set = Self::new();
        set.extend(iter);
        set
    }
}
