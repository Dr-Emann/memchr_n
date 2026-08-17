use core::range::RangeInclusive;

const TABLE_BITS: usize = 256;
const TABLE_BYTES: usize = TABLE_BITS / u8::BITS as usize;

#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub(crate) struct ByteSet([u8; TABLE_BYTES]);

impl ByteSet {
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
        let table_index = (byte / 8) as usize;
        let bit_index = byte % 8;
        let mask = 1 << bit_index;
        self.0[table_index] |= mask;
    }

    pub(crate) const fn add_range(&mut self, range: RangeInclusive<u8>) {
        let RangeInclusive { start, last } = range;
        if start > last {
            return;
        }
        let first_table_byte = (start / 8) as usize;
        let last_table_byte = (last / 8) as usize;
        let from_start = u8::MAX << (start % 8);
        let through_last = u8::MAX >> (7 - last % 8);

        if first_table_byte == last_table_byte {
            self.0[first_table_byte] |= from_start & through_last;
            return;
        }

        self.0[first_table_byte] |= from_start;
        let mut i = first_table_byte + 1;
        while i < last_table_byte {
            self.0[i] = u8::MAX;
            i += 1;
        }
        self.0[last_table_byte] |= through_last;
    }

    pub(crate) const fn contains(&self, byte: u8) -> bool {
        let table_index = (byte / 8) as usize;
        let bit_index = byte % 8;
        self.0[table_index] & (1 << bit_index) != 0
    }

    pub(crate) const fn as_contiguous_range(&self) -> Option<RangeInclusive<u8>> {
        let mut first = None;
        let mut last = 0;
        let mut count = 0;
        let mut i = 0;
        while i < TABLE_BYTES / 8 {
            let word = self.word(i);
            if word != 0 {
                let base = (i * 64) as u32;
                if first.is_none() {
                    first = Some(base + word.trailing_zeros());
                }
                last = base + 64 - 1 - word.leading_zeros();
                count += word.count_ones();
            }
            i += 1;
        }

        let Some(first) = first else {
            return None;
        };
        // A set is contiguous when its members fill its span.
        if count != last - first + 1 {
            return None;
        }
        Some(RangeInclusive {
            start: first as u8,
            last: last as u8,
        })
    }

    pub(crate) const fn write_members<const N: usize>(&self, members: &mut [u8; N]) -> Option<u8> {
        let mut count = 0;
        let mut i = 0;
        while i < TABLE_BYTES / 8 {
            let mut word = self.word(i);
            while word != 0 {
                if count == N {
                    return None;
                }
                members[count] = (i * 64) as u8 + word.trailing_zeros() as u8;
                word &= word - 1;
                count += 1;
            }
            i += 1;
        }
        Some(count as u8)
    }

    const fn word(&self, i: usize) -> u64 {
        let b = i * 8;
        u64::from_le_bytes([
            self.0[b],
            self.0[b + 1],
            self.0[b + 2],
            self.0[b + 3],
            self.0[b + 4],
            self.0[b + 5],
            self.0[b + 6],
            self.0[b + 7],
        ])
    }
}

impl Extend<u8> for ByteSet {
    fn extend<T: IntoIterator<Item = u8>>(&mut self, iter: T) {
        for byte in iter {
            self.add(byte);
        }
    }
}

impl FromIterator<u8> for ByteSet {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        let mut set = Self::new();
        set.extend(iter);
        set
    }
}
