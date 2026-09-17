use core::fmt;
use core::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Bound, Not, RangeBounds,
    RangeInclusive, Sub, SubAssign,
};

const TABLE_BITS: usize = 256;
const TABLE_BYTES: usize = TABLE_BITS / u8::BITS as usize;

#[derive(Copy, Clone)]
pub(crate) struct ByteRange {
    pub(crate) start: u8,
    pub(crate) last: u8,
}

/// A set of byte values to search for.
///
/// A set is built up one byte, range, or slice at a time, then handed to
/// [`MemchrN::from_byte_set`](crate::MemchrN::from_byte_set). Membership is one bit per value, so a
/// set occupies 32 bytes no matter how many members it holds, and adding a byte that is already a
/// member changes nothing.
///
/// Two sets combine with the bit operators: `|` is [`union`](Self::union), `&` is
/// [`intersection`](Self::intersection), `^` is
/// [`symmetric_difference`](Self::symmetric_difference), `-` is
/// [`difference`](Self::difference), and `!` is the complement. Every method is usable in a `const`
/// context; the operators are not, so a `const` set combines through the named methods.
///
/// # Examples
///
/// ```
/// use memchr_n::{ByteSet, MemchrN};
///
/// let mut identifier = ByteSet::from_bytes(b"_");
/// identifier.add_range(b'0'..=b'9');
/// identifier.add_range(b'a'..=b'z');
///
/// let finder = MemchrN::from_byte_set(identifier);
/// assert_eq!(finder.find(b"?! x"), Some(3));
/// ```
///
/// Building the same set in a `const`:
///
/// ```
/// use memchr_n::ByteSet;
///
/// const IDENTIFIER: ByteSet = {
///     let mut set = ByteSet::from_bytes(b"_");
///     set.add_range(b'0'..=b'9');
///     set.add_range(b'a'..=b'z');
///     set
/// };
/// assert!(IDENTIFIER.contains(b'x'));
/// ```
#[derive(Copy, Clone, Default, PartialEq, Eq)]
// The bitset kernel loads the whole table into a vector register.
#[repr(align(16))]
pub struct ByteSet([u8; TABLE_BYTES]);

impl ByteSet {
    /// Creates a set with no members.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let mut set = ByteSet::new();
    /// assert!(set.is_empty());
    ///
    /// set.add(b'x');
    /// assert!(set.contains(b'x'));
    /// ```
    pub const fn new() -> Self {
        Self([0; TABLE_BYTES])
    }

    /// Creates a set containing every byte value.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let mut set = ByteSet::full();
    /// set.remove(b'\n');
    /// assert!(set.contains(b'x'));
    /// assert!(!set.contains(b'\n'));
    /// ```
    pub const fn full() -> Self {
        Self([u8::MAX; TABLE_BYTES])
    }

    /// Creates a set of the distinct values in `bytes`.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let set = ByteSet::from_bytes(b"aeiou");
    /// assert_eq!(set.len(), 5);
    /// assert!(set.contains(b'e'));
    /// ```
    pub const fn from_bytes(bytes: &[u8]) -> Self {
        let mut set = Self::new();
        set.add_all(bytes);
        set
    }

    /// Adds `byte` to the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let mut set = ByteSet::new();
    /// set.add(b'?');
    /// set.add(b'?');
    /// assert_eq!(set.len(), 1);
    /// ```
    pub const fn add(&mut self, byte: u8) {
        let table_index = (byte / 8) as usize;
        let bit_index = byte % 8;
        let mask = 1 << bit_index;
        self.0[table_index] |= mask;
    }

    /// Adds every value in `bytes` to the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let mut set = ByteSet::from_bytes(b"aeiou");
    /// set.add_all(b"AEIOU");
    /// assert_eq!(set.len(), 10);
    /// assert!(set.contains(b'A'));
    /// ```
    pub const fn add_all(&mut self, bytes: &[u8]) {
        let mut i = 0;
        while i < bytes.len() {
            self.add(bytes[i]);
            i += 1;
        }
    }

    /// Removes `byte` from the set.
    ///
    /// Removing a byte that is not a member has no effect.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let mut set = ByteSet::from_bytes(b"aeiou");
    /// set.remove(b'e');
    /// assert!(!set.contains(b'e'));
    /// ```
    pub const fn remove(&mut self, byte: u8) {
        let table_index = (byte / 8) as usize;
        let bit_index = byte % 8;
        let mask = 1 << bit_index;
        self.0[table_index] &= !mask;
    }

    /// Adds every byte in an inclusive range to the set.
    ///
    /// A range whose start is past its end adds nothing. Only [`RangeInclusive<u8>`] is accepted,
    /// because a generic [`RangeBounds<u8>`] cannot be taken apart in a `const` context; use
    /// `0..=last` and `start..=255` for the half-open ends of the byte domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let mut set = ByteSet::new();
    /// set.add_range(b'0'..=b'9');
    /// set.add_range(0..=b' ');
    /// assert!(set.contains(b'7'));
    /// assert!(set.contains(b'\n'));
    /// ```
    pub const fn add_range(&mut self, range: RangeInclusive<u8>) {
        self.add_byte_range(ByteRange {
            start: *range.start(),
            last: *range.end(),
        });
    }

    /// Removes every byte in an inclusive range from the set.
    ///
    /// A range whose start is past its end removes nothing. Only [`RangeInclusive<u8>`] is
    /// accepted, for the reason given on [`add_range`](Self::add_range).
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let mut set = ByteSet::full();
    /// set.remove_range(b'a'..=b'z');
    /// assert!(set.contains(b'A'));
    /// assert!(!set.contains(b'a'));
    /// ```
    pub const fn remove_range(&mut self, range: RangeInclusive<u8>) {
        let mut removed = Self::new();
        removed.add_range(range);
        *self = self.difference(removed);
    }

    /// Replaces the set with its complement.
    ///
    /// Every member is removed, and every byte that was not a member becomes one.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let mut set = ByteSet::from_bytes(b" \t\n");
    /// set.invert();
    /// assert!(set.contains(b'x'));
    /// assert!(!set.contains(b' '));
    /// ```
    pub const fn invert(&mut self) {
        let mut i = 0;
        while i < self.0.len() {
            self.0[i] = !self.0[i];
            i += 1;
        }
    }

    /// Returns the set of bytes belonging to either set, as the `|` operator does.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let vowels = ByteSet::from_bytes(b"aeiou");
    /// let digits = ByteSet::from_bytes(b"0123456789");
    /// assert_eq!(vowels.union(digits), vowels | digits);
    /// assert_eq!(vowels.union(digits).len(), 15);
    /// ```
    pub const fn union(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < TABLE_BYTES {
            self.0[i] |= other.0[i];
            i += 1;
        }
        self
    }

    /// Returns the set of bytes belonging to both sets, as the `&` operator does.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let vowels = ByteSet::from_bytes(b"aeiou");
    /// let hex = ByteSet::from_bytes(b"0123456789abcdef");
    /// assert_eq!(vowels.intersection(hex), vowels & hex);
    /// assert_eq!(vowels.intersection(hex), ByteSet::from_bytes(b"ae"));
    /// ```
    pub const fn intersection(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < TABLE_BYTES {
            self.0[i] &= other.0[i];
            i += 1;
        }
        self
    }

    /// Returns the members of `self` that are not members of `other`, as the `-` operator does.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let hex = ByteSet::from_bytes(b"0123456789abcdef");
    /// let vowels = ByteSet::from_bytes(b"aeiou");
    /// assert_eq!(hex.difference(vowels), hex - vowels);
    /// assert!(!hex.difference(vowels).contains(b'a'));
    /// assert!(hex.difference(vowels).contains(b'b'));
    /// ```
    pub const fn difference(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < TABLE_BYTES {
            self.0[i] &= !other.0[i];
            i += 1;
        }
        self
    }

    /// Returns the bytes belonging to exactly one of the sets, as the `^` operator does.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let lower = ByteSet::from_bytes(b"abc");
    /// let vowels = ByteSet::from_bytes(b"aeiou");
    /// assert_eq!(lower.symmetric_difference(vowels), lower ^ vowels);
    /// assert_eq!(lower.symmetric_difference(vowels), ByteSet::from_bytes(b"bceiou"));
    /// ```
    pub const fn symmetric_difference(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < TABLE_BYTES {
            self.0[i] ^= other.0[i];
            i += 1;
        }
        self
    }

    /// Returns `true` when `byte` is a member of the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// let set = ByteSet::from_bytes(b"aeiou");
    /// assert!(set.contains(b'a'));
    /// assert!(!set.contains(b'z'));
    /// ```
    pub const fn contains(&self, byte: u8) -> bool {
        let table_index = (byte / 8) as usize;
        let bit_index = byte % 8;
        self.0[table_index] & (1 << bit_index) != 0
    }

    /// Returns the number of members, between zero and 256.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// assert_eq!(ByteSet::from_bytes(b"hello").len(), 4);
    /// assert_eq!(ByteSet::full().len(), 256);
    /// ```
    pub const fn len(&self) -> usize {
        let mut len = 0;
        let mut i = 0;
        while i < TABLE_BYTES / 8 {
            len += self.word(i).count_ones() as usize;
            i += 1;
        }
        len
    }

    /// Returns `true` when the set has no members.
    ///
    /// A [`MemchrN`](crate::MemchrN) built from an empty set never matches.
    ///
    /// # Examples
    ///
    /// ```
    /// use memchr_n::ByteSet;
    ///
    /// assert!(ByteSet::new().is_empty());
    /// assert!(!ByteSet::from_bytes(b"x").is_empty());
    /// ```
    pub const fn is_empty(&self) -> bool {
        let mut i = 0;
        while i < TABLE_BYTES / 8 {
            if self.word(i) != 0 {
                return false;
            }
            i += 1;
        }
        true
    }

    pub(crate) fn as_array(&self) -> &[u8; 32] {
        &self.0
    }

    pub(crate) const fn add_byte_range(&mut self, ByteRange { start, last }: ByteRange) {
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

    /// Returns the set as a 16-by-16 bit matrix indexed by high and low nibble.
    ///
    /// Bit `lo` of `rows[hi]` is set exactly when the byte `(hi << 4) | lo` is a member.
    /// Each row occupies two consecutive storage bytes; little-endian decoding keeps
    /// low nibbles 0 through 7 in the low byte and 8 through 15 in the high byte.
    pub(crate) fn nibble_rows(&self) -> [u16; 16] {
        let mut rows = [0; 16];
        for (hi, row) in rows.iter_mut().enumerate() {
            *row = u16::from_le_bytes([self.0[hi * 2], self.0[hi * 2 + 1]]);
        }
        rows
    }

    pub(crate) fn excluded_byte(&self) -> Option<u8> {
        let mut excluded = None;
        for i in 0..TABLE_BYTES / 8 {
            let missing = !self.word(i);
            if missing == 0 {
                continue;
            }
            if excluded.is_some() || missing.count_ones() != 1 {
                return None;
            }
            excluded = Some((i * 64) as u8 + missing.trailing_zeros() as u8);
        }
        excluded
    }

    pub(crate) const fn as_contiguous_range(&self) -> Option<ByteRange> {
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
        Some(ByteRange {
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

pub(crate) fn inclusive_range(range: impl RangeBounds<u8>) -> Option<ByteRange> {
    let start = match range.start_bound() {
        Bound::Included(start) => *start,
        Bound::Excluded(start) => start.checked_add(1)?,
        Bound::Unbounded => u8::MIN,
    };
    let last = match range.end_bound() {
        Bound::Included(last) => *last,
        Bound::Excluded(last) => last.checked_sub(1)?,
        Bound::Unbounded => u8::MAX,
    };
    if start > last {
        return None;
    }
    Some(ByteRange { start, last })
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

impl Not for ByteSet {
    type Output = Self;

    fn not(mut self) -> Self {
        self.invert();
        self
    }
}

impl BitOr for ByteSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl BitOrAssign for ByteSet {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

impl BitAnd for ByteSet {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        self.intersection(rhs)
    }
}

impl BitAndAssign for ByteSet {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = self.intersection(rhs);
    }
}

impl BitXor for ByteSet {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        self.symmetric_difference(rhs)
    }
}

impl BitXorAssign for ByteSet {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = self.symmetric_difference(rhs);
    }
}

impl Sub for ByteSet {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self.difference(rhs)
    }
}

impl SubAssign for ByteSet {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.difference(rhs);
    }
}

impl fmt::Debug for ByteSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut members = f.debug_set();
        for byte in 0..=u8::MAX {
            if self.contains(byte) {
                members.entry(&format_args!("b'{}'", byte.escape_ascii()));
            }
        }
        members.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_exactly_one_excluded_byte() {
        assert_eq!(ByteSet::new().excluded_byte(), None);
        let mut full = ByteSet::new();
        for byte in 0..=u8::MAX {
            full.add(byte);
        }
        assert_eq!(full.excluded_byte(), None);
        for excluded in 0..=u8::MAX {
            let mut set = ByteSet::new();
            for byte in 0..=u8::MAX {
                if byte != excluded {
                    set.add(byte);
                }
            }
            assert_eq!(set.excluded_byte(), Some(excluded));
            for second in 0..=u8::MAX {
                if second == excluded {
                    continue;
                }
                let mut set = set;
                set.0[usize::from(second / 8)] &= !(1 << (second % 8));
                assert_eq!(set.excluded_byte(), None);
            }
        }
    }

    #[test]
    fn full_contains_every_byte() {
        let full = ByteSet::full();
        for byte in 0..=u8::MAX {
            assert!(full.contains(byte), "{byte}");
        }
        assert_eq!(full.len(), 256);

        let mut inverted = ByteSet::new();
        inverted.invert();
        assert_eq!(inverted, full);
    }

    #[test]
    fn remove_affects_only_its_byte() {
        for byte in 0..=u8::MAX {
            let mut set = ByteSet::full();
            set.remove(byte);
            for other in 0..=u8::MAX {
                let expected = other != byte;
                assert_eq!(set.contains(other), expected, "{other} without {byte}");
            }
            set.remove(byte);
            set.add(byte);
            assert_eq!(set, ByteSet::full());
        }
    }

    #[test]
    fn remove_range_removes_exactly_its_members() {
        let bounds = [0u8, 1, 7, 8, 63, 64, 127, 128, 200, 254, 255];
        for &start in &bounds {
            for &last in bounds.iter().filter(|&&b| b >= start) {
                let mut set = ByteSet::full();
                set.remove_range(start..=last);
                for byte in 0..=u8::MAX {
                    let expected = byte < start || byte > last;
                    assert_eq!(set.contains(byte), expected, "{byte} in {start}..={last}");
                }
            }
        }
    }

    #[test]
    // A range that starts past its end is what the test is about.
    #[expect(clippy::reversed_empty_ranges)]
    fn remove_range_of_empty_range_removes_nothing() {
        let mut set = ByteSet::from_bytes(b"abc");
        let before = set;
        set.remove_range(10..=9);
        set.remove_range(u8::MAX..=u8::MIN);
        assert_eq!(set, before);
    }

    #[test]
    fn len_counts_distinct_members() {
        assert_eq!(ByteSet::new().len(), 0);
        assert!(ByteSet::new().is_empty());

        let mut set = ByteSet::new();
        for byte in 0..=u8::MAX {
            assert_eq!(set.len(), usize::from(byte));
            set.add(byte);
            set.add(byte);
            assert!(!set.is_empty());
        }
        assert_eq!(set.len(), 256);
    }

    #[test]
    fn debug_lists_members_as_byte_literals() {
        let set = ByteSet::from_bytes(b"a\nz'");
        assert_eq!(format!("{set:?}"), r"{b'\n', b'\'', b'a', b'z'}");
    }

    #[test]
    fn builds_in_const_context() {
        const NOT_SPACE: ByteSet = {
            let mut set = ByteSet::from_bytes(b" \t");
            set.add(b'\n');
            set.remove(b'\t');
            set.invert();
            set
        };
        assert!(NOT_SPACE.contains(b'\t'));
        assert!(!NOT_SPACE.contains(b' '));
        assert!(!NOT_SPACE.contains(b'\n'));
        assert_eq!(NOT_SPACE.len(), 254);
    }

    #[test]
    fn ranges_and_combinations_work_in_const_context() {
        const IDENTIFIER: ByteSet = {
            let mut set = ByteSet::from_bytes(b"_");
            set.add_range(b'a'..=b'z');
            set.add_range(b'0'..=b'9');
            set
        };
        const NOT_ASCII: ByteSet = {
            let mut set = ByteSet::full();
            set.remove_range(0..=0x7F);
            set
        };
        const HEX: ByteSet = {
            let mut set = ByteSet::new();
            set.add_range(b'0'..=b'9');
            set.add_range(b'a'..=b'f');
            set
        };
        const BEYOND_HEX: ByteSet = IDENTIFIER.difference(HEX);
        const EITHER: ByteSet = IDENTIFIER.union(NOT_ASCII);

        assert_eq!(IDENTIFIER.len(), 37);
        assert_eq!(NOT_ASCII.len(), 128);
        assert_eq!(IDENTIFIER.intersection(HEX), HEX);
        assert!(BEYOND_HEX.contains(b'g'));
        assert!(!BEYOND_HEX.contains(b'a'));
        assert_eq!(EITHER.symmetric_difference(NOT_ASCII), IDENTIFIER);
    }

    #[test]
    fn add_all_unions_in_the_slice() {
        let mut set = ByteSet::from_bytes(b"aeiou");
        set.add_all(b"");
        assert_eq!(set, ByteSet::from_bytes(b"aeiou"));

        set.add_all(b"eeAEIOU\xff");
        assert_eq!(
            set,
            ByteSet::from_bytes(b"aeiou") | ByteSet::from_bytes(b"AEIOU\xff")
        );
        assert_eq!(set.len(), 11);
    }

    #[test]
    fn bit_operators_match_member_by_member() {
        let left = ByteSet::from_bytes(b"abcdef0123");
        let right = ByteSet::from_bytes(b"aeiou0123456789");

        for byte in 0..=u8::MAX {
            let in_left = left.contains(byte);
            let in_right = right.contains(byte);
            assert_eq!((left | right).contains(byte), in_left || in_right, "{byte}");
            assert_eq!((left & right).contains(byte), in_left && in_right, "{byte}");
            assert_eq!((left ^ right).contains(byte), in_left != in_right, "{byte}");
            assert_eq!(
                (left - right).contains(byte),
                in_left && !in_right,
                "{byte}"
            );
            assert_eq!((!left).contains(byte), !in_left, "{byte}");
        }
    }

    #[test]
    fn assigning_operators_match_their_binary_forms() {
        let left = ByteSet::from_bytes(b"abcdef0123");
        let right = ByteSet::from_bytes(b"aeiou0123456789");

        let mut union = left;
        union |= right;
        let mut intersection = left;
        intersection &= right;
        let mut symmetric_difference = left;
        symmetric_difference ^= right;
        let mut difference = left;
        difference -= right;

        assert_eq!(union, left | right);
        assert_eq!(intersection, left & right);
        assert_eq!(symmetric_difference, left ^ right);
        assert_eq!(difference, left - right);
    }
}
