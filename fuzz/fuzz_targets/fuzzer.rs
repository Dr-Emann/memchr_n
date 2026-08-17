#![no_main]

use libfuzzer_sys::fuzz_target;
use memchr_n::Backend;

fuzz_target!(|haystack: (bool, [u64; 4], &[u8])| {
    let (allow_simd, byte_set, haystack) = haystack;
    target(allow_simd, ByteSet(byte_set), haystack);
});

struct ByteSet([u64; 4]);

impl ByteSet {
    fn contains(&self, byte: u8) -> bool {
        let word_idx = byte / 64;
        let bit_idx = byte % 64;
        (self.0[word_idx as usize] & (1 << bit_idx)) != 0
    }
}

fn target(allow_simd: bool, byte_set: ByteSet, haystack: &[u8]) {
    let finder = finder_for_byte_set(allow_simd, &byte_set);
    let mut previous_match = None;
    let mut count = 0;
    for idx in finder.iter(haystack) {
        let non_matching_start = previous_match.map_or(0, |previous_match| previous_match + 1);
        assert_eq!(
            finder.find(&haystack[non_matching_start..]),
            Some(idx - non_matching_start)
        );
        for &byte in &haystack[non_matching_start..idx] {
            assert!(!byte_set.contains(byte))
        }
        assert!(byte_set.contains(haystack[idx]));
        count += 1;

        previous_match = Some(idx);
    }

    assert_eq!(finder.iter(haystack).count(), count);
}

fn finder_for_byte_set(allow_simd: bool, byte_set: &ByteSet) -> memchr_n::MemchrN {
    let mut bytes = Vec::new();
    for (i, mut chunk) in byte_set.0.iter().copied().enumerate() {
        let base = (i * 64) as u8;
        while chunk != 0 {
            let bit = chunk.trailing_zeros() as u8;

            bytes.push(base + bit);

            chunk &= chunk - 1;
        }
    }
    memchr_n::MemchrN::new_with_backend(&bytes, if allow_simd { Backend::Auto } else { Backend::Swar })
}
