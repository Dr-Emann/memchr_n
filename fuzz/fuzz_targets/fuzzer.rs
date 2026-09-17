#![no_main]

use libfuzzer_sys::fuzz_target;
use memchr_n::{Backend, ByteSet};

fuzz_target!(|haystack: (bool, [u64; 4], &[u8])| {
    let (allow_simd, byte_set, haystack) = haystack;
    target(allow_simd, byte_set, haystack);
});

fn target(allow_simd: bool, words: [u64; 4], haystack: &[u8]) {
    let byte_set = byte_set_from_words(words);
    let backend = if allow_simd {
        Backend::Auto
    } else {
        Backend::Swar
    };
    let finder = memchr_n::MemchrN::from_byte_set_with_backend(byte_set, backend);

    let mut expected = Vec::new();
    for (offset, &byte) in haystack.iter().enumerate() {
        if byte_set.contains(byte) {
            expected.push(offset);
        }
    }

    assert_eq!(finder.iter(haystack).collect::<Vec<_>>(), expected);
    assert_eq!(finder.iter(haystack).count(), expected.len());

    let mut start = 0;
    for offset in expected {
        assert_eq!(finder.find(&haystack[start..]), Some(offset - start));
        start = offset + 1;
    }
    assert_eq!(finder.find(&haystack[start..]), None);
}

fn byte_set_from_words(words: [u64; 4]) -> ByteSet {
    let mut byte_set = ByteSet::new();
    for (i, mut word) in words.into_iter().enumerate() {
        let base = (i * 64) as u8;
        while word != 0 {
            byte_set.add(base + word.trailing_zeros() as u8);
            word &= word - 1;
        }
    }
    byte_set
}
