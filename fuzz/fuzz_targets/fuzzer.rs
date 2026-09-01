#![no_main]

use libfuzzer_sys::fuzz_target;
use memchr_n::Backend;

fuzz_target!(|data: (bool, [u64; 4], &[u8])| {
    let (simd, bitset, data) = data;
    target(simd, Bitset(bitset), data);
});

struct Bitset([u64; 4]);

impl Bitset {
    fn get(&self, b: u8) -> bool {
        let word_idx = b / 64;
        let bit_idx = b % 64;
        (self.0[word_idx as usize] & (1 << bit_idx)) != 0
    }
}

fn target(simd: bool, bitset: Bitset, data: &[u8]) {
    let finder = finder_for_bitset(simd, &bitset);
    let mut prev = None;
    let mut count = 0;
    for idx in finder.iter(data) {
        let non_matching_start = prev.map_or(0, |prev| prev + 1);
        for &b in &data[non_matching_start..idx] {
            assert!(!bitset.get(b))
        }
        assert!(bitset.get(data[idx]));
        count += 1;

        prev = Some(idx);
    }

    assert_eq!(finder.iter(data).count(), count);
}

fn finder_for_bitset(simd: bool, bitset: &Bitset) -> memchr_n::MemchrN {
    let mut bytes = Vec::new();
    for (i, mut chunk) in bitset.0.iter().copied().enumerate() {
        let base = (i * 64) as u8;
        while chunk != 0 {
            let bit = chunk.trailing_zeros() as u8;

            bytes.push(base + bit);

            chunk &= chunk - 1;
        }
    }
    memchr_n::MemchrN::new_with(&bytes, if simd { Backend::Auto } else { Backend::Scalar })
}
