#![no_main]

use libfuzzer_sys::fuzz_target;
use memchr_n::MemchrN;

fuzz_target!(|data: &[u8]| {
    let Some((&needle, haystack)) = data.split_first() else {
        return;
    };
    MemchrN::new(&[needle]).iter(haystack).count();
});
