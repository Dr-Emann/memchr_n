#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use memchr_n::MemchrN;

#[derive(Arbitrary, Debug)]
enum Operation {
    Next,
    Nth(u8),
    AdvanceTo(u16),
}

#[derive(Arbitrary, Debug)]
struct Input<'a> {
    needle: u8,
    operations: Vec<Operation>,
    haystack: &'a [u8],
}

fuzz_target!(|input: Input<'_>| {
    let Input {
        needle,
        operations,
        haystack,
    } = input;
    let finder = MemchrN::new(&[needle]);
    let mut iter = finder.iter(haystack);
    let mut cursor = 0;

    for operation in operations {
        match operation {
            Operation::Next => {
                assert_eq!(iter.next(), scalar_next(needle, haystack, &mut cursor));
            }
            Operation::Nth(n) => {
                let n = usize::from(n);
                let mut expected = None;
                for _ in 0..=n {
                    expected = scalar_next(needle, haystack, &mut cursor);
                    if expected.is_none() {
                        break;
                    }
                }
                assert_eq!(iter.nth(n), expected);
            }
            Operation::AdvanceTo(idx) => {
                let idx = usize::from(idx) % haystack.len().saturating_add(2);
                iter.advance_to(idx);
                cursor = cursor.max(idx);
            }
        }
    }

    loop {
        let expected = scalar_next(needle, haystack, &mut cursor);
        let actual = iter.next();
        assert_eq!(actual, expected);
        if expected.is_none() {
            break;
        }
    }
    iter.advance_to(0);
    assert_eq!(iter.next(), None);
});

fn scalar_next(needle: u8, haystack: &[u8], cursor: &mut usize) -> Option<usize> {
    for (offset, &byte) in haystack.iter().enumerate().skip(*cursor) {
        if byte == needle {
            *cursor = offset + 1;
            return Some(offset);
        }
    }
    *cursor = haystack.len();
    None
}
