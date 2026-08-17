use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use memchr_n::{Backend, MemchrN};
use std::hint::black_box;
use std::time::Duration;

const SHERLOCK: &[u8] = include_bytes!("haystacks/sherlock/huge.txt");
const ALNUM: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const HEX_LOWER: &[u8] = b"0123456789abcdef";
const OURS: &str = "memchr_n";
const THEIRS: &str = "memchr";

#[derive(Copy, Clone)]
enum ByteSet {
    List(&'static [u8]),
    Range(u8, u8),
}

impl ByteSet {
    fn finder(self, backend: Backend) -> MemchrN {
        match self {
            ByteSet::List(bytes) => MemchrN::new_with_backend(bytes, backend),
            ByteSet::Range(start, end) => MemchrN::from_range_with_backend(start..=end, backend),
        }
    }

    fn contains(self, byte: u8) -> bool {
        match self {
            ByteSet::List(bytes) => bytes.contains(&byte),
            ByteSet::Range(start, end) => start <= byte && byte <= end,
        }
    }
}

#[derive(Copy, Clone)]
enum Needles {
    One(u8),
    Two(u8, u8),
    Three(u8, u8, u8),
}

impl Needles {
    fn from_set(set: ByteSet) -> Option<Self> {
        let ByteSet::List(bytes) = set else {
            return None;
        };
        match *bytes {
            [a] => Some(Self::One(a)),
            [a, b] => Some(Self::Two(a, b)),
            [a, b, c] => Some(Self::Three(a, b, c)),
            _ => None,
        }
    }

    fn find(self, haystack: &[u8]) -> Option<usize> {
        match self {
            Self::One(a) => memchr::memchr(a, haystack),
            Self::Two(a, b) => memchr::memchr2(a, b, haystack),
            Self::Three(a, b, c) => memchr::memchr3(a, b, c, haystack),
        }
    }

    fn count(self, haystack: &[u8]) -> usize {
        match self {
            Self::One(a) => memchr::memchr_iter(a, haystack).count(),
            Self::Two(a, b) => memchr::memchr2_iter(a, b, haystack).count(),
            Self::Three(a, b, c) => memchr::memchr3_iter(a, b, c, haystack).count(),
        }
    }

    fn offset_sum(self, haystack: &[u8]) -> usize {
        let mut sum = 0usize;
        match self {
            Self::One(a) => {
                for offset in memchr::memchr_iter(a, haystack) {
                    sum = sum.wrapping_add(offset);
                }
            }
            Self::Two(a, b) => {
                for offset in memchr::memchr2_iter(a, b, haystack) {
                    sum = sum.wrapping_add(offset);
                }
            }
            Self::Three(a, b, c) => {
                for offset in memchr::memchr3_iter(a, b, c, haystack) {
                    sum = sum.wrapping_add(offset);
                }
            }
        }
        sum
    }
}

#[derive(Copy, Clone)]
enum BuildCase {
    Bytes(&'static [u8]),
    Range(u8, u8),
}

impl BuildCase {
    fn build(self) -> MemchrN {
        match self {
            Self::Bytes(bytes) => MemchrN::new(bytes),
            Self::Range(start, end) => MemchrN::from_range(start..=end),
        }
    }
}

const BUILD_CASES: &[(&str, BuildCase)] = &[
    ("empty", BuildCase::Bytes(b"")),
    ("one-byte", BuildCase::Bytes(b"z")),
    ("three-bytes", BuildCase::Bytes(b"zRJ")),
    ("one-range", BuildCase::Range(b'0', b'9')),
    ("small-set", BuildCase::Bytes(b"aeiouAEI")),
    ("fixed-nibble", BuildCase::Bytes(b"abcdefghjl")),
    ("bitset-16", BuildCase::Bytes(HEX_LOWER)),
    ("bitset-62", BuildCase::Bytes(ALNUM)),
];

const COUNT_CASES: &[(&str, ByteSet)] = &[
    ("rare-one", ByteSet::List(b"z")),
    ("common-one", ByteSet::List(b"a")),
    ("rare-two", ByteSet::List(b"zR")),
    ("common-three", ByteSet::List(b"ato")),
    ("one-range", ByteSet::Range(b'0', b'9')),
    ("small-set", ByteSet::List(b"aeiouAEI")),
    ("fixed-nibble", ByteSet::List(b"abcdefghjl")),
    ("bitset-16", ByteSet::List(HEX_LOWER)),
    ("bitset-62", ByteSet::List(ALNUM)),
];

const ITERATE_CASES: &[(&str, ByteSet)] = &[
    ("rare-one", ByteSet::List(b"z")),
    ("common-one", ByteSet::List(b"a")),
    ("common-three", ByteSet::List(b"ato")),
    ("bitset-16", ByteSet::List(HEX_LOWER)),
];

fn expected_count(set: ByteSet, haystack: &[u8]) -> usize {
    let mut count = 0;
    for &byte in haystack {
        if set.contains(byte) {
            count += 1;
        }
    }
    count
}

fn expected_find(set: ByteSet, haystack: &[u8]) -> Option<usize> {
    for (offset, &byte) in haystack.iter().enumerate() {
        if set.contains(byte) {
            return Some(offset);
        }
    }
    None
}

fn expected_offset_sum(set: ByteSet, haystack: &[u8]) -> usize {
    let mut sum = 0usize;
    for (offset, &byte) in haystack.iter().enumerate() {
        if set.contains(byte) {
            sum = sum.wrapping_add(offset);
        }
    }
    sum
}

fn offset_sum(finder: &MemchrN, haystack: &[u8]) -> usize {
    let mut sum = 0usize;
    for offset in finder.iter(haystack) {
        sum = sum.wrapping_add(offset);
    }
    sum
}

fn verified_finder(set: ByteSet, backend: Backend, haystack: &[u8], label: &str) -> MemchrN {
    let finder = set.finder(backend);
    assert_eq!(
        finder.find(haystack),
        expected_find(set, haystack),
        "first-match mismatch for {label}"
    );
    assert_eq!(
        finder.iter(haystack).count(),
        expected_count(set, haystack),
        "count mismatch for {label}"
    );
    assert_eq!(
        offset_sum(&finder, haystack),
        expected_offset_sum(set, haystack),
        "offset mismatch for {label}"
    );
    finder
}

fn verified_needles(set: ByteSet, haystack: &[u8], label: &str) -> Option<Needles> {
    let needles = Needles::from_set(set)?;
    assert_eq!(
        needles.find(haystack),
        expected_find(set, haystack),
        "memchr first-match mismatch for {label}"
    );
    assert_eq!(
        needles.count(haystack),
        expected_count(set, haystack),
        "memchr count mismatch for {label}"
    );
    assert_eq!(
        needles.offset_sum(haystack),
        expected_offset_sum(set, haystack),
        "memchr offset mismatch for {label}"
    );
    Some(needles)
}

fn planted_haystack(set: ByteSet, offset: Option<usize>) -> [u8; 128] {
    let mut haystack = [b'.'; 128];
    let Some(offset) = offset else {
        return haystack;
    };
    let byte = match set {
        ByteSet::List(bytes) => bytes[0],
        ByteSet::Range(start, _) => start,
    };
    haystack[offset] = byte;
    haystack
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("build");
    for &(name, case) in BUILD_CASES {
        group.bench_function(name, |b| b.iter(|| black_box(case).build()));
    }
    group.finish();
}

fn bench_find(c: &mut Criterion) {
    let mut group = c.benchmark_group("find");
    let one_byte = ByteSet::List(b"x");
    for (name, offset) in [
        ("hit-000", Some(0usize)),
        ("hit-015", Some(15)),
        ("hit-016", Some(16)),
        ("hit-063", Some(63)),
        ("hit-064", Some(64)),
        ("hit-127", Some(127)),
        ("miss-128", None),
    ] {
        let haystack = planted_haystack(one_byte, offset);
        let label = format!("one-byte/{name}");
        let finder = verified_finder(one_byte, Backend::Auto, &haystack, &label);
        let needles = verified_needles(one_byte, &haystack, &label).unwrap();
        group.bench_function(BenchmarkId::new(OURS, &label), |b| {
            b.iter(|| black_box(&finder).find(black_box(&haystack)))
        });
        group.bench_function(BenchmarkId::new(THEIRS, &label), |b| {
            b.iter(|| black_box(needles).find(black_box(&haystack)))
        });
    }

    let bitset = ByteSet::List(HEX_LOWER);
    for (name, offset) in [
        ("hit-000", Some(0usize)),
        ("hit-064", Some(64)),
        ("miss-128", None),
    ] {
        let haystack = planted_haystack(bitset, offset);
        let label = format!("bitset-16/{name}");
        let finder = verified_finder(bitset, Backend::Auto, &haystack, &label);
        group.bench_function(BenchmarkId::new(OURS, &label), |b| {
            b.iter(|| black_box(&finder).find(black_box(&haystack)))
        });
    }

    for (name, set) in [("one-byte", one_byte), ("bitset-16", bitset)] {
        for len in [4096, 65536] {
            let haystack = vec![b'.'; len];
            let label = format!("{name}/miss-{len}");
            let finder = verified_finder(set, Backend::Auto, &haystack, &label);
            group.bench_function(BenchmarkId::new(OURS, &label), |b| {
                b.iter(|| black_box(&finder).find(black_box(&haystack)))
            });
            let Some(needles) = verified_needles(set, &haystack, &label) else {
                continue;
            };
            group.bench_function(BenchmarkId::new(THEIRS, &label), |b| {
                b.iter(|| black_box(needles).find(black_box(&haystack)))
            });
        }
    }

    for len in [7usize, 8, 9] {
        let haystack = &SHERLOCK[..len];
        let label = format!("swar/miss-{len:03}");
        let set = ByteSet::List(b"<");
        let finder = verified_finder(set, Backend::Swar, haystack, &label);
        group.bench_function(BenchmarkId::new(OURS, &label), |b| {
            b.iter(|| black_box(&finder).find(black_box(haystack)))
        });
    }
    group.finish();
}

fn bench_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("count/sherlock");
    group.throughput(Throughput::Bytes(SHERLOCK.len() as u64));
    for &(name, set) in COUNT_CASES {
        let finder = verified_finder(set, Backend::Auto, SHERLOCK, name);
        group.bench_function(BenchmarkId::new(OURS, name), |b| {
            b.iter(|| black_box(&finder).iter(black_box(SHERLOCK)).count())
        });
        let Some(needles) = verified_needles(set, SHERLOCK, name) else {
            continue;
        };
        group.bench_function(BenchmarkId::new(THEIRS, name), |b| {
            b.iter(|| black_box(needles).count(black_box(SHERLOCK)))
        });
    }

    let set = ByteSet::List(HEX_LOWER);
    let finder = verified_finder(set, Backend::Swar, SHERLOCK, "bitset-16/swar");
    group.bench_function(BenchmarkId::new(OURS, "bitset-16/swar"), |b| {
        b.iter(|| black_box(&finder).iter(black_box(SHERLOCK)).count())
    });
    group.finish();
}

fn bench_iterate(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterate/sherlock");
    group.throughput(Throughput::Bytes(SHERLOCK.len() as u64));
    for &(name, set) in ITERATE_CASES {
        let finder = verified_finder(set, Backend::Auto, SHERLOCK, name);
        group.bench_function(BenchmarkId::new(OURS, name), |b| {
            b.iter(|| offset_sum(black_box(&finder), black_box(SHERLOCK)))
        });
        let Some(needles) = verified_needles(set, SHERLOCK, name) else {
            continue;
        };
        group.bench_function(BenchmarkId::new(THEIRS, name), |b| {
            b.iter(|| black_box(needles).offset_sum(black_box(SHERLOCK)))
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = bench_build, bench_find, bench_count, bench_iterate
}
criterion_main!(benches);
