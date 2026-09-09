use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use memchr_n::{Backend, MemchrN};
use std::hint::black_box;

const SHERLOCK_TINY: &[u8] = include_bytes!("haystacks/sherlock/tiny.txt");
const SHERLOCK_SMALL: &[u8] = include_bytes!("haystacks/sherlock/small.txt");
const SHERLOCK_HUGE: &[u8] = include_bytes!("haystacks/sherlock/huge.txt");
const SUBTITLES_EN: &[u8] = include_bytes!("haystacks/opensubtitles/en-huge.txt");
const SUBTITLES_RU: &[u8] = include_bytes!("haystacks/opensubtitles/ru-huge.txt");
const SUBTITLES_ZH: &[u8] = include_bytes!("haystacks/opensubtitles/zh-huge.txt");
const CODE_RUST: &[u8] = include_bytes!("haystacks/code/rust-library.rs");
const MD5: &[u8] = include_bytes!("haystacks/pathological/md5-huge.txt");
const RANDOM: &[u8] = include_bytes!("haystacks/pathological/random-huge.txt");

const ALNUM: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const HEX_LOWER: &[u8] = b"0123456789abcdef";

const OURS: &str = "memchr_n";
const THEIRS: &str = "memchr";

const BACKENDS: &[(&str, Backend)] = &[("auto", Backend::Auto), ("swar", Backend::Swar)];

#[derive(Copy, Clone)]
enum ByteSet {
    List(&'static [u8]),
    Range(u8, u8),
    Stride { start: u8, last: u8, step: u8 },
}

impl ByteSet {
    fn finder(self, backend: Backend) -> MemchrN {
        match self {
            ByteSet::List(bytes) => MemchrN::new_with_backend(bytes, backend),
            ByteSet::Range(start, end) => MemchrN::from_range_with_backend(start..=end, backend),
            ByteSet::Stride { start, last, step } => {
                let mut bytes = Vec::new();
                let mut byte = start;
                while byte <= last {
                    bytes.push(byte);
                    let Some(next) = byte.checked_add(step) else {
                        break;
                    };
                    byte = next;
                }
                MemchrN::new_with_backend(&bytes, backend)
            }
        }
    }

    fn contains(self, byte: u8) -> bool {
        match self {
            ByteSet::List(bytes) => bytes.contains(&byte),
            ByteSet::Range(start, end) => start <= byte && byte <= end,
            ByteSet::Stride { start, last, step } => {
                start <= byte && byte <= last && (byte - start).is_multiple_of(step)
            }
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
    fn from_set(set: ByteSet) -> Option<Needles> {
        let ByteSet::List(bytes) = set else {
            return None;
        };
        match *bytes {
            [n1] => Some(Needles::One(n1)),
            [n1, n2] => Some(Needles::Two(n1, n2)),
            [n1, n2, n3] => Some(Needles::Three(n1, n2, n3)),
            _ => None,
        }
    }

    fn count(self, haystack: &[u8]) -> usize {
        match self {
            Needles::One(n1) => memchr::memchr_iter(n1, haystack).count(),
            Needles::Two(n1, n2) => memchr::memchr2_iter(n1, n2, haystack).count(),
            Needles::Three(n1, n2, n3) => memchr::memchr3_iter(n1, n2, n3, haystack).count(),
        }
    }

    fn first(self, haystack: &[u8]) -> Option<usize> {
        match self {
            Needles::One(n1) => memchr::memchr(n1, haystack),
            Needles::Two(n1, n2) => memchr::memchr2(n1, n2, haystack),
            Needles::Three(n1, n2, n3) => memchr::memchr3(n1, n2, n3, haystack),
        }
    }

    fn offset_sum(self, haystack: &[u8]) -> usize {
        let mut sum = 0usize;
        match self {
            Needles::One(n1) => {
                for offset in memchr::memchr_iter(n1, haystack) {
                    sum = sum.wrapping_add(offset);
                }
            }
            Needles::Two(n1, n2) => {
                for offset in memchr::memchr2_iter(n1, n2, haystack) {
                    sum = sum.wrapping_add(offset);
                }
            }
            Needles::Three(n1, n2, n3) => {
                for offset in memchr::memchr3_iter(n1, n2, n3, haystack) {
                    sum = sum.wrapping_add(offset);
                }
            }
        }
        sum
    }
}

const DENSITY_SETS: &[(&str, ByteSet)] = &[
    ("never1", ByteSet::List(b"<")),
    ("rare1", ByteSet::List(b"z")),
    ("rare2", ByteSet::List(b"zR")),
    ("rare3", ByteSet::List(b"zRJ")),
    ("uncommon1", ByteSet::List(b"b")),
    ("uncommon3", ByteSet::List(b"bp.")),
    ("common1", ByteSet::List(b"a")),
    ("common3", ByteSet::List(b"ato")),
    ("verycommon1", ByteSet::List(b" ")),
];

const KIND_SETS: &[(&str, ByteSet)] = &[
    ("never", ByteSet::List(b"")),
    ("one-byte", ByteSet::List(b"z")),
    ("one-range", ByteSet::Range(b'0', b'9')),
    ("small-set", ByteSet::List(b"aeiouAEI")),
    ("single-nibble", ByteSet::List(b"abcdefghjl")),
    ("any-byte-16", ByteSet::List(HEX_LOWER)),
    ("any-byte-62", ByteSet::List(ALNUM)),
];

const CORPORA: &[(&str, &[u8])] = &[
    ("sherlock", SHERLOCK_HUGE),
    ("subtitles-en", SUBTITLES_EN),
    ("subtitles-ru", SUBTITLES_RU),
    ("subtitles-zh", SUBTITLES_ZH),
    ("code-rust", CODE_RUST),
    ("md5", MD5),
    ("random", RANDOM),
];

const CORPUS_SETS: &[(&str, ByteSet)] = &[
    ("space", ByteSet::List(b" ")),
    ("nonascii", ByteSet::Range(0x80, 0xFF)),
    ("alnum", ByteSet::List(ALNUM)),
];

const SIZE_SETS: &[(&str, ByteSet)] = &[
    ("rare1", ByteSet::List(b"z")),
    ("common1", ByteSet::List(b"a")),
    ("any-byte-16", ByteSet::List(HEX_LOWER)),
];

// Absent sets force tail handling onto the critical path.
const FIRST_SETS: &[(&str, ByteSet)] = &[
    ("never1", ByteSet::List(b"<")),
    (
        "never-any-byte",
        ByteSet::Stride {
            start: 0x80,
            last: 0xFF,
            step: 6,
        },
    ),
];

// Brackets 32- and 64-byte scan boundaries absent from the file-backed sizes.
fn latency_haystacks() -> Vec<(String, &'static [u8])> {
    let mut haystacks = vec![("empty".to_owned(), &SHERLOCK_HUGE[..0])];
    for len in [31usize, 32, 33, 63, 64, 65, 127, 128] {
        haystacks.push((format!("len{len:03}"), &SHERLOCK_HUGE[..len]));
    }
    haystacks.push(("tiny".to_owned(), SHERLOCK_TINY));
    haystacks.push(("small".to_owned(), SHERLOCK_SMALL));
    haystacks.push(("huge".to_owned(), SHERLOCK_HUGE));
    haystacks
}

fn naive_count(set: ByteSet, haystack: &[u8]) -> usize {
    let mut count = 0;
    for &byte in haystack {
        if set.contains(byte) {
            count += 1;
        }
    }
    count
}

fn naive_offset_sum(set: ByteSet, haystack: &[u8]) -> usize {
    let mut sum = 0usize;
    for (offset, &byte) in haystack.iter().enumerate() {
        if set.contains(byte) {
            sum = sum.wrapping_add(offset);
        }
    }
    sum
}

fn naive_first(set: ByteSet, haystack: &[u8]) -> Option<usize> {
    for (offset, &byte) in haystack.iter().enumerate() {
        if set.contains(byte) {
            return Some(offset);
        }
    }
    None
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
        finder.iter(haystack).count(),
        naive_count(set, haystack),
        "count mismatch for {label}"
    );
    assert_eq!(
        finder.iter(haystack).next(),
        naive_first(set, haystack),
        "first-match mismatch for {label}"
    );
    assert_eq!(
        offset_sum(&finder, haystack),
        naive_offset_sum(set, haystack),
        "offset mismatch for {label}"
    );
    finder
}

fn verified_needles(set: ByteSet, haystack: &[u8], label: &str) -> Option<Needles> {
    let needles = Needles::from_set(set)?;
    assert_eq!(
        needles.count(haystack),
        naive_count(set, haystack),
        "memchr count mismatch for {label}"
    );
    assert_eq!(
        needles.first(haystack),
        naive_first(set, haystack),
        "memchr first-match mismatch for {label}"
    );
    assert_eq!(
        needles.offset_sum(haystack),
        naive_offset_sum(set, haystack),
        "memchr offset mismatch for {label}"
    );
    Some(needles)
}

// `memchr_iter().count()` is specialized; its multi-needle iterators count match by match.
fn bench_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("count/sherlock");
    group.throughput(Throughput::Bytes(SHERLOCK_HUGE.len() as u64));
    for &(name, set) in DENSITY_SETS {
        let finder = verified_finder(set, Backend::Auto, SHERLOCK_HUGE, name);
        group.bench_with_input(BenchmarkId::new(OURS, name), &finder, |b, finder| {
            b.iter(|| black_box(finder).iter(black_box(SHERLOCK_HUGE)).count())
        });
        let Some(needles) = verified_needles(set, SHERLOCK_HUGE, name) else {
            continue;
        };
        group.bench_with_input(BenchmarkId::new(THEIRS, name), &needles, |b, &needles| {
            b.iter(|| black_box(needles).count(black_box(SHERLOCK_HUGE)))
        });
    }
    group.finish();
}

fn bench_find_first(c: &mut Criterion) {
    let mut group = c.benchmark_group("find-first/sherlock");
    group.throughput(Throughput::Bytes(SHERLOCK_HUGE.len() as u64));
    for &(name, set) in DENSITY_SETS {
        let finder = verified_finder(set, Backend::Auto, SHERLOCK_HUGE, name);
        group.bench_with_input(BenchmarkId::new(OURS, name), &finder, |b, finder| {
            b.iter(|| black_box(finder).find(black_box(SHERLOCK_HUGE)))
        });
        let Some(needles) = verified_needles(set, SHERLOCK_HUGE, name) else {
            continue;
        };
        group.bench_with_input(BenchmarkId::new(THEIRS, name), &needles, |b, &needles| {
            b.iter(|| black_box(needles).first(black_box(SHERLOCK_HUGE)))
        });
    }
    group.finish();
}

fn bench_first_call(c: &mut Criterion) {
    let mut group = c.benchmark_group("first-call");
    for &(name, set) in KIND_SETS {
        let finder = set.finder(Backend::Auto);
        let needles = Needles::from_set(set);
        for len in [1usize, 4, 8, 16, 32, 64, 256] {
            let haystack = &SHERLOCK_HUGE[..len];
            let param = format!("{name}/{len}");
            group.bench_with_input(BenchmarkId::new("find", &param), &finder, |b, finder| {
                b.iter(|| black_box(finder).find(black_box(haystack)))
            });
            group.bench_with_input(
                BenchmarkId::new("iter-next", &param),
                &finder,
                |b, finder| b.iter(|| black_box(finder).iter(black_box(haystack)).next()),
            );
            if let Some(needles) = needles {
                group.bench_with_input(
                    BenchmarkId::new(THEIRS, &param),
                    &needles,
                    |b, &needles| b.iter(|| black_box(needles).first(black_box(haystack))),
                );
            }
        }
    }
    group.finish();
}

fn bench_iterate(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterate/sherlock");
    group.throughput(Throughput::Bytes(SHERLOCK_HUGE.len() as u64));
    for &(name, set) in DENSITY_SETS {
        let finder = verified_finder(set, Backend::Auto, SHERLOCK_HUGE, name);
        group.bench_with_input(BenchmarkId::new(OURS, name), &finder, |b, finder| {
            b.iter(|| offset_sum(black_box(finder), black_box(SHERLOCK_HUGE)))
        });
        let Some(needles) = verified_needles(set, SHERLOCK_HUGE, name) else {
            continue;
        };
        group.bench_with_input(BenchmarkId::new(THEIRS, name), &needles, |b, &needles| {
            b.iter(|| black_box(needles).offset_sum(black_box(SHERLOCK_HUGE)))
        });
    }
    group.finish();
}

fn bench_kinds(c: &mut Criterion) {
    let mut group = c.benchmark_group("kind/sherlock");
    group.throughput(Throughput::Bytes(SHERLOCK_HUGE.len() as u64));
    for &(name, set) in KIND_SETS {
        for &(backend_name, backend) in BACKENDS {
            let param = format!("{name}/{backend_name}");
            let finder = verified_finder(set, backend, SHERLOCK_HUGE, &param);
            group.bench_function(&param, |b| {
                b.iter(|| black_box(&finder).iter(black_box(SHERLOCK_HUGE)).count())
            });
        }
    }
    group.finish();
}

fn bench_corpora(c: &mut Criterion) {
    let mut group = c.benchmark_group("count/corpora");
    for &(corpus, haystack) in CORPORA {
        group.throughput(Throughput::Bytes(haystack.len() as u64));
        for &(name, set) in CORPUS_SETS {
            let param = format!("{name}/{corpus}");
            let finder = verified_finder(set, Backend::Auto, haystack, &param);
            group.bench_with_input(BenchmarkId::new(OURS, &param), &finder, |b, finder| {
                b.iter(|| black_box(finder).iter(black_box(haystack)).count())
            });
            let Some(needles) = verified_needles(set, haystack, &param) else {
                continue;
            };
            group.bench_with_input(BenchmarkId::new(THEIRS, &param), &needles, |b, &needles| {
                b.iter(|| black_box(needles).count(black_box(haystack)))
            });
        }
    }
    group.finish();
}

fn bench_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("count/sizes");
    for &(name, set) in SIZE_SETS {
        for (size, haystack) in latency_haystacks() {
            for &(backend_name, backend) in BACKENDS {
                let param = format!("{name}/{size}/{backend_name}");
                let finder = verified_finder(set, backend, haystack, &param);
                group.bench_with_input(BenchmarkId::new(OURS, &param), &finder, |b, finder| {
                    b.iter(|| black_box(finder).iter(black_box(haystack)).count())
                });
            }
            let param = format!("{name}/{size}");
            let Some(needles) = verified_needles(set, haystack, &param) else {
                continue;
            };
            group.bench_with_input(BenchmarkId::new(THEIRS, &param), &needles, |b, &needles| {
                b.iter(|| black_box(needles).count(black_box(haystack)))
            });
        }
    }
    group.finish();
}

fn bench_find_first_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("find-first/sizes");
    for &(name, set) in FIRST_SETS {
        for (size, haystack) in latency_haystacks() {
            for &(backend_name, backend) in BACKENDS {
                let param = format!("{name}/{size}/{backend_name}");
                let finder = verified_finder(set, backend, haystack, &param);
                group.bench_with_input(BenchmarkId::new(OURS, &param), &finder, |b, finder| {
                    b.iter(|| black_box(finder).iter(black_box(haystack)).next())
                });
            }
            let param = format!("{name}/{size}");
            let Some(needles) = verified_needles(set, haystack, &param) else {
                continue;
            };
            group.bench_with_input(BenchmarkId::new(THEIRS, &param), &needles, |b, &needles| {
                b.iter(|| black_box(needles).first(black_box(haystack)))
            });
        }
    }
    group.finish();
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("build");
    for &(name, set) in KIND_SETS {
        group.bench_function(BenchmarkId::new(name, "from-bytes"), |b| {
            b.iter(|| black_box(set).finder(Backend::Auto))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_count,
    bench_find_first,
    bench_iterate,
    bench_kinds,
    bench_corpora,
    bench_sizes,
    bench_find_first_sizes,
    bench_build,
    bench_first_call,
);
criterion_main!(benches);
