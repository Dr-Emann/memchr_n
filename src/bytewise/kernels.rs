use crate::KernelData;
use crate::bitset::Bitset;
use crate::bytewise::Kernel;

/// Probes a 256-bit membership table, one byte per probe.
///
/// The counterpart of the vector `AnyByte` kernel for a target with no shuffle to run that
/// kernel's gather on.
#[derive(Copy, Clone)]
pub(crate) struct AnyByte {
    bitset: Bitset,
}

impl Kernel for AnyByte {
    unsafe fn from_data(data: &KernelData) -> Self {
        // SAFETY: the caller guarantees `bitset` is live.
        Self {
            bitset: unsafe { data.bitset },
        }
    }

    #[inline]
    fn matches(&self, byte: u8) -> bool {
        self.bitset.contains(byte)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_byte_accepts_exactly_the_set() {
        let sets: [Vec<u8>; 4] = [
            Vec::new(),
            vec![0x41],
            (0..=u8::MAX).step_by(3).collect(),
            (0x80..=u8::MAX).collect(),
        ];
        for set in sets {
            let bitset = Bitset::from_bytes(&set);
            let kernel = AnyByte { bitset };
            for byte in 0..=u8::MAX {
                assert_eq!(
                    kernel.matches(byte),
                    set.contains(&byte),
                    "{byte} in {set:?}"
                );
            }
        }
    }
}
