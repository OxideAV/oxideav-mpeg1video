//! Linear-scan + small-prefix LUT VLC decoder.
//!
//! MPEG-1 VLC tables from Annex B of ISO/IEC 11172-2 are listed in the spec
//! as (codeword, symbol) pairs with codeword lengths up to ~17 bits. The
//! hot-path decoder is AC-coefficient decode (`dct_coeffs::table()`), which
//! fires ~10 times per 8×8 block × ~1500 blocks per 256×256 frame.
//!
//! For throughput we build a [`VlcTable`] that:
//!   1. caches `max_bits` so `decode()` doesn't re-scan the entries to
//!      figure out how many bits to peek;
//!   2. precomputes a 9-bit prefix LUT that resolves any codeword of
//!      length ≤ 9 bits in a single array lookup — catching the vast
//!      majority of symbols on common content.
//!
//! Codewords longer than 9 bits fall through to the original linear scan.

use oxideav_core::{Error, Result};

use oxideav_core::bits::BitReader;

/// One entry in a VLC table. `code` occupies the low `bits` bits (MSB-first).
#[derive(Clone, Copy, Debug)]
pub struct VlcEntry<T: Copy> {
    pub code: u32,
    pub bits: u8,
    pub value: T,
}

impl<T: Copy> VlcEntry<T> {
    pub const fn new(bits: u8, code: u32, value: T) -> Self {
        Self { code, bits, value }
    }
}

/// LUT entry: `bits == 0` means "this prefix is ambiguous (longer code
/// matches) — fall back to the linear scan". Otherwise, `value` is the
/// decoded symbol and `bits` is how many bits to consume from the stream.
#[derive(Clone, Copy)]
struct LutEntry<T: Copy> {
    value: Option<T>,
    bits: u8,
}

impl<T: Copy> LutEntry<T> {
    const EMPTY: Self = Self {
        value: None,
        bits: 0,
    };
}

/// Prefix width used for the LUT. 9 bits → 512-entry table. A 2-bit code
/// fills 128 slots; a 9-bit code fills 1. Anything longer marks its slots
/// as "fallback".
const LUT_BITS: u8 = 9;
const LUT_SIZE: usize = 1 << LUT_BITS;

/// A VLC table with pre-computed `max_bits` and a short-prefix LUT for fast
/// decode.
pub struct VlcTable<T: Copy + 'static> {
    pub entries: Vec<VlcEntry<T>>,
    pub max_bits: u8,
    /// Effective LUT width (`min(LUT_BITS, max_bits)`). When `max_bits`
    /// is already ≤ `LUT_BITS` the LUT covers everything and the linear
    /// fallback is only hit on malformed streams.
    lut_bits: u8,
    /// Box<[..; LUT_SIZE]> so the table isn't copied on every lookup.
    lut: Box<[LutEntry<T>; LUT_SIZE]>,
}

impl<T: Copy + 'static> VlcTable<T> {
    pub fn new(entries: Vec<VlcEntry<T>>) -> Self {
        let max_bits = entries.iter().map(|e| e.bits).max().unwrap_or(0);
        let lut_bits = max_bits.min(LUT_BITS);
        let mut lut = Box::new([LutEntry::EMPTY; LUT_SIZE]);
        if lut_bits > 0 {
            for e in entries.iter() {
                if e.bits <= lut_bits {
                    // A `b`-bit code with value `c` matches prefixes
                    // `(c << (lut_bits - b)) | s` for every `s` in
                    // `0..2^(lut_bits - b)`. Populate them all.
                    let shift = (lut_bits - e.bits) as u32;
                    let base = (e.code << shift) as usize;
                    let count = 1usize << shift;
                    for k in 0..count {
                        let idx = base | k;
                        lut[idx] = LutEntry {
                            value: Some(e.value),
                            bits: e.bits,
                        };
                    }
                }
                // Longer codes implicitly leave their slots as EMPTY ⇒
                // the decoder falls back to the linear scan.
            }
        }
        Self {
            entries,
            max_bits,
            lut_bits,
            lut,
        }
    }

    pub fn from_slice(entries: &[VlcEntry<T>]) -> Self {
        Self::new(entries.to_vec())
    }
}

/// Decode one symbol from the table.
#[inline]
pub fn decode<T: Copy>(br: &mut BitReader<'_>, table: &VlcTable<T>) -> Result<T> {
    let max_bits = table.max_bits as u32;
    if max_bits == 0 {
        return Err(Error::invalid("vlc: empty table"));
    }
    let remaining = br.bits_remaining() as u32;
    let lut_bits = table.lut_bits as u32;

    // Fast path: peek `lut_bits` and look up. Only viable when we have
    // enough bits for a full LUT lookup.
    if lut_bits > 0 && remaining >= lut_bits {
        let prefix = br.peek_u32(lut_bits)? as usize;
        let entry = &table.lut[prefix];
        if let Some(v) = entry.value {
            br.consume(entry.bits as u32)?;
            return Ok(v);
        }
        // `bits == 0` ⇒ ambiguous, fall through to linear scan.
    }

    // Slow / fallback path: linear scan.
    let peek_bits = max_bits.min(remaining);
    if peek_bits == 0 {
        return Err(Error::invalid("vlc: no bits available"));
    }
    let peeked = br.peek_u32(peek_bits)?;
    let peeked_full = peeked << (max_bits - peek_bits);
    for e in table.entries.iter() {
        if (e.bits as u32) > peek_bits {
            continue;
        }
        let shift = max_bits - e.bits as u32;
        let prefix = peeked_full >> shift;
        if prefix == e.code {
            br.consume(e.bits as u32)?;
            return Ok(e.value);
        }
    }
    Err(Error::invalid("vlc: no matching codeword"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_dc_luma_size() {
        let table = crate::tables::dct_dc::luma();
        let v: u16 = 0b1000_0011_0111_1100;
        let data = [(v >> 8) as u8, (v & 0xff) as u8];
        let mut br = BitReader::new(&data);

        assert_eq!(decode(&mut br, table).unwrap(), 0);
        assert_eq!(decode(&mut br, table).unwrap(), 1);
        assert_eq!(decode(&mut br, table).unwrap(), 2);
        assert_eq!(decode(&mut br, table).unwrap(), 3);
        assert_eq!(decode(&mut br, table).unwrap(), 6);
    }

    #[test]
    fn lut_resolves_short_codes() {
        // With dct_coeffs::table() the LUT should catch the 2-bit EOB
        // (code 0b10 → 0x2) and the 3-bit (run=0, level=1) (code 0b110 → 0x3).
        let table = crate::tables::dct_coeffs::table();
        assert_eq!(table.lut_bits, LUT_BITS);
        // EOB (2-bit code `10`): populates LUT prefixes 0b10_xxxxxxx →
        // indices 0x100..0x200 in a 9-bit LUT.
        let eob_slot = &table.lut[0x100];
        assert!(eob_slot.value.is_some());
        assert_eq!(eob_slot.bits, 2);
    }

    #[test]
    fn lut_fallback_on_long_codes() {
        // dct_coeffs has codes up to 17 bits. A 12-bit-or-longer code
        // occupies a prefix that must be marked as "fallback" (bits=0).
        // Pick a known escape-prefixed run=31, level=-2 code path that
        // exceeds 9 bits.
        let table = crate::tables::dct_coeffs::table();
        // At least SOME LUT slot must be "empty" (fallback) since not
        // every 9-bit prefix is a terminal symbol.
        let empty_count = table.lut.iter().filter(|e| e.value.is_none()).count();
        assert!(empty_count > 0, "expected some fallback slots in LUT");
    }
}
