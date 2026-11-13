//! Astra's own use of Astracode: a member number in a one-ring code.
//!
//! The format ([`crate::format`]) carries a string of bits and gives it no
//! meaning. This module is one convention on top of it, the one Astra's apps
//! follow. Nothing in the scanner or the format knows about it.
//!
//! * A payload of exactly [`NUMBER_BITS`] = 38 bits is an *Astra number*. That
//!   is the most one ring holds, so an Astra number is always a one-ring code;
//!   and since 38 bits is not a whole number of bytes, it cannot be mistaken
//!   for a byte string.
//! * Read as an integer, the 38 bits are `metadata * 10^10 + payload`: a
//!   payload of up to ten digits and a metadata value `0..=26` saying what the
//!   payload is.
//! * A *member ID* is 12 digits: the ten-digit payload followed by the payload
//!   modulo 97. Only the ten are stored; the last two are recomputed when
//!   reading.

use crate::format::Payload;

/// Length of an Astra number's payload, in bits: everything one ring holds.
pub const NUMBER_BITS: usize = 38;
/// Payloads are `0 .. PAYLOAD_LIMIT`: ten decimal digits.
pub const PAYLOAD_LIMIT: u64 = 10_000_000_000;
/// Metadata is `0 .. METADATA_LIMIT`, which is 27.
pub const METADATA_LIMIT: u64 = (1 << NUMBER_BITS) / PAYLOAD_LIMIT;

/// Why something is not an Astra number.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NumberError {
    #[error("metadata must be 0..{}", METADATA_LIMIT - 1)]
    Metadata,
    #[error("payload must have at most 10 digits")]
    Payload,
    #[error("not a valid member ID")]
    MemberId,
}

/// An Astra number: what a one-ring Astra code holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Number {
    /// What the payload is, `0..=26`.
    pub metadata: u8,
    /// Up to ten digits.
    pub payload: u64,
}

impl Number {
    /// A number from its parts, if they are in range.
    pub fn new(metadata: u8, payload: u64) -> Result<Number, NumberError> {
        if metadata as u64 >= METADATA_LIMIT {
            return Err(NumberError::Metadata);
        }
        if payload >= PAYLOAD_LIMIT {
            return Err(NumberError::Payload);
        }
        Ok(Number { metadata, payload })
    }

    /// The number a 12-digit member ID stands for, with the given metadata.
    /// Anything but digits is ignored; fails if there are not twelve digits or
    /// the last two do not check out.
    pub fn from_member_id(metadata: u8, text: &str) -> Result<Number, NumberError> {
        let digits: String = text.chars().filter(char::is_ascii_digit).collect();
        if digits.len() != 12 {
            return Err(NumberError::MemberId);
        }
        let payload: u64 = digits[..10].parse().map_err(|_| NumberError::MemberId)?;
        let check: u64 = digits[10..].parse().map_err(|_| NumberError::MemberId)?;
        if payload % 97 != check {
            return Err(NumberError::MemberId);
        }
        Number::new(metadata, payload)
    }

    /// The 12-digit member ID: the payload, then the payload modulo 97.
    pub fn member_id(&self) -> String {
        format!("{:010}{:02}", self.payload, self.payload % 97)
    }

    /// The payload of the code for this number: 38 bits, which fill one ring.
    pub fn to_payload(&self) -> Payload {
        Payload::from_number(self.metadata as u64 * PAYLOAD_LIMIT + self.payload, NUMBER_BITS).expect("an Astra number fits in 38 bits")
    }

    /// Read a code's payload as an Astra number. `None` if it is not one:
    /// any other length, or a value past the last metadata.
    pub fn from_payload(payload: &Payload) -> Option<Number> {
        if payload.len() != NUMBER_BITS {
            return None;
        }
        let value = payload.as_number()?;
        Number::new(u8::try_from(value / PAYLOAD_LIMIT).ok()?, value % PAYLOAD_LIMIT).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::code;

    #[test]
    fn member_ids_round_trip() {
        let number = Number::new(0, 4_320_880_580).unwrap();
        assert_eq!(number.member_id(), "432088058060");
        assert_eq!(Number::from_member_id(0, "4320 8805 8060"), Ok(number));
        assert_eq!(Number::from_member_id(0, "432088058061"), Err(NumberError::MemberId));
        assert_eq!(Number::new(27, 0), Err(NumberError::Metadata));
        assert_eq!(Number::new(0, PAYLOAD_LIMIT), Err(NumberError::Payload));
    }

    #[test]
    fn an_astra_number_is_always_one_ring_and_recognisable() {
        assert_eq!(code::capacity_bits(1, 0), NUMBER_BITS);
        for number in [Number::new(0, 0).unwrap(), Number::new(5, 4_320_880_580).unwrap(), Number::new(26, 9_999_999_999).unwrap()] {
            let payload = number.to_payload();
            for level in 0..4 {
                let rings = code::encode(&payload, level).unwrap();
                assert_eq!(rings.len(), 1);
                assert_eq!(Number::from_payload(&code::decode(&rings).unwrap().0), Some(number));
            }
        }
    }

    #[test]
    fn other_payloads_are_not_astra_numbers() {
        // Whole bytes never are, whatever they hold.
        for len in 0..8 {
            assert_eq!(Number::from_payload(&Payload::from_bytes(vec![0u8; len])), None);
        }
        assert_eq!(Number::from_payload(&Payload::from_number(1, 37).unwrap()), None);
        // 38 bits past the last metadata value.
        assert_eq!(Number::from_payload(&Payload::from_number(27 * PAYLOAD_LIMIT, 38).unwrap()), None);
    }
}
