//! The format must agree bit for bit with the Python description of it
//! (`astracode/spec.py` and `astracode/astra.py`). `data/format_vectors.txt`
//! was written by them.

use astracode::astra::Number;
use astracode::format::{code, mix, ring0, Payload};

fn numbers<T: std::str::FromStr>(words: &[&str]) -> Vec<T> {
    words.iter().map(|w| w.parse().ok().expect("a number")).collect()
}

/// A payload from its length in bits and its bits as hex, the last byte padded with zeros.
fn payload(bits: &str, hex: &str) -> Payload {
    let hex = if hex == "-" { "" } else { hex };
    let bytes: Vec<u8> = (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect();
    let all = Payload::from_bytes(bytes);
    Payload::from_bits(all.bits().take(bits.parse().unwrap()))
}

#[test]
fn matches_the_python_reference() {
    let text = include_str!("data/format_vectors.txt");
    let mut checked = 0;
    for line in text.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words[0] {
            "ring0" => {
                let value: u64 = words[1].parse().unwrap();
                let symbols: ring0::Symbols = numbers::<u8>(&words[2..]).try_into().unwrap();
                let header = ring0::read(&symbols, None).expect(line);
                assert_eq!((header.value, header.repaired), (value, 0), "{line}");
            }
            "mix" => {
                let (value, mixed): (u64, u64) = (words[1].parse().unwrap(), words[2].parse().unwrap());
                assert_eq!(mix::mix(value), mixed);
                assert_eq!(mix::unmix(mixed), value);
            }
            "astra" => {
                let number = Number::new(words[1].parse().unwrap(), words[2].parse().unwrap()).unwrap();
                assert_eq!(number.member_id(), words[3]);
                assert_eq!(number.to_payload(), payload("38", words[4]));
                let symbols = code::encode(&number.to_payload(), 1).unwrap();
                assert_eq!(symbols, vec![numbers::<u8>(&words[5..])], "{line}");
                assert_eq!(Number::from_payload(&code::decode(&symbols).unwrap().0), Some(number));
            }
            "code" => {
                // level, rings asked for (0 = the fewest), rings, bits, payload, then every star.
                let (level, asked, rings): (u8, usize, usize) =
                    (words[1].parse().unwrap(), words[2].parse().unwrap(), words[3].parse().unwrap());
                let payload = payload(words[4], words[5]);
                let symbols =
                    if asked == 0 { code::encode(&payload, level) } else { code::encode_with_rings(&payload, level, asked) }.unwrap();
                assert_eq!(symbols.len(), rings, "code of {} bits at level {level}", words[4]);
                let flat: Vec<u8> = symbols.iter().flatten().copied().collect();
                assert_eq!(flat, numbers::<u8>(&words[6..]), "code of {} bits, {rings} rings, level {level}", words[4]);
                assert_eq!(code::decode(&symbols), Some((payload, 0)));
            }
            "cap" => {
                let rings: usize = words[1].parse().unwrap();
                for (level, entry) in words[2..].iter().enumerate() {
                    let parts: Vec<usize> = entry.split(':').map(|p| p.parse().unwrap()).collect();
                    assert_eq!(code::check_symbols(rings, level as u8), parts[0]);
                    assert_eq!(code::capacity_bits(rings, level as u8), parts[1]);
                }
            }
            other => panic!("unknown vector kind {other}"),
        }
        checked += 1;
    }
    assert!(checked > 60);
}
