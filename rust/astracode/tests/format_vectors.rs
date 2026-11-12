//! The format must agree bit for bit with the Python reference
//! (`astracode/spec.py`). `data/format_vectors.txt` was written by it.

use astracode::format::{message, mix, ring0};

fn numbers<T: std::str::FromStr>(words: &[&str]) -> Vec<T> {
    words.iter().map(|w| w.parse().ok().expect("a number")).collect()
}

#[test]
fn matches_the_python_reference() {
    let text = include_str!("data/format_vectors.txt");
    let mut checked = 0;
    for line in text.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words[0] {
            "ring" => {
                let (metadata, payload, value): (u64, u64, u64) =
                    (words[1].parse().unwrap(), words[2].parse().unwrap(), words[3].parse().unwrap());
                assert_eq!(ring0::pack(metadata, payload), Ok(value));
                assert_eq!(ring0::unpack(value), (metadata, payload));
                assert_eq!(ring0::member_id(payload), words[4]);
                let symbols = ring0::encode(value).unwrap();
                assert_eq!(symbols.to_vec(), numbers::<u8>(&words[5..]), "{line}");
                assert_eq!(ring0::decode(&symbols, None), Some((value, 0)));
            }
            "mix" => {
                let (value, mixed): (u64, u64) = (words[1].parse().unwrap(), words[2].parse().unwrap());
                assert_eq!(mix::mix(value), mixed);
                assert_eq!(mix::unmix(mixed), value);
            }
            "msg" => {
                let (rings, level): (usize, u8) = (words[1].parse().unwrap(), words[2].parse().unwrap());
                let hex = if words[3] == "-" { "" } else { words[3] };
                let bytes: Vec<u8> = (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect();
                let symbols = message::encode(&bytes, rings, level).unwrap();
                let flat: Vec<u8> = symbols.iter().flatten().copied().collect();
                assert_eq!(flat, numbers::<u8>(&words[4..]), "message code, {rings} rings, level {level}");
                assert_eq!(message::decode(&symbols), Some((bytes, 0)));
            }
            "cap" => {
                let rings: usize = words[1].parse().unwrap();
                for (level, entry) in words[2..].iter().enumerate() {
                    let parts: Vec<usize> = entry.split(':').map(|p| p.parse().unwrap()).collect();
                    assert_eq!(message::check_symbols(rings, level as u8), parts[0]);
                    assert_eq!(message::capacity(rings, level as u8), parts[2]);
                }
            }
            other => panic!("unknown vector kind {other}"),
        }
        checked += 1;
    }
    assert!(checked > 40);
}
