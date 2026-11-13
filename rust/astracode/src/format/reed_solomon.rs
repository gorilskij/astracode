//! Reed-Solomon coding over the fields of [`super::gf`].
//!
//! Codewords are systematic: the data symbols come first and the check symbols
//! follow. A codeword `c[0..n]` is read as the polynomial
//! `c[0] x^(n-1) + ... + c[n-1]`, and is valid when that polynomial has the
//! roots `a^0, a^1, ..., a^(checks-1)`.
//!
//! Two decoders live here because the two parts of a code need different
//! things:
//!
//! * Ring 0 has only four check symbols and must never be "repaired" into a
//!   different code, so it gets the cautious pair [`correct_one`] and
//!   [`fill_two`].
//! * The outer rings form one long codeword and get a general
//!   Berlekamp-Massey decoder, [`correct_errors`], with an explicit limit on
//!   how many symbols it may repair, and [`correct_errata`], the same for a
//!   reader that can say which symbols it is unsure of.

use super::gf::Field;

/// Check symbols for `data`: the remainder of `data(x) * x^checks` divided by
/// the generator polynomial.
pub fn parity(field: &Field, data: &[u16], checks: usize) -> Vec<u16> {
    // Generator polynomial with roots a^0 .. a^(checks-1), highest degree first.
    let mut gen = vec![1u16];
    for j in 0..checks {
        let root = field.alpha(j);
        let mut next = vec![0u16; gen.len() + 1];
        for (i, &g) in gen.iter().enumerate() {
            next[i] ^= g;
            next[i + 1] ^= field.mul(g, root);
        }
        gen = next;
    }
    // Long division, one data symbol at a time.
    let mut rem = vec![0u16; checks];
    for &d in data {
        let factor = d ^ rem[0];
        rem.rotate_left(1);
        rem[checks - 1] = 0;
        if factor != 0 {
            for (r, &g) in rem.iter_mut().zip(&gen[1..]) {
                *r ^= field.mul(factor, g);
            }
        }
    }
    rem
}

/// The syndromes `S_j = c(a^j)` for `j` in `0..checks`. All zero for a valid codeword.
pub fn syndromes(field: &Field, codeword: &[u16], checks: usize) -> Vec<u16> {
    let n = codeword.len();
    (0..checks)
        .map(|j| {
            let mut s = 0u16;
            for (i, &c) in codeword.iter().enumerate() {
                if c != 0 {
                    s ^= field.alpha(field.log(c) + j * (n - 1 - i));
                }
            }
            s
        })
        .collect()
}

/// Repair at most one wrong symbol of a codeword with four check symbols.
///
/// Returns how many symbols were repaired (0 or 1), or `None` if the damage is
/// more than that. Four check symbols could repair two symbols, but that would
/// leave nothing to notice a third; repairing one keeps the code able to detect
/// up to three.
pub fn correct_one(field: &Field, codeword: &mut [u16]) -> Option<usize> {
    let s = syndromes(field, codeword, 4);
    if s.iter().all(|&x| x == 0) {
        return Some(0);
    }
    if s.contains(&0) {
        return None;
    }
    // A single error e at position p gives S_j = e * X^j with X = a^p.
    let x = field.div(s[1], s[0]);
    if field.mul(s[1], x) != s[2] || field.mul(s[2], x) != s[3] {
        return None;
    }
    let pos = field.log(x);
    if pos >= codeword.len() {
        return None;
    }
    let at = codeword.len() - 1 - pos;
    codeword[at] ^= s[0];
    Some(1)
}

/// Treat symbols `i1` and `i2` of a codeword with four check symbols as unknown
/// and solve for them. Returns false, leaving the codeword untouched, unless
/// the result satisfies the remaining two checks.
///
/// Two check symbols rebuild the two unknowns and the other two verify, so this
/// is only as trustworthy as the caller's reason for doubting exactly these two.
pub fn fill_two(field: &Field, codeword: &mut [u16], i1: usize, i2: usize) -> bool {
    let n = codeword.len();
    let s = syndromes(field, codeword, 4);
    let x1 = field.alpha(n - 1 - i1);
    let x2 = field.alpha(n - 1 - i2);
    let e2 = field.div(s[1] ^ field.mul(s[0], x1), x1 ^ x2);
    let e1 = s[0] ^ e2;
    let x1s = field.mul(x1, x1);
    let x2s = field.mul(x2, x2);
    if field.mul(e1, x1s) ^ field.mul(e2, x2s) != s[2] {
        return false;
    }
    if field.mul(e1, field.mul(x1s, x1)) ^ field.mul(e2, field.mul(x2s, x2)) != s[3] {
        return false;
    }
    codeword[i1] ^= e1;
    codeword[i2] ^= e2;
    true
}

/// Evaluate a polynomial given lowest degree first.
fn eval(field: &Field, poly: &[u16], x: u16) -> u16 {
    let mut value = 0u16;
    let mut power = 1u16;
    for &c in poly {
        value ^= field.mul(c, power);
        power = field.mul(power, x);
    }
    value
}

/// Berlekamp-Massey: the shortest recurrence that generates `sequence`, as a
/// polynomial lowest degree first, with its length.
fn shortest_recurrence(field: &Field, sequence: &[u16]) -> (Vec<u16>, usize) {
    let mut loc = vec![1u16];
    let mut prev = vec![1u16];
    let mut shift = 1usize;
    let mut last = 1u16;
    let mut errs = 0usize;
    for i in 0..sequence.len() {
        let mut delta = sequence[i];
        for j in 1..=errs {
            if j < loc.len() {
                delta ^= field.mul(loc[j], sequence[i - j]);
            }
        }
        if delta == 0 {
            shift += 1;
            continue;
        }
        let coef = field.div(delta, last);
        let mut next = loc.clone();
        if next.len() < prev.len() + shift {
            next.resize(prev.len() + shift, 0);
        }
        for (j, &c) in prev.iter().enumerate() {
            next[j + shift] ^= field.mul(coef, c);
        }
        if 2 * errs <= i {
            prev = loc;
            last = delta;
            errs = i + 1 - errs;
            shift = 1;
        } else {
            shift += 1;
        }
        loc = next;
    }
    while loc.last() == Some(&0) {
        loc.pop();
    }
    (loc, errs)
}

/// Repair up to `max_errors` wrong symbols anywhere in the codeword
/// (Berlekamp-Massey, Chien search, Forney).
///
/// `max_errors` may be at most `checks / 2`. Asking for fewer leaves check
/// symbols over for detection, which is what keeps a badly misread codeword
/// from being turned into a different valid one. Returns the repaired codeword
/// and the number of symbols changed, or `None` if it cannot be repaired
/// within the limit.
pub fn correct_errors(field: &Field, codeword: &[u16], checks: usize, max_errors: usize) -> Option<(Vec<u16>, usize)> {
    correct_errata(field, codeword, checks, &[], max_errors)
}

/// As [`correct_errors`], for a reader that doubts some symbols to begin with:
/// the symbols at the indices `erased` count as unknown, and up to
/// `max_errors` of the others may be wrong.
///
/// An unknown symbol costs one check symbol to rebuild and a wrong one costs
/// two, so `erased.len() + 2 * max_errors` may be at most `checks`; whatever
/// is left over is what detects a codeword too damaged to repair. Returns the
/// repaired codeword and the number of symbols changed.
pub fn correct_errata(field: &Field, codeword: &[u16], checks: usize, erased: &[usize], max_errors: usize) -> Option<(Vec<u16>, usize)> {
    let n = codeword.len();
    let synd = syndromes(field, codeword, checks);
    if synd.iter().all(|&x| x == 0) {
        return Some((codeword.to_vec(), 0));
    }
    let unknown = erased.len();
    if unknown > checks || n > field.order() || erased.iter().any(|&i| i >= n) {
        return None;
    }
    let max_errors = max_errors.min((checks - unknown) / 2);

    // The locator of the unknown symbols: the product of (1 + X x) over their places X.
    let mut known_places = vec![1u16];
    for &index in erased {
        let place = field.alpha(n - 1 - index);
        known_places.push(0);
        for j in (1..known_places.len()).rev() {
            known_places[j] ^= field.mul(known_places[j - 1], place);
        }
    }
    // With the unknown symbols' part taken out of the syndromes, what is left obeys the
    // recurrence of the wrong symbols alone.
    let rest: Vec<u16> =
        (unknown..checks).map(|i| known_places.iter().enumerate().fold(0, |sum, (j, &g)| sum ^ field.mul(g, synd[i - j]))).collect();
    let (wrong_places, errs) = shortest_recurrence(field, &rest);
    if wrong_places.len() != errs + 1 || errs > max_errors {
        return None;
    }
    // Both kinds together.
    let mut loc = vec![0u16; known_places.len() + wrong_places.len() - 1];
    for (i, &a) in known_places.iter().enumerate() {
        for (j, &b) in wrong_places.iter().enumerate() {
            loc[i + j] ^= field.mul(a, b);
        }
    }

    // Evaluator: syndromes(x) * loc(x) mod x^checks.
    let mut omega = vec![0u16; checks];
    for (i, &a) in loc.iter().enumerate().take(checks) {
        for j in 0..checks - i {
            omega[i + j] ^= field.mul(a, synd[j]);
        }
    }
    // The formal derivative of loc keeps its odd terms: loc'(x) = sum loc[2k+1] x^(2k).
    let derivative: Vec<u16> = loc.iter().skip(1).step_by(2).copied().collect();

    let mut fixed = codeword.to_vec();
    let mut found = 0usize;
    for pos in 0..n {
        let x_inv = field.alpha(field.order() - pos % field.order());
        if eval(field, &loc, x_inv) != 0 {
            continue;
        }
        let num = eval(field, &omega, x_inv);
        let den = eval(field, &derivative, field.mul(x_inv, x_inv));
        if den == 0 {
            return None;
        }
        fixed[n - 1 - pos] ^= field.mul(field.alpha(pos), field.div(num, den));
        found += 1;
    }
    // Fewer roots than the locator's degree means the errors are not where a
    // codeword of this length could have them (or a symbol was named unknown twice).
    if found != errs + unknown || syndromes(field, &fixed, checks).iter().any(|&x| x != 0) {
        return None;
    }
    let changed = fixed.iter().zip(codeword).filter(|(a, b)| a != b).count();
    Some((fixed, changed))
}

#[cfg(test)]
mod tests {
    use super::super::gf::{gf1024, gf32};
    use super::*;

    fn codeword(field: &Field, data: &[u16], checks: usize) -> Vec<u16> {
        let mut cw = data.to_vec();
        cw.extend(parity(field, data, checks));
        cw
    }

    #[test]
    fn a_codeword_has_zero_syndromes() {
        let cw = codeword(gf32(), &[1, 2, 3, 4, 5, 6, 7, 31], 4);
        assert!(syndromes(gf32(), &cw, 4).iter().all(|&s| s == 0));
    }

    #[test]
    fn one_wrong_symbol_is_repaired_and_two_are_refused() {
        let f = gf32();
        let good = codeword(f, &[9, 0, 17, 4, 30, 6, 7, 12], 4);
        for at in 0..12 {
            let mut cw = good.clone();
            cw[at] ^= 21;
            assert_eq!(correct_one(f, &mut cw), Some(1));
            assert_eq!(cw, good);
        }
        let mut cw = good.clone();
        cw[2] ^= 5;
        cw[9] ^= 11;
        assert_eq!(correct_one(f, &mut cw), None);
        assert!(fill_two(f, &mut cw, 2, 9));
        assert_eq!(cw, good);
    }

    #[test]
    fn the_general_decoder_repairs_up_to_its_limit() {
        let f = gf1024();
        let data: Vec<u16> = (0..40).map(|i| (i * 37 + 5) % 1024).collect();
        let good = codeword(f, &data, 10);
        let mut cw = good.clone();
        for (k, at) in [3usize, 11, 20, 33, 47].into_iter().enumerate() {
            cw[at] ^= 100 + k as u16;
        }
        assert_eq!(correct_errors(f, &cw, 10, 5), Some((good.clone(), 5)));
        // With a lower limit the same damage is refused rather than repaired.
        assert_eq!(correct_errors(f, &cw, 10, 4), None);
        cw[49] ^= 1;
        assert_ne!(correct_errors(f, &cw, 10, 5).map(|r| r.0), Some(good));
    }

    #[test]
    fn symbols_known_to_be_doubtful_cost_half_as_much() {
        let f = gf1024();
        let data: Vec<u16> = (0..40).map(|i| (i * 53 + 9) % 1024).collect();
        let good = codeword(f, &data, 10);
        // Eight wrong symbols are beyond ten check symbols, unless the reader knows which.
        let wrong = [0usize, 5, 12, 19, 26, 33, 41, 49];
        let mut cw = good.clone();
        for (k, &at) in wrong.iter().enumerate() {
            cw[at] ^= 300 + k as u16;
        }
        assert_ne!(correct_errors(f, &cw, 10, 5).map(|r| r.0), Some(good.clone()));
        assert_eq!(correct_errata(f, &cw, 10, &wrong, 0), Some((good.clone(), 8)));
        // Doubting a symbol that was right costs a check symbol and changes nothing:
        // six doubted, of which four wrong, and two more wrong elsewhere.
        let mut cw = good.clone();
        for at in [5usize, 12, 19, 26, 44, 47] {
            cw[at] ^= 77;
        }
        assert_eq!(correct_errata(f, &cw, 10, &[5, 12, 19, 26, 30, 31], 2), Some((good.clone(), 6)));
        // One more wrong symbol than that is refused, as is a limit lower than the damage.
        assert_eq!(correct_errata(f, &cw, 10, &[5, 12, 19, 26, 30, 31], 1), None);
        cw[1] ^= 3;
        assert_eq!(correct_errata(f, &cw, 10, &[5, 12, 19, 26, 30, 31], 2), None);
        // Nothing doubted is the plain decoder.
        let mut cw = good.clone();
        cw[7] ^= 1;
        assert_eq!(correct_errata(f, &cw, 10, &[], 5), Some((good, 1)));
    }
}
