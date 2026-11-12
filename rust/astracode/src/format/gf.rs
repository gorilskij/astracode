//! Arithmetic in the finite fields GF(2^m).
//!
//! A star carries five bits, so ring 0 works in GF(32) with one star per
//! symbol; the outer rings take stars in pairs and work in GF(1024). Both are
//! represented the usual way: every non-zero element is a power of a
//! primitive element `a`, and multiplication adds exponents.

use std::sync::OnceLock;

/// A field GF(2^m) with log and antilog tables.
pub struct Field {
    /// `exp[i] = a^i`, stored twice over so that a sum of two logs needs no reduction.
    exp: Vec<u16>,
    /// `log[x]` for non-zero `x`; `log[0]` is unused.
    log: Vec<u16>,
    /// Number of non-zero elements, `2^m - 1`.
    order: usize,
}

impl Field {
    /// Build GF(2^bits) from a primitive polynomial given with its leading term,
    /// for example `0x25` for `x^5 + x^2 + 1`.
    fn new(bits: u32, poly: u32) -> Field {
        let order = (1usize << bits) - 1;
        let mut exp = vec![0u16; 2 * order];
        let mut log = vec![0u16; order + 1];
        let mut x = 1u32;
        for i in 0..order {
            exp[i] = x as u16;
            exp[i + order] = x as u16;
            log[x as usize] = i as u16;
            x <<= 1;
            if x >> bits != 0 {
                x ^= poly;
            }
        }
        Field { exp, log, order }
    }

    /// Number of non-zero elements, which is also the longest codeword the field supports.
    pub fn order(&self) -> usize {
        self.order
    }

    #[inline]
    pub fn mul(&self, a: u16, b: u16) -> u16 {
        if a == 0 || b == 0 {
            0
        } else {
            self.exp[self.log[a as usize] as usize + self.log[b as usize] as usize]
        }
    }

    /// `a / b`. `b` must not be zero.
    #[inline]
    pub fn div(&self, a: u16, b: u16) -> u16 {
        debug_assert!(b != 0, "division by zero in GF(2^m)");
        if a == 0 {
            0
        } else {
            self.exp[self.log[a as usize] as usize + self.order - self.log[b as usize] as usize]
        }
    }

    /// `a^e` for the primitive element `a`, for any non-negative exponent.
    #[inline]
    pub fn alpha(&self, e: usize) -> u16 {
        self.exp[e % self.order]
    }

    /// The exponent `e` with `a^e = x`. `x` must not be zero.
    #[inline]
    pub fn log(&self, x: u16) -> usize {
        debug_assert!(x != 0, "log of zero in GF(2^m)");
        self.log[x as usize] as usize
    }
}

/// GF(32), `x^5 + x^2 + 1`: one star per symbol. Used by ring 0.
pub fn gf32() -> &'static Field {
    static FIELD: OnceLock<Field> = OnceLock::new();
    FIELD.get_or_init(|| Field::new(5, 0x25))
}

/// GF(1024), `x^10 + x^3 + 1`: two stars per symbol. Used by the outer rings.
pub fn gf1024() -> &'static Field {
    static FIELD: OnceLock<Field> = OnceLock::new();
    FIELD.get_or_init(|| Field::new(10, 0x409))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_element_is_a_power_of_the_generator() {
        for field in [gf32(), gf1024()] {
            let mut seen = vec![false; field.order() + 1];
            for i in 0..field.order() {
                let x = field.alpha(i) as usize;
                assert!(x != 0 && !seen[x]);
                seen[x] = true;
            }
        }
    }

    #[test]
    fn division_undoes_multiplication() {
        let f = gf1024();
        for a in [1u16, 2, 37, 500, 1023] {
            for b in [1u16, 3, 64, 999] {
                assert_eq!(f.div(f.mul(a, b), b), a);
            }
        }
        assert_eq!(f.mul(0, 5), 0);
        assert_eq!(f.div(0, 5), 0);
    }
}
