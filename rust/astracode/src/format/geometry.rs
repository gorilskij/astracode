//! Where the stars are and what shape they have, in model units.

use super::{ring_radius, ring_stars, STAR_RI, STAR_RO};

/// A point or offset in model units, `[x, y]` with y pointing down.
pub type Point = [f64; 2];

/// A point at `radius` from the origin in direction `degrees`, measured
/// clockwise on screen from the positive x axis.
fn polar(radius: f64, degrees: f64) -> Point {
    let a = degrees.to_radians();
    [radius * a.cos(), radius * a.sin()]
}

/// Centre of star `i` of ring `ring`. Star 0 is at 12 o'clock, then clockwise.
pub fn star_center(i: usize, ring: usize) -> Point {
    polar(ring_radius(ring), -90.0 + 360.0 * i as f64 / ring_stars(ring) as f64)
}

/// The centres of all stars of a ring, in order.
pub fn ring_centers(ring: usize) -> Vec<Point> {
    (0..ring_stars(ring)).map(|i| star_center(i, ring)).collect()
}

/// Whether limb `k` of a star with this symbol is solid. `k = 0` is the top
/// limb, 1 to 4 go clockwise.
pub fn limb_solid(symbol: u8, k: usize) -> bool {
    (symbol >> (4 - k)) & 1 != 0
}

/// Limb `k` of a star centred on `center`: a quadrilateral closed through the
/// star's centre, so that a light limb drawn first tucks under the solid core
/// without a seam.
pub fn limb_polygon(k: usize, center: Point) -> [Point; 4] {
    let at = |radius: f64, degrees: f64| {
        let p = polar(radius, degrees);
        [center[0] + p[0], center[1] + p[1]]
    };
    let axis = -90.0 + 72.0 * k as f64;
    [at(STAR_RI, axis - 36.0), at(STAR_RO, axis), at(STAR_RI, axis + 36.0), center]
}

/// Outline of the solid part of a star with this symbol: the centre pentagon
/// plus its solid limbs. Symbol 31 gives the complete pentagram.
pub fn star_polygon(symbol: u8, center: Point) -> Vec<Point> {
    let mut points = Vec::with_capacity(10);
    for k in 0..5 {
        let notch = polar(STAR_RI, -90.0 - 36.0 + 72.0 * k as f64);
        points.push([center[0] + notch[0], center[1] + notch[1]]);
        if limb_solid(symbol, k) {
            let tip = polar(STAR_RO, -90.0 + 72.0 * k as f64);
            points.push([center[0] + tip[0], center[1] + tip[1]]);
        }
    }
    points
}

/// Whether `p` is inside (or on the edge of) a polygon given by its corners in order.
pub fn inside(polygon: &[Point], p: Point) -> bool {
    let mut winding = false;
    let n = polygon.len();
    for i in 0..n {
        let (a, b) = (polygon[i], polygon[(i + 1) % n]);
        // On the edge counts as inside.
        let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        let within = (p[0] - a[0]) * (p[0] - b[0]) <= 0.0 && (p[1] - a[1]) * (p[1] - b[1]) <= 0.0;
        if cross.abs() < 1e-9 && within {
            return true;
        }
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]) {
            winding = !winding;
        }
    }
    winding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn star_zero_is_at_twelve_o_clock() {
        let top = star_center(0, 0);
        assert!(top[0].abs() < 1e-9 && (top[1] + 100.0).abs() < 1e-9);
        let right = star_center(3, 0);
        assert!((right[0] - 100.0).abs() < 1e-9 && right[1].abs() < 1e-9);
        assert_eq!(ring_centers(2).len(), 24);
        assert!((star_center(0, 2)[1] + 180.0).abs() < 1e-9);
    }

    #[test]
    fn the_pentagram_contains_its_centre_and_tips_but_not_its_notches() {
        let star = star_polygon(31, [0.0, 0.0]);
        assert_eq!(star.len(), 10);
        assert!(inside(&star, [0.0, 0.0]));
        assert!(inside(&star, [0.0, -19.0]));
        assert!(!inside(&star, [0.0, 15.0])); // between the two bottom limbs
        assert_eq!(star_polygon(0, [0.0, 0.0]).len(), 5);
    }
}
