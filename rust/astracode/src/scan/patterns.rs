//! Where the scanner looks: fixed sets of sample points in model units, and
//! the star outline template built from them.
//!
//! Everything here is constant and built once, on first use.

use std::sync::OnceLock;

use crate::format::geometry::{inside, ring_centers, star_polygon};
use crate::format::{CENTER_R, MARGIN, MAX_RINGS, R, STAR_RI, STAR_RO};
use crate::image::Gray;
use crate::math::{add, offset, Homography, LocalMap, Point};

/// The thinned outline keeps every this-many-th point.
pub(crate) const THIN: usize = 3;

/// Number of sample points on one limb, and in one gap between limbs.
pub(crate) const LIMB_POINTS: usize = 15;
pub(crate) const GAP_POINTS: usize = 9;

/// The scanner's sample points. See [`patterns`].
pub(crate) struct Patterns {
    /// Points over a star's solid centre, as offsets from the star's centre.
    pub core: Vec<Point>,
    /// One star's points for reading: the core, then the five limbs
    /// ([`LIMB_POINTS`] each), then the five gaps ([`GAP_POINTS`] each). Gap `k`
    /// lies between limbs `k` and `k + 1`.
    pub star: Vec<Point>,
    /// A grid over a star's disc, for matching the star's outline...
    pub grid: Vec<Point>,
    /// ...with a weight per grid point: positive inside the pentagram, negative
    /// outside it, each group summing to 1. The weighted sum of image levels is
    /// "mean outside minus mean inside" up to sign: large when a star sits here.
    pub weights: Vec<f32>,
    /// Every third point of the grid with its weight, for when there are
    /// hundreds of stars to match and a rougher answer will do.
    pub grid_thin: Vec<Point>,
    pub weights_thin: Vec<f32>,
    /// Offsets tried when re-locating a star near where a flat fit puts it.
    pub search: Vec<Point>,
    /// The same, wider, for a surface that bends.
    pub search_wide: Vec<Point>,
    /// Star centres of every ring.
    pub rings: Vec<Vec<Point>>,
    /// A circle in the clear band between the centre picture and ring 0's stars.
    pub band: Vec<Point>,
    /// Midway between neighbouring stars of ring 0: always background.
    pub between: Vec<Point>,
    /// A circle in the clear margin outside ring 0 of a one-ring code.
    pub quiet: Vec<Point>,
    /// Places near the centre, and all over the centre, where a look-alike loop has stars.
    pub inner_near: Vec<Point>,
    pub inner_all: Vec<Point>,
}

impl Patterns {
    /// Star centres of ring 0.
    pub fn ring0(&self) -> &[Point] {
        &self.rings[0]
    }
}

fn circle(radius: f64, count: usize, offset_degrees: f64) -> Vec<Point> {
    (0..count)
        .map(|i| (offset_degrees + 360.0 * i as f64 / count as f64).to_radians())
        .map(|a| [radius * a.cos(), radius * a.sin()])
        .collect()
}

/// `first, first + step, ...` up to and including `last`.
fn steps(first: f64, last: f64, step: f64) -> impl Iterator<Item = f64> + Clone {
    let count = ((last - first) / step + 1e-9).floor() as usize + 1;
    (0..count).map(move |i| first + step * i as f64)
}

fn square_grid(reach: f64, step: f64) -> Vec<Point> {
    let axis = steps(-reach, reach, step);
    axis.clone().flat_map(|x| axis.clone().map(move |y| [x, y])).collect()
}

fn build() -> Patterns {
    // A limb is a triangle from its base, at BASE_R from the centre and BASE_W
    // wide, to its tip at STAR_RO.
    let base_w = 2.0 * STAR_RI * 36f64.to_radians().sin();
    let base_r = STAR_RI * 36f64.to_radians().cos();
    let limb_len = STAR_RO - base_r;

    let core: Vec<Point> = square_grid(3.75, 1.5).into_iter().filter(|p| p[0] * p[0] + p[1] * p[1] <= 16.5).collect();
    let mut star = core.clone();
    for k in 0..5 {
        // Along the limb's axis, across the middle half of its width.
        let (ay, ax) = (-90.0 + 72.0 * k as f64).to_radians().sin_cos();
        for r in [8.0, 9.5, 11.0, 12.5, 14.0] {
            let w = base_w * (1.0 - (r - base_r) / limb_len);
            for t in [-0.25, 0.0, 0.25] {
                star.push([ax * r - ay * t * w, ay * r + ax * t * w]);
            }
        }
    }
    for k in 0..5 {
        // In the notch between limbs k and k + 1, beyond the notch's corner.
        let (ay, ax) = (-90.0 + 36.0 + 72.0 * k as f64).to_radians().sin_cos();
        for r in [15.0, 17.0, 19.0] {
            for t in [-2.0, 0.0, 2.0] {
                star.push([ax * r - ay * t, ay * r + ax * t]);
            }
        }
    }

    let grid: Vec<Point> = square_grid(21.0, 1.5).into_iter().filter(|p| p[0] * p[0] + p[1] * p[1] <= 22.0 * 22.0).collect();
    let outline = star_polygon(31, [0.0, 0.0]);
    let ink: Vec<bool> = grid.iter().map(|&p| inside(&outline, p)).collect();
    let inked = ink.iter().filter(|&&i| i).count() as f32;
    let clear = ink.len() as f32 - inked;
    let weights = ink.iter().map(|&i| if i { 1.0 / inked } else { -1.0 / clear }).collect();

    let weights: Vec<f32> = weights;
    let near = [-6.0, -3.0, 0.0, 3.0, 6.0];
    Patterns {
        core,
        star,
        grid_thin: grid.iter().step_by(THIN).copied().collect(),
        weights_thin: weights.iter().step_by(THIN).copied().collect(),
        grid,
        weights,
        search: square_grid(4.0, 1.0),
        search_wide: square_grid(8.0, 1.0),
        rings: (0..MAX_RINGS).map(ring_centers).collect(),
        band: circle(0.5 * (CENTER_R + R - STAR_RO), 72, 0.0),
        between: circle(R, 12, 15.0),
        quiet: circle(R + STAR_RO + 0.5 * MARGIN, 72, 0.0),
        inner_near: near.iter().flat_map(|&x| near.iter().map(move |&y| [x, y])).collect(),
        inner_all: square_grid(66.0, 3.0).into_iter().filter(|p| p[0] * p[0] + p[1] * p[1] <= 66.0 * 66.0).collect(),
    }
}

/// The scanner's sample points, built on first use.
pub(crate) fn patterns() -> &'static Patterns {
    static PATTERNS: OnceLock<Patterns> = OnceLock::new();
    PATTERNS.get_or_init(build)
}

/// How well a star outline `scale` times the standard size, centred at model
/// point `at`, matches the image under the flat fit `h`: mean level outside the
/// pentagram minus mean level inside it. Positive for dark ink on a light
/// background; callers flip the sign for the other polarity.
///
/// The fit is applied once, at the star's centre; across a star's width its
/// local linear map is indistinguishable from the full projective map.
pub(crate) fn outline_flat(image: &Gray, h: &Homography, at: Point, scale: f64) -> f64 {
    outline_local(image, h.apply(at), &(h.local_map(at) * scale), false)
}

/// As [`outline_flat`], for a star at image position `pos` with its own local
/// map. With `thin`, only every third outline point is used.
pub(crate) fn outline_local(image: &Gray, pos: Point, map: &LocalMap, thin: bool) -> f64 {
    let p = patterns();
    let (grid, weights) = if thin { (&p.grid_thin, &p.weights_thin) } else { (&p.grid, &p.weights) };
    let mut sum = 0.0f32;
    for (g, w) in grid.iter().zip(weights) {
        let q = add(pos, offset(map, *g));
        sum += w * image.sample(q[0], q[1]);
    }
    -sum as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_patterns_have_the_reference_sizes() {
        // Counts taken from the Python reference scanner.
        let p = patterns();
        assert_eq!(p.core.len(), 24);
        assert_eq!(p.star.len(), 24 + 5 * LIMB_POINTS + 5 * GAP_POINTS);
        assert_eq!(p.grid.len(), 673);
        assert_eq!(p.weights.iter().filter(|&&w| w > 0.0).count(), 198);
        assert_eq!((p.search.len(), p.search_wide.len(), p.inner_all.len()), (81, 289, 1517));
        assert_eq!(p.rings.len(), 24);
        assert_eq!(p.rings[23].len(), 150);
        // First limb point and first gap point, as in the reference.
        let limb = p.star[24];
        assert!((limb[0] + 1.868_288).abs() < 1e-5 && (limb[1] + 8.0).abs() < 1e-9);
        let gap = p.star[24 + 75];
        assert!((gap[0] - 7.198_745).abs() < 1e-5 && (gap[1] + 13.310_825).abs() < 1e-5);
    }

    #[test]
    fn outline_weights_balance() {
        let p = patterns();
        let (plus, minus): (Vec<f32>, Vec<f32>) = p.weights.iter().partition(|&&w| w > 0.0);
        assert!((plus.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!((minus.iter().sum::<f32>() + 1.0).abs() < 1e-4);
    }
}
