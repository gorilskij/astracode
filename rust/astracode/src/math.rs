//! Plane geometry for the scanner: points, local linear maps and homographies.

use nalgebra::{Matrix2, Matrix3, SMatrix, Vector3};

/// A point or offset, `[x, y]`, in model units or in image pixels depending on context.
pub type Point = [f64; 2];

/// A local linear map from model offsets to image offsets around one star:
/// `image offset = map * model offset`. It carries the star's local scale,
/// rotation, stretch and shear.
pub type LocalMap = Matrix2<f64>;

/// Apply a local map to a model offset.
#[inline]
pub fn offset(map: &LocalMap, v: Point) -> Point {
    [map[(0, 0)] * v[0] + map[(0, 1)] * v[1], map[(1, 0)] * v[0] + map[(1, 1)] * v[1]]
}

#[inline]
pub fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1]]
}

#[inline]
pub fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}

#[inline]
pub fn distance(a: Point, b: Point) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// The median of a list of numbers (the mean of the two middle ones for an even count).
pub fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// A projective map of the plane: how a flat code appears in a photo.
///
/// Throughout the scanner a `Homography` maps **model units to image pixels**
/// unless a name says otherwise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography(Matrix3<f64>);

impl Homography {
    pub fn new(matrix: Matrix3<f64>) -> Homography {
        Homography(matrix)
    }

    pub fn matrix(&self) -> &Matrix3<f64> {
        &self.0
    }

    /// Scale by `sx, sy`, then move by `tx, ty`.
    pub fn scale_then_shift(sx: f64, sy: f64, tx: f64, ty: f64) -> Homography {
        Homography(Matrix3::new(sx, 0.0, tx, 0.0, sy, ty, 0.0, 0.0, 1.0))
    }

    /// Rotation about the origin by `steps` ring positions of 30 degrees,
    /// clockwise on screen.
    pub fn ring_rotation(steps: i32) -> Homography {
        let (s, c) = (30.0 * steps as f64).to_radians().sin_cos();
        Homography(Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0))
    }

    /// Reflection in the vertical axis: a code seen from behind.
    pub fn mirror() -> Homography {
        Homography::scale_then_shift(-1.0, 1.0, 0.0, 0.0)
    }

    /// The map that applies `first` and then `self`.
    pub fn after(&self, first: &Homography) -> Homography {
        Homography(self.0 * first.0)
    }

    pub fn inverse(&self) -> Option<Homography> {
        self.0.try_inverse().map(Homography)
    }

    #[inline]
    pub fn apply(&self, p: Point) -> Point {
        let v = self.0 * Vector3::new(p[0], p[1], 1.0);
        [v[0] / v[2], v[1] / v[2]]
    }

    /// The local linear map at `p`: how one model unit along x and along y
    /// appears in the image there.
    pub fn local_map(&self, p: Point) -> LocalMap {
        let at = self.apply(p);
        let dx = sub(self.apply([p[0] + 1.0, p[1]]), at);
        let dy = sub(self.apply([p[0], p[1] + 1.0]), at);
        Matrix2::new(dx[0], dy[0], dx[1], dy[1])
    }

    /// The least-squares homography taking each `src` point to its `dst`
    /// point (normalised direct linear transform). Needs at least four pairs.
    pub fn fit(src: &[Point], dst: &[Point]) -> Homography {
        assert_eq!(src.len(), dst.len());
        // Move each set to its centroid and scale it to a mean distance of sqrt 2,
        // which keeps the system well conditioned.
        let normalise = |pts: &[Point]| {
            let n = pts.len() as f64;
            let c = [pts.iter().map(|p| p[0]).sum::<f64>() / n, pts.iter().map(|p| p[1]).sum::<f64>() / n];
            let spread = pts.iter().map(|&p| distance(p, c)).sum::<f64>() / n;
            let s = std::f64::consts::SQRT_2 / (spread + 1e-12);
            Homography::scale_then_shift(s, s, -s * c[0], -s * c[1])
        };
        let (ts, td) = (normalise(src), normalise(dst));
        // Each pair gives two equations a . h = 0 in the nine entries of H; the
        // solution is the direction in which the sum of a a' is smallest.
        let mut ata = SMatrix::<f64, 9, 9>::zeros();
        for (&s, &d) in src.iter().zip(dst) {
            let (s, d) = (ts.apply(s), td.apply(d));
            let rows = [
                [s[0], s[1], 1.0, 0.0, 0.0, 0.0, -d[0] * s[0], -d[0] * s[1], -d[0]],
                [0.0, 0.0, 0.0, s[0], s[1], 1.0, -d[1] * s[0], -d[1] * s[1], -d[1]],
            ];
            for row in rows {
                let a = SMatrix::<f64, 9, 1>::from_column_slice(&row);
                ata += a * a.transpose();
            }
        }
        let eigen = ata.symmetric_eigen();
        let smallest = eigen.eigenvalues.imin();
        let h = eigen.eigenvectors.column(smallest);
        let h = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
        let m = td.0.try_inverse().expect("a scaling is invertible") * h * ts.0;
        Homography(m / m[(2, 2)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fit_recovers_the_map_it_was_made_from() {
        let truth = Homography::new(Matrix3::new(1.2, 0.3, 400.0, -0.2, 0.9, 300.0, 0.0004, -0.0003, 1.0));
        let src: Vec<Point> = (0..12).map(|i| (30.0 * i as f64).to_radians()).map(|a| [100.0 * a.cos(), 100.0 * a.sin()]).collect();
        let dst: Vec<Point> = src.iter().map(|&p| truth.apply(p)).collect();
        let fit = Homography::fit(&src, &dst);
        for p in [[0.0, 0.0], [150.0, -80.0], [-60.0, 200.0]] {
            assert!(distance(fit.apply(p), truth.apply(p)) < 1e-6);
        }
        let back = fit.inverse().unwrap();
        assert!(distance(back.apply(fit.apply([12.0, 34.0])), [12.0, 34.0]) < 1e-9);
    }

    #[test]
    fn ring_rotation_moves_a_star_to_the_next_place() {
        let top = [0.0, -100.0];
        let next = Homography::ring_rotation(1).apply(top);
        assert!(distance(next, [50.0, -86.602_540_378]) < 1e-6);
        assert!(distance(Homography::mirror().apply([3.0, 4.0]), [-3.0, 4.0]) < 1e-12);
    }

    #[test]
    fn the_local_map_matches_the_homography_nearby() {
        let h = Homography::new(Matrix3::new(1.2, 0.3, 400.0, -0.2, 0.9, 300.0, 0.0002, -0.0001, 1.0));
        let p = [80.0, -40.0];
        let map = h.local_map(p);
        let moved = add(h.apply(p), offset(&map, [3.0, -2.0]));
        assert!(distance(moved, h.apply([83.0, -42.0])) < 0.05);
        assert_eq!(median(&[5.0, 1.0, 3.0]), 3.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}
