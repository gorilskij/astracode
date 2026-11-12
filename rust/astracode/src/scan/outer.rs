//! The rings beyond ring 0: finding their stars and reading the message.

use nalgebra::{Matrix3, Matrix3x2, RowVector2, Vector3};

use super::align::{local_fit, Neighbours};
use super::bent::fit_shapes;
use super::limbs::{classify, limb_strength, read_levels};
use super::patterns::{patterns, THIN};
use super::{Code, Payload, View, MIN_SHADE_SEPARATION};
use crate::format::ring0::Header;
use crate::format::{half_side, message, ring_radius, ring_stars, MAX_RINGS, STAR_RO};
use crate::image::Gray;
use crate::math::{add, offset, sub, Homography, LocalMap, Point};

/// How far from its predicted place a star is looked for, in model units.
const SEARCH_UNITS: f64 = 7.0;
/// Star sizes tried, relative to the flat fit. A bent surface shows some stars
/// larger and some smaller.
const STAR_SIZES: [f64; 3] = [0.75, 1.0, 1.3];
/// The image is straightened again after this many rings have been located.
const CHUNKS: [usize; 4] = [3, 7, 15, MAX_RINGS];
/// Clear space kept around the outermost ring in the straightened image, in
/// model units: enough for the largest star outline at the edge of the search.
const BORDER: f64 = 17.0;

/// The star outline as pixel offsets into a straightened image, for one star size.
struct Stencil {
    /// Offset of each outline point from the star's centre, as `(dx, dy)` in pixels.
    points: Vec<(i32, i32)>,
    /// The same as offsets into the pixel array, for when the whole stencil is inside the image.
    indices: Vec<isize>,
    weights: Vec<f32>,
    /// Every third point, as offsets into the pixel array, with its weight.
    indices_thin: Vec<isize>,
    weights_thin: Vec<f32>,
    /// The furthest any point is from the centre, in pixels.
    reach: i32,
}

impl Stencil {
    fn new(pixels_per_unit: f64, size: f64, image_width: usize) -> Stencil {
        let p = patterns();
        let points: Vec<(i32, i32)> = p
            .grid
            .iter()
            .map(|g| ((g[0] * size * pixels_per_unit).round() as i32, (g[1] * size * pixels_per_unit).round() as i32))
            .collect();
        let indices: Vec<isize> = points.iter().map(|&(dx, dy)| dy as isize * image_width as isize + dx as isize).collect();
        Stencil {
            indices_thin: indices.iter().step_by(THIN).copied().collect(),
            weights_thin: p.weights_thin.clone(),
            indices,
            reach: points.iter().map(|&(dx, dy)| dx.abs().max(dy.abs())).max().unwrap_or(0),
            weights: p.weights.clone(),
            points,
        }
    }

    /// Mean level outside the pentagram minus mean level inside it, for a star
    /// centred on pixel `(x, y)`. With `thin`, only every third outline point is used.
    fn outline(&self, image: &Gray, x: i32, y: i32, thin: bool) -> f32 {
        let (w, h) = (image.width as i32, image.height as i32);
        let mut sum = 0.0f32;
        if x >= self.reach && y >= self.reach && x + self.reach < w && y + self.reach < h {
            let base = (y * w + x) as isize;
            let (indices, weights) = if thin { (&self.indices_thin, &self.weights_thin) } else { (&self.indices, &self.weights) };
            for (&i, &weight) in indices.iter().zip(weights) {
                sum += weight * image.data[(base + i) as usize];
            }
        } else {
            // Partly outside (rare): the edge pixels stand in for what is beyond them.
            for (&(dx, dy), &weight) in self.points.iter().zip(&self.weights) {
                let (px, py) = ((x + dx).clamp(0, w - 1), (y + dy).clamp(0, h - 1));
                sum += weight * image.data[(py * w + px) as usize];
            }
        }
        -sum
    }
}

/// Predict where the stars of a new ring are, in straightened pixels, from a
/// local affine fit to each one's nearest located neighbours (or to all of
/// them, for ring 1, whose only neighbours are the twelve stars of ring 0).
fn predict_ring(new: &[Point], known: &[Point], known_at: &[Point], use_all: bool) -> Vec<Point> {
    new.iter()
        .map(|&m| {
            let mut nearest: Vec<usize> = (0..known.len()).collect();
            if !use_all && known.len() > 8 {
                let d2 = |i: usize| (known[i][0] - m[0]).powi(2) + (known[i][1] - m[1]).powi(2);
                nearest.select_nth_unstable_by(7, |&a, &b| d2(a).total_cmp(&d2(b)));
                nearest.truncate(8);
            }
            let mut a = Matrix3::<f64>::identity() * 1e-9;
            let mut b = Matrix3x2::<f64>::zeros();
            for &i in &nearest {
                let x = Vector3::new(known[i][0], known[i][1], 1.0);
                a += x * x.transpose();
                b += x * RowVector2::new(known_at[i][0], known_at[i][1]);
            }
            match a.lu().solve(&b) {
                Some(coef) => {
                    let at = Vector3::new(m[0], m[1], 1.0).transpose() * coef;
                    [at[0], at[1]]
                }
                None => known_at[nearest[0]],
            }
        })
        .collect()
}

/// Image positions of every star of rings `0..rings`, working outwards from ring 0.
///
/// The image is straightened with the fit so far, and each new ring is then
/// looked for where its already located neighbours say it should be, by
/// matching the star outline around that spot. Predicting from neighbours
/// rather than from one flat fit is what follows a surface that is not flat.
///
/// `ring0` holds ring 0's stars as found. `shapes`, if given, are ring 0's
/// local maps as fitted from the star shapes: ring 0 by itself only shows how
/// the surface runs along the ring, but the shapes of its stars also show how
/// it runs outwards, towards ring 1.
pub(crate) fn locate_rings(view: View, h: &Homography, rings: usize, ring0: &[Point], shapes: Option<&[LocalMap]>) -> Vec<Vec<Point>> {
    let centers = &patterns().rings[..rings];
    let mut found: Vec<Vec<Point>> = vec![ring0.to_vec()];
    // Straighten at about the photo's own resolution.
    let native = h.local_map([0.0, 0.0]).determinant().abs().sqrt();
    let ppu = native.clamp(0.8, 2.0);
    let mut done = 1;
    for stop in CHUNKS {
        let stop = stop.min(rings);
        if stop <= done {
            continue;
        }
        let fit = Homography::fit(&centers[..done].concat(), &found.concat());
        let half = ring_radius(stop - 1) + STAR_RO + BORDER;
        let side = (2.0 * half * ppu).ceil() as usize;
        // Straightened pixel (u, v) is model point (u / ppu - half, v / ppu - half).
        let to_image = fit.after(&Homography::scale_then_shift(1.0 / ppu, 1.0 / ppu, -half, -half));
        let Some(from_image) = to_image.inverse() else { break };
        let flat = view.image.warp(&to_image, side, side);
        let stencils = STAR_SIZES.map(|size| Stencil::new(ppu, size, side));
        // The best of the star sizes at a pixel. A third of the outline points is enough to
        // find the neighbourhood of the peak; the peak itself is settled with all of them.
        let score_with = |x: i32, y: i32, thin: bool| -> f32 {
            stencils.iter().map(|s| view.polarity as f32 * s.outline(&flat, x, y, thin)).fold(f32::NEG_INFINITY, f32::max)
        };
        let rough = |x: i32, y: i32| score_with(x, y, true);
        let score = |x: i32, y: i32| score_with(x, y, false);
        let mut at: Vec<Vec<Point>> = found.iter().map(|ring| ring.iter().map(|&p| from_image.apply(p)).collect()).collect();

        let reach = (SEARCH_UNITS * ppu).ceil() as i32;
        // Search on a grid of about one model unit, then pixel by pixel around the best.
        let stride = (ppu.round() as i32).max(1);
        for k in done..stop {
            let guess: Vec<Point> = match shapes {
                Some(shapes) if k == 1 => centers[1]
                    .iter()
                    .map(|&m| {
                        // Step out from each of the two nearest ring 0 stars along its own map, and average.
                        let mut nearest: Vec<usize> = (0..12).collect();
                        nearest.sort_by(|&a, &b| {
                            let d2 = |i: usize| (centers[0][i][0] - m[0]).powi(2) + (centers[0][i][1] - m[1]).powi(2);
                            d2(a).total_cmp(&d2(b))
                        });
                        let from = |i: usize| add(ring0[i], offset(&shapes[i], sub(m, centers[0][i])));
                        let (a, b) = (from(nearest[0]), from(nearest[1]));
                        from_image.apply([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])])
                    })
                    .collect(),
                _ => {
                    let recent = k.saturating_sub(2)..k;
                    predict_ring(&centers[k], &centers[recent.clone()].concat(), &at[recent].concat(), k == 1)
                }
            };
            let limit = (side as i32 - reach - 2).max(reach + 1);
            let here = super::par::map(guess.len(), |i| {
                let gx = (guess[i][0].round() as i32).clamp(reach + 1, limit);
                let gy = (guess[i][1].round() as i32).clamp(reach + 1, limit);
                let mut best = (f32::NEG_INFINITY, gx, gy);
                let coarse = (reach / stride) * stride;
                let try_at = |x: i32, y: i32, best: &mut (f32, i32, i32)| {
                    let s = rough(x, y);
                    if s > best.0 {
                        *best = (s, x, y);
                    }
                };
                for dy in (-coarse..=coarse).step_by(stride as usize) {
                    for dx in (-coarse..=coarse).step_by(stride as usize) {
                        try_at(gx + dx, gy + dy, &mut best);
                    }
                }
                if stride > 1 {
                    let (cx, cy) = (best.1, best.2);
                    for dy in 1 - stride..stride {
                        for dx in 1 - stride..stride {
                            try_at(cx + dx, cy + dy, &mut best);
                        }
                    }
                }
                // Settle the peak with the full outline: climb to the best of the 3 x 3 pixels
                // around it, twice at most.
                let (mut bx, mut by) = (best.1, best.2);
                let mut moves = 0;
                let around: Vec<(f32, i32, i32)> = loop {
                    let around: Vec<(f32, i32, i32)> = (-1..=1)
                        .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
                        .map(|(dx, dy)| (score(bx + dx, by + dy), dx, dy))
                        .collect();
                    let top = around.iter().fold(around[4], |top, &a| if a.0 > top.0 { a } else { top });
                    if (top.1, top.2) == (0, 0) || moves == 2 {
                        break around;
                    }
                    (bx, by) = (bx + top.1, by + top.2);
                    moves += 1;
                };
                // Below a pixel: the centroid of the 3 x 3 scores around the peak.
                let floor = around.iter().map(|a| a.0).fold(f32::INFINITY, f32::min);
                let (mut wx, mut wy, mut total) = (0.0f64, 0.0f64, 0.0f64);
                for &(s, dx, dy) in &around {
                    let w = (s - floor) as f64 + 1e-6;
                    wx += w * dx as f64;
                    wy += w * dy as f64;
                    total += w;
                }
                [bx as f64 + wx / total, by as f64 + wy / total]
            });
            found.push(here.iter().map(|&p| to_image.apply(p)).collect());
            at.push(here);
        }
        done = stop;
    }
    found
}

/// Read the outer rings of a code whose ring 0 announced them.
///
/// `ring0` and `shapes` are ring 0's stars as the caller found them; see [`locate_rings`].
pub(crate) fn read_rings(
    view: View,
    h: &Homography,
    mirrored: bool,
    header: &Header,
    ring0: &[Point],
    shapes: Option<&[LocalMap]>,
) -> Option<Code> {
    let rings = header.rings;
    let found: Vec<Point> = locate_rings(view, h, rings, ring0, shapes).concat();
    let centers: Vec<Point> = patterns().rings[..rings].concat();
    if found.len() != centers.len() {
        return None;
    }
    let h = Homography::fit(&centers, &found);
    let mut stars = local_fit(&h, &centers, &found, Neighbours::Surrounded);
    // First with each star placed by its neighbours, which is all a flat or gently curved
    // surface needs. If that does not decode, fit every star's own outline as well: on a
    // strongly bent surface the neighbours leave a star about a unit off centre, enough
    // to misread one limb in a hundred, which is one star pair in ten.
    for shaped in [false, true] {
        if shaped {
            stars = fit_shapes(view, &stars);
        }
        let reading = classify(&limb_strength(&read_levels(view.image, &stars), view.polarity), &centers, true);
        if !(0.3 < reading.ratio && reading.ratio < 0.97) || reading.separation < MIN_SHADE_SEPARATION {
            continue;
        }
        let symbols = reading.symbols();
        let mut outer_rings = Vec::with_capacity(rings - 1);
        let mut at = 12;
        for k in 1..rings {
            outer_rings.push(symbols[at..at + ring_stars(k)].to_vec());
            at += ring_stars(k);
        }
        let Some((bytes, repaired)) = message::decode_outer(header, &outer_rings) else { continue };
        let half = half_side(rings);
        return Some(Code {
            payload: Payload::Message { level: header.level, bytes },
            rings,
            corners: [[-half, -half], [half, -half], [half, half], [-half, half]].map(|c| h.apply(c)),
            mirrored,
            repaired,
            shaped,
            confidence: reading.star_confidence().into_iter().fold(f64::INFINITY, f64::min),
            separation: reading.separation,
            light_level: reading.ratio,
        });
    }
    None
}
