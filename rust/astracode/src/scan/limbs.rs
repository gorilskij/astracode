//! Reading the limbs: from grey levels to bits.
//!
//! Each star is sampled at its solid centre, on its five limbs and in the five
//! gaps between them ([`read_levels`]). A limb's *strength* compares it with
//! the background on its own two sides and with the star's centre
//! ([`limb_strength`]). All limbs of the code are then split into a light and
//! a solid group ([`classify`]), which calibrates the two shades for this
//! photo; nothing assumes a particular shade.

use nalgebra::{SMatrix, SVector};

use super::patterns::{patterns, GAP_POINTS, LIMB_POINTS};
use super::Placement;
use crate::format::R;
use crate::image::Gray;
use crate::math::{add, median, offset, Point};

/// Mean grey levels of the parts of each star.
pub(crate) struct Levels {
    pub core: Vec<f64>,
    pub limb: Vec<[f64; 5]>,
    /// Gap `k` lies between limbs `k` and `k + 1`.
    pub gap: Vec<[f64; 5]>,
}

/// Sample every star where `stars` places it.
pub(crate) fn read_levels(image: &Gray, stars: &Placement) -> Levels {
    let p = patterns();
    let n_core = p.core.len();
    let mean = |pos: Point, map, points: &[Point]| -> f64 {
        points.iter().map(|&g| add(pos, offset(map, g))).map(|q| image.sample(q[0], q[1]) as f64).sum::<f64>() / points.len() as f64
    };
    let mut levels = Levels { core: Vec::new(), limb: Vec::new(), gap: Vec::new() };
    for (&pos, map) in stars.at.iter().zip(&stars.maps) {
        levels.core.push(mean(pos, map, &p.star[..n_core]));
        let limbs = &p.star[n_core..n_core + 5 * LIMB_POINTS];
        let gaps = &p.star[n_core + 5 * LIMB_POINTS..];
        levels.limb.push(std::array::from_fn(|k| mean(pos, map, &limbs[k * LIMB_POINTS..(k + 1) * LIMB_POINTS])));
        levels.gap.push(std::array::from_fn(|k| mean(pos, map, &gaps[k * GAP_POINTS..(k + 1) * GAP_POINTS])));
    }
    levels
}

/// How far each star's centre stands out from the background around it, in
/// grey levels. Positive whatever the polarity.
pub(crate) fn contrast(levels: &Levels, polarity: f64) -> Vec<f64> {
    levels.core.iter().zip(&levels.gap).map(|(core, gap)| -polarity * (core - median(gap))).collect()
}

/// How inked each limb is, on a scale where 0 is background and about 1 is solid.
///
/// Each limb is compared with the background on its own two sides, so a shadow
/// edge or light gradient across a star cancels out; the star's always-solid
/// centre sets the scale.
///
/// With dark ink on a light background, the background beside a limb also says
/// how much light falls there, and that is divided out limb by limb. With light
/// ink on a dark background it says nothing of the kind: it is a small number,
/// mostly noise and stray light, and dividing by it would turn a little noise
/// in the dark into a large error. There the limb is simply measured against
/// the star's centre, and uneven light is left to [`classify`]'s model.
pub(crate) fn limb_strength(levels: &Levels, polarity: f64) -> Vec<[f64; 5]> {
    const EPS: f64 = 8.0;
    (0..levels.core.len())
        .map(|i| {
            let (core, limb, gap) = (levels.core[i], &levels.limb[i], &levels.gap[i]);
            let background = median(gap);
            if polarity > 0.0 {
                // The core's contrast, with the local light level divided out.
                let scale = ((background - core) / (background + EPS)).max(1e-3);
                std::array::from_fn(|k| {
                    let flank = 0.5 * (gap[k] + gap[(k + 4) % 5]);
                    (flank - limb[k]) / (flank + EPS) / scale
                })
            } else {
                let scale = (core - background).max(1e-3);
                std::array::from_fn(|k| (limb[k] - 0.5 * (gap[k] + gap[(k + 4) % 5])) / scale)
            }
        })
        .collect()
}

/// The result of splitting a code's limbs into light and solid.
pub(crate) struct Reading {
    /// Per star, whether each limb is solid.
    pub solid: Vec<[bool; 5]>,
    /// Per limb, how far it is from the light/solid threshold, as a fraction of
    /// half the gap between the two shades, capped at 1. Zero for a limb that
    /// could not be read at all.
    pub confidence: Vec<[f64; 5]>,
    /// The light shade's strength relative to solid, as measured in this photo.
    pub ratio: f64,
    /// The gap between the two shades, in units of their spread.
    pub separation: f64,
}

impl Reading {
    /// Each star's five bits as a symbol: top limb is bit 4, then clockwise.
    pub fn symbols(&self) -> Vec<u8> {
        self.solid.iter().map(|s| s.iter().fold(0u8, |acc, &bit| acc << 1 | bit as u8)).collect()
    }

    /// The least confident limb of each star.
    pub fn star_confidence(&self) -> Vec<f64> {
        self.confidence.iter().map(|c| c.iter().copied().fold(f64::INFINITY, f64::min)).collect()
    }
}

/// The value below which `q` percent of `sorted` lies, interpolating linearly.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let at = q / 100.0 * (sorted.len() - 1) as f64;
    let lo = at.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (at - lo as f64)
}

/// Split values into a low and a high group (two-means in one dimension).
/// Returns the two centres.
fn two_groups(values: &[f64]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (mut lo, mut hi) = (percentile(&sorted, 10.0), percentile(&sorted, 90.0));
    for _ in 0..20 {
        let cut = 0.5 * (lo + hi);
        let (high, low): (Vec<f64>, Vec<f64>) = values.iter().partition(|&&v| v > cut);
        if high.is_empty() || low.is_empty() {
            break;
        }
        let (new_lo, new_hi) = (low.iter().sum::<f64>() / low.len() as f64, high.iter().sum::<f64>() / high.len() as f64);
        let moved = (new_lo - lo).abs() + (new_hi - hi).abs();
        (lo, hi) = (new_lo, new_hi);
        if moved < 1e-5 {
            break;
        }
    }
    (lo, hi)
}

/// Pull of each star's own level towards the common one. See [`fit_limb_model`].
const STAR_RIDGE: f64 = 4.0;

/// Least-squares fit of the limb model
///
/// ```text
/// log strength = a[limb direction] + plane(star position) + d[star] + light * log(ratio)
/// ```
///
/// Blur, motion and lighting weaken limbs unevenly: by limb direction (motion
/// blur), smoothly across the code (perspective, focus), and star by star
/// (glare, reflections). The per-star offsets `d` are pulled towards zero. A
/// star's own five limbs easily outweigh that pull, so a star dimmed by glare
/// or shadow gets its own level; but for a star whose limbs are all one shade,
/// it is what says "that shade, at the usual level" rather than "the other
/// shade, at an unusual level".
///
/// The per-star offsets are eliminated exactly, star by star, which leaves an
/// eight-column problem however many stars there are.
///
/// Returns each limb's log strength with everything but the light term
/// removed, and the fitted `log(ratio)`.
fn fit_limb_model(v: &[[f64; 5]], usable: &[[bool; 5]], light: &[[bool; 5]], centers: &[Point]) -> (Vec<[f64; 5]>, f64) {
    type Row = SVector<f64, 8>;
    let row = |star: usize, k: usize| -> Row {
        let mut g = Row::zeros();
        g[k] = 1.0;
        g[5] = centers[star][0] / R;
        g[6] = centers[star][1] / R;
        g[7] = light[star][k] as u8 as f64;
        g
    };
    // Minimising over d[star] leaves r' (I - 11'/(count + ridge)) r for each star's
    // residuals r. `shrink` gives the square root of that matrix, I - shrink 11'.
    let mut normal = SMatrix::<f64, 8, 8>::zeros();
    let mut rhs = Row::zeros();
    for star in 0..v.len() {
        let used: Vec<usize> = (0..5).filter(|&k| usable[star][k]).collect();
        let count = used.len() as f64;
        if used.is_empty() {
            continue;
        }
        let shrink = (1.0 - (STAR_RIDGE / (count + STAR_RIDGE)).sqrt()) / count;
        let row_sum: Row = used.iter().map(|&k| row(star, k)).sum();
        let value_sum: f64 = used.iter().map(|&k| v[star][k]).sum();
        for &k in &used {
            let g = row(star, k) - shrink * row_sum;
            let value = v[star][k] - shrink * value_sum;
            normal += g * g.transpose();
            rhs += g * value;
        }
    }
    // Minimum-norm solution: columns can be exactly dependent (all limbs one shade).
    let eigen = normal.symmetric_eigen();
    let cut = 1e-12 * eigen.eigenvalues.max();
    let mut coef = Row::zeros();
    for (i, &lambda) in eigen.eigenvalues.iter().enumerate() {
        if lambda > cut {
            let axis = eigen.eigenvectors.column(i);
            coef += axis * (axis.dot(&rhs) / lambda);
        }
    }

    let log_ratio = coef[7];
    let mut without_light = coef;
    without_light[7] = 0.0;
    let residuals = (0..v.len())
        .map(|star| {
            let used = (0..5).filter(|&k| usable[star][k]);
            let count = used.clone().count() as f64;
            let own: f64 = used.map(|k| v[star][k] - row(star, k).dot(&coef)).sum::<f64>() / (count + STAR_RIDGE);
            std::array::from_fn(|k| v[star][k] - row(star, k).dot(&without_light) - own)
        })
        .collect();
    (residuals, log_ratio)
}

/// Decide light or solid for every limb.
///
/// Rather than one threshold, this fits the model of [`fit_limb_model`],
/// alternating between the fit and the assignment until they agree.
///
/// With `robust`, the spread used for [`Reading::separation`] ignores
/// outliers: with hundreds of stars, a few ruined ones (torn off, covered)
/// must not decide whether the two shades count as separable. That is for the
/// error correction.
pub(crate) fn classify(strength: &[[f64; 5]], centers: &[Point], robust: bool) -> Reading {
    let n = strength.len();
    let flat: Vec<f64> = strength.iter().flatten().copied().collect();
    // A limb washed out by glare or covered reads as hardly inked at all. Leave
    // such limbs out of the fit and report them with zero confidence.
    let floor = 0.3 * median(&flat);
    let usable: Vec<[bool; 5]> = strength.iter().map(|s| s.map(|x| x > floor)).collect();
    let v: Vec<[f64; 5]> = strength.iter().map(|s| s.map(|x| x.max(1e-3).ln())).collect();
    let used: Vec<f64> = (0..n).flat_map(|i| (0..5).map(move |k| (i, k))).filter(|&(i, k)| usable[i][k]).map(|(i, k)| v[i][k]).collect();
    if used.len() < 4 {
        // Nothing to read: report shades that cannot be told apart.
        return Reading { solid: vec![[false; 5]; n], confidence: vec![[0.0; 5]; n], ratio: 1.0, separation: 0.0 };
    }

    let (lo, hi) = two_groups(&used);
    let mut light: Vec<[bool; 5]> = v.iter().map(|s| s.map(|x| x < 0.5 * (lo + hi))).collect();
    let (mut residual, mut log_ratio) = (Vec::new(), 0.0);
    for _ in 0..10 {
        (residual, log_ratio) = fit_limb_model(&v, &usable, &light, centers);
        log_ratio = log_ratio.min(-0.05);
        let next: Vec<[bool; 5]> = residual.iter().map(|r| r.map(|x| x < 0.5 * log_ratio)).collect();
        if next == light {
            break;
        }
        light = next;
    }
    let light: Vec<[bool; 5]> = residual.iter().map(|r| r.map(|x| x < 0.5 * log_ratio)).collect();

    let confidence = (0..n)
        .map(|i| {
            std::array::from_fn(|k| {
                if usable[i][k] {
                    ((residual[i][k] - 0.5 * log_ratio).abs() / (0.5 * log_ratio.abs())).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
        })
        .collect();
    // Each usable limb's distance from the centre of its own shade.
    let errors: Vec<f64> = (0..n)
        .flat_map(|i| (0..5).map(move |k| (i, k)))
        .filter(|&(i, k)| usable[i][k])
        .map(|(i, k)| residual[i][k] - if light[i][k] { log_ratio } else { 0.0 })
        .collect();
    let spread = if robust {
        let middle = median(&errors);
        1.4826 * median(&errors.iter().map(|e| (e - middle).abs()).collect::<Vec<f64>>())
    } else {
        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        (errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / errors.len() as f64).sqrt()
    };
    Reading {
        solid: light.iter().map(|l| l.map(|x| !x)).collect(),
        confidence,
        ratio: log_ratio.exp(),
        separation: log_ratio.abs() / (spread + 1e-6),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strengths for the given symbols with the light shade at `ratio`, some
    /// stars dimmer overall, and a little noise.
    fn strengths(symbols: &[u8], ratio: f64) -> Vec<[f64; 5]> {
        symbols
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let dim = 1.0 - 0.04 * (i % 5) as f64;
                std::array::from_fn(|k| {
                    let solid = (s >> (4 - k)) & 1 == 1;
                    let noise = 1.0 + 0.02 * (((i * 7 + k * 3) % 5) as f64 - 2.0);
                    dim * noise * if solid { 1.0 } else { ratio }
                })
            })
            .collect()
    }

    #[test]
    fn two_shades_are_told_apart_and_measured_as_in_the_reference() {
        let symbols: Vec<u8> = vec![0b10110, 0, 31, 0b00101, 0b11000, 7, 19, 0b01010, 28, 3, 31, 0];
        let centers = &patterns().rings[0];
        // (light level, ratio and separation the Python reference measures on the same input)
        for (level, ratio, separation) in [(0.4, 0.3952, 24.162), (0.6, 0.5927, 13.610), (0.8, 0.7877, 5.919)] {
            let reading = classify(&strengths(&symbols, level), centers, false);
            assert!((reading.ratio - ratio).abs() < 1e-3, "ratio {} at level {level}", reading.ratio);
            assert!((reading.separation - separation).abs() < 0.02, "separation {} at level {level}", reading.separation);
            if level < 0.8 {
                assert_eq!(reading.symbols(), symbols, "light level {level}");
            }
        }
    }

    #[test]
    fn light_on_dark_is_measured_as_steadily_as_dark_on_light() {
        // One star: solid centre, limbs alternately solid and 70% light, on a background 177 levels away.
        let star = |background: f64, ink: f64, wobble: f64| Levels {
            core: vec![ink],
            limb: vec![std::array::from_fn(|k| if k % 2 == 0 { ink } else { background + 0.7 * (ink - background) })],
            // The background beside each limb is off by a few grey levels, as noise makes it.
            gap: vec![std::array::from_fn(|k| background + wobble * [1.0, -1.0, 0.5, -0.5, 0.0][k])],
        };
        let dark_on_light = limb_strength(&star(197.0, 20.0, 0.0), 1.0)[0];
        let light_on_dark = limb_strength(&star(20.0, 197.0, 0.0), -1.0)[0];
        for k in 0..5 {
            let expected = if k % 2 == 0 { 1.0 } else { 0.7 };
            assert!((dark_on_light[k] - expected).abs() < 0.02 && (light_on_dark[k] - expected).abs() < 1e-9);
        }
        // Six grey levels of noise in the background move a limb's strength by a few percent
        // in either polarity, not by a third as dividing by a dark background would.
        for (levels, polarity) in [(star(197.0, 20.0, 6.0), 1.0), (star(20.0, 197.0, 6.0), -1.0)] {
            let noisy = limb_strength(&levels, polarity)[0];
            for k in 0..5 {
                let expected = if k % 2 == 0 { 1.0 } else { 0.7 };
                assert!((noisy[k] - expected).abs() < 0.06, "limb {k}: {} with polarity {polarity}", noisy[k]);
            }
        }
    }

    #[test]
    fn a_washed_out_limb_gets_no_confidence() {
        let symbols: Vec<u8> = vec![0b10110, 0, 31, 0b00101, 0b11000, 7, 19, 0b01010, 28, 3, 31, 0];
        let mut s = strengths(&symbols, 0.6);
        s[4][2] = 0.01;
        let reading = classify(&s, &patterns().rings[0], false);
        assert_eq!(reading.confidence[4][2], 0.0);
        assert_eq!(reading.star_confidence()[4], 0.0);
        assert!(reading.star_confidence()[0] > 0.5);
    }

    #[test]
    fn percentiles_interpolate() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 50.0), 3.0);
        assert!((percentile(&v, 90.0) - 4.6).abs() < 1e-12);
    }
}
