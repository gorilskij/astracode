//! Stage 1: from pixels to candidate star blobs.
//!
//! The image is binarised against its local mean, so that uneven light does
//! not matter, and every connected dark (or light) region of plausible size
//! and shape becomes a [`Blob`]. How far from the mean a pixel must be rises
//! with the image's noise ([`noise_level`]), so that noise does not become blobs.

use crate::image::Gray;

/// A connected region of the binarised image: possibly a star.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Blob {
    /// Centroid, in pixels of the image that was binarised.
    pub x: f64,
    pub y: f64,
    /// Number of pixels.
    pub area: f64,
}

/// Blobs are dropped when there are more than this many, smallest first.
const MAX_BLOBS: usize = 3500;

/// Smallest blob worth considering, in pixels.
const MIN_AREA: f64 = 9.0;

/// Connected regions (8-connectivity) of the pixels where `ink` is true, kept
/// if they could be a star: not tiny, not huge, not a line, not a hollow frame.
///
/// Works on runs of ink along each row rather than on single pixels: a run
/// joins every run of the row above that it touches, and a region's size,
/// centroid and bounding box are summed run by run. No label image is needed.
pub(crate) fn find_blobs(width: usize, height: usize, ink: &[bool], max_area: f64) -> Vec<Blob> {
    #[derive(Clone, Copy)]
    struct Stats {
        area: f64,
        sx: f64,
        sy: f64,
        x0: u32,
        x1: u32,
        y0: u32,
        y1: u32,
    }
    /// A run of ink in one row: pixels `start..end`, belonging to region `label`.
    #[derive(Clone, Copy)]
    struct Run {
        start: u32,
        end: u32,
        label: u32,
    }
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }

    // Provisional regions, joined in a union-find forest as rows reveal that they touch.
    let mut parent: Vec<u32> = Vec::new();
    let mut stats: Vec<Stats> = Vec::new();
    let (mut above, mut here): (Vec<Run>, Vec<Run>) = (Vec::new(), Vec::new());
    for y in 0..height {
        let row = &ink[y * width..(y + 1) * width];
        here.clear();
        let mut first = 0; // the first run above that could still touch a run in this row
        let mut x = 0;
        while x < width {
            if !row[x] {
                x += 1;
                continue;
            }
            let start = x as u32;
            while x < width && row[x] {
                x += 1;
            }
            let end = x as u32;
            // Runs above touch this one if they overlap it or meet it corner to corner.
            while first < above.len() && above[first].end < start {
                first += 1;
            }
            let mut label = u32::MAX;
            let mut i = first;
            while i < above.len() && above[i].start <= end {
                let other = root(&mut parent, above[i].label);
                if label == u32::MAX {
                    label = other;
                } else if other != label {
                    let (lo, hi) = (label.min(other), label.max(other));
                    parent[hi as usize] = lo;
                    label = lo;
                }
                i += 1;
            }
            if label == u32::MAX {
                label = parent.len() as u32;
                parent.push(label);
                stats.push(Stats { area: 0.0, sx: 0.0, sy: 0.0, x0: u32::MAX, x1: 0, y0: y as u32, y1: y as u32 });
            }
            let len = (end - start) as f64;
            let s = &mut stats[label as usize];
            s.area += len;
            s.sx += len * (start + end - 1) as f64 / 2.0;
            s.sy += len * y as f64;
            s.x0 = s.x0.min(start);
            s.x1 = s.x1.max(end - 1);
            s.y1 = y as u32;
            here.push(Run { start, end, label });
        }
        std::mem::swap(&mut above, &mut here);
    }
    // Fold every provisional region into the one it ended up part of.
    for label in (0..parent.len() as u32).rev() {
        let into = root(&mut parent, label);
        if into != label {
            let from = stats[label as usize];
            let s = &mut stats[into as usize];
            s.area += from.area;
            s.sx += from.sx;
            s.sy += from.sy;
            s.x0 = s.x0.min(from.x0);
            s.x1 = s.x1.max(from.x1);
            s.y0 = s.y0.min(from.y0);
            s.y1 = s.y1.max(from.y1);
        }
    }

    let mut blobs: Vec<Blob> = (0..parent.len())
        .filter(|&label| parent[label] == label as u32)
        .filter_map(|label| {
            let s = &stats[label];
            let (w, h) = ((s.x1 - s.x0 + 1) as f64, (s.y1 - s.y0 + 1) as f64);
            let plausible = s.area >= MIN_AREA && s.area <= max_area && w < 3.5 * h && h < 3.5 * w && s.area > 0.12 * w * h;
            plausible.then_some(Blob { x: s.sx / s.area, y: s.sy / s.area, area: s.area })
        })
        .collect();
    if blobs.len() > MAX_BLOBS {
        blobs.sort_by(|a, b| b.area.total_cmp(&a.area));
        blobs.truncate(MAX_BLOBS);
    }
    blobs
}

/// The standard deviation of the image's pixel noise, judged from its flattest parts.
///
/// The image is cut into squares; in each, the second difference along the
/// rows (which a smooth change of light does not show up in) gives a noise
/// level, and the answer is that of a square at the quiet end of the range.
/// Squares with edges or texture in them come out higher and are passed over,
/// as long as a fifth of the image is flat.
pub(crate) fn noise_level(image: &Gray) -> f32 {
    const SQUARE: usize = 16;
    let (w, h) = (image.width, image.height);
    if w < SQUARE || h < SQUARE {
        return 0.0;
    }
    let mut energy: Vec<f32> = Vec::with_capacity((w / SQUARE) * (h / SQUARE));
    for top in (0..=h - SQUARE).step_by(SQUARE) {
        for left in (0..=w - SQUARE).step_by(SQUARE) {
            let mut sum = 0.0f32;
            for y in top..top + SQUARE {
                let row = &image.data[y * w + left..y * w + left + SQUARE];
                sum += row.windows(3).map(|v| (v[0] - 2.0 * v[1] + v[2]).powi(2)).sum::<f32>();
            }
            energy.push(sum / (SQUARE * (SQUARE - 2)) as f32);
        }
    }
    let quiet = energy.len() / 5;
    let (_, level, _) = energy.select_nth_unstable_by(quiet, |a, b| a.total_cmp(b));
    // The second difference of independent noise has six times its variance.
    (*level / 6.0).sqrt()
}

/// The blobs of one image, for both polarities and two neighbourhood sizes.
///
/// Returns `(polarity, blobs)` four times over. Polarity is `1.0` for dark ink
/// on a light background and `-1.0` for light ink on dark.
pub(crate) fn binarise(image: &Gray) -> Vec<(f64, Vec<Blob>)> {
    /// A pixel is ink when it differs from its local mean by more than this many levels,
    const CONTRAST: f32 = 7.0;
    /// or, in a noisy image, by this many times what is left of the noise after smoothing;
    /// otherwise the noise itself turns into thousands of blobs.
    const NOISE_CLEARANCE: f32 = 3.0;
    /// What a Gaussian blur of radius 1 leaves of independent pixel noise.
    const SMOOTHED: f32 = 0.282;
    const WINDOWS: [usize; 2] = [31, 91];
    let contrast = CONTRAST.max(NOISE_CLEARANCE * SMOOTHED * noise_level(image));
    let smooth = image.gaussian_blur(1.0);
    let max_area = (smooth.width.min(smooth.height) as f64 / 5.0).powi(2);
    // The four combinations are independent of each other.
    let means = super::par::map_few(2, |i| smooth.box_blur(WINDOWS[i]));
    super::par::map_few(4, |i| {
        let (mean, polarity) = (&means[i / 2], if i % 2 == 0 { 1.0f32 } else { -1.0 });
        let ink: Vec<bool> = smooth.data.iter().zip(&mean.data).map(|(v, m)| (v - m) * polarity < -contrast).collect();
        (polarity as f64, find_blobs(smooth.width, smooth.height, &ink, max_area))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_touching_at_a_corner_are_one_blob() {
        // Two 3x3 squares touching diagonally, and a separate 4x4 square.
        let (w, h) = (20, 12);
        let mut ink = vec![false; w * h];
        let mut fill = |x0: usize, y0: usize, size: usize| {
            for y in y0..y0 + size {
                for x in x0..x0 + size {
                    ink[y * w + x] = true;
                }
            }
        };
        fill(1, 1, 3);
        fill(4, 4, 3);
        fill(12, 5, 4);
        let mut blobs = find_blobs(w, h, &ink, 1000.0);
        blobs.sort_by(|a, b| a.x.total_cmp(&b.x));
        assert_eq!(blobs.len(), 2);
        assert_eq!(blobs[0].area, 18.0);
        assert!((blobs[0].x - 3.5).abs() < 1e-9 && (blobs[0].y - 3.5).abs() < 1e-9);
        assert_eq!(blobs[1].area, 16.0);
        assert!((blobs[1].x - 13.5).abs() < 1e-9 && (blobs[1].y - 6.5).abs() < 1e-9);
    }

    /// A frame of 200 by 160 with a dark square on a light ground and noise of the given
    /// standard deviation (uniform, from a hash of the pixel's index).
    fn noisy_frame(deviation: f32) -> Gray {
        let (w, h) = (200, 160);
        let mut image = Gray::new(w, h);
        for (i, p) in image.data.iter_mut().enumerate() {
            let (x, y) = (i % w, i / w);
            let level = if (60..80).contains(&x) && (50..70).contains(&y) { 40.0 } else { 180.0 };
            let mut z = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let unit = ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32;
            *p = level + deviation * 12f32.sqrt() * (unit - 0.5);
        }
        image
    }

    #[test]
    fn noise_is_measured_where_the_image_is_flat() {
        assert!(noise_level(&noisy_frame(0.0)) < 1e-3);
        for deviation in [5.0, 20.0] {
            let measured = noise_level(&noisy_frame(deviation));
            assert!((measured / deviation - 1.0).abs() < 0.15, "{measured} for {deviation}");
        }
    }

    #[test]
    fn noise_does_not_become_blobs() {
        // The square is the one dark blob, however noisy the frame. (Around it the light
        // ground stands out from a local mean that the square pulls down, so the
        // light-on-dark passes are not empty.)
        for deviation in [0.0, 12.0, 30.0] {
            for (polarity, blobs) in binarise(&noisy_frame(deviation)) {
                if polarity > 0.0 {
                    assert_eq!(blobs.len(), 1, "noise {deviation}");
                    assert!((blobs[0].x - 69.5).abs() < 1.0 && (blobs[0].y - 59.5).abs() < 1.0);
                }
            }
        }
        // With the threshold left where it is for a clean frame, the same noise gives dozens.
        let noisy = noisy_frame(30.0).gaussian_blur(1.0);
        let mean = noisy.box_blur(31);
        let ink: Vec<bool> = noisy.data.iter().zip(&mean.data).map(|(v, m)| v - m < -7.0).collect();
        assert!(find_blobs(noisy.width, noisy.height, &ink, 1024.0).len() > 20);
    }

    #[test]
    fn lines_and_specks_are_dropped() {
        let (w, h) = (40, 10);
        let mut ink = vec![false; w * h];
        for x in 2..38 {
            ink[3 * w + x] = true; // a long thin line
        }
        ink[7 * w + 5] = true; // a speck
        assert!(find_blobs(w, h, &ink, 1000.0).is_empty());
    }
}
