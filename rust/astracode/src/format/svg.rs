//! Drawing a code as SVG.
//!
//! The output is one square, [`half_side`] model units from the centre to each
//! edge, filled with the background colour. Light limbs are drawn first in a
//! colour blended between background and ink, then the solid parts over them,
//! then the centre picture.
//!
//! The stars are drawn in the shape of the star in the Astra Europa mark: a
//! little fuller than a regular pentagram ([`DRAWN_RATIO`]), with the tips
//! slightly rounded ([`TIP_ROUNDING`]). The scanner matches a regular
//! pentagram and reads these just as well; a star's centre, its size and the
//! directions of its limbs are the same in both.

use super::geometry::{limb_solid, star_center, Point};
use super::{half_side, CENTER_R, SHADE, STAR_RO};

/// What to draw in the middle of the code. Readers ignore it.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Centre {
    /// The Astra Europa mark.
    #[default]
    Logo,
    /// Nothing.
    Empty,
    /// Any picture, drawn into a square by an SVG `<image>` (so it can be a PNG,
    /// JPEG or SVG, typically as a `data:` URL), scaled to keep its aspect ratio.
    Picture {
        /// The picture's address: what goes in `href`.
        href: String,
        /// How far from the middle of its square the picture paints, as a fraction of
        /// half the square's side: 1 for a disc that fills the square, about 1.41 for
        /// a picture that fills it to the corners. This is what is sized to
        /// [`Style::centre_size`].
        reach: f64,
    },
}

/// How to draw a code.
#[derive(Debug, Clone, PartialEq)]
pub struct Style {
    /// Ink colour, `#rrggbb`.
    pub ink: String,
    /// Background colour, `#rrggbb`.
    pub background: String,
    /// Ink fraction of a light limb, between 0 and 1. Lower is easier to scan
    /// and more visibly two-tone.
    pub shade: f64,
    /// Width and height of the image in pixels.
    pub size: u32,
    pub centre: Centre,
    /// How far the centre picture reaches from the middle, as a fraction of
    /// [`CENTER_R`], the most it may: 1 at most.
    pub centre_size: f64,
}

impl Default for Style {
    fn default() -> Self {
        Style { ink: "#021639".into(), background: "#febc5c".into(), shade: SHADE, size: 512, centre: Centre::Logo, centre_size: 1.0 }
    }
}

/// The Astra Europa mark: three paths in a 3840-unit square source drawing.
const LOGO_PATHS: [&str; 3] = [
    "M338.37,465.61c23-62.6,49.94-122.56,81.41-172.56,19.82,31.52,37.88,67,54.29,104.62a1242.84,1242.84,0,0,1,128.34-46.05C485.58,134.07,419.78,91.42,419.78,91.42s-78.92,51.16-217.27,327C169,485.21,134,563.55,99.17,654.68,152.15,594.63,228.47,529.46,338.37,465.61Z",
    "M590.17,543.37c-16.66,11.13-34.06,23.38-51.87,36.74,49.12,170.74,70.06,328.47,70.06,328.47H825.5C776.38,741,723.47,603.64,672.6,492.63,646.63,507.34,618.78,524.25,590.17,543.37Z",
    "M982.17,298.05l-98-5.3-38.63-90.22a4,4,0,0,0-7.39.13l-35.32,91.58-97.75,8.86a4,4,0,0,0-2.16,7.07l51.57,41.89a1245.36,1245.36,0,0,0-135.25,31.47,1205.4,1205.4,0,0,0-131.61,46.65A1107.27,1107.27,0,0,0,384.41,480q-34.05,18.71-64,37.47c-149,93.34-225.58,186.86-264.91,259-42.9,78.78-41.47,132.11-41.47,132.11H231.19c58.29-158.17,179.5-278.94,296.43-364.38a1326.91,1326.91,0,0,1,130-83.58c46.2-26.07,86.34-45.26,114.42-57.74l-14.77,64.88a4,4,0,0,0,6.06,4.24l82.4-53.33L930.06,469a4,4,0,0,0,5.9-4.46l-25.25-94.84L984.58,305A4,4,0,0,0,982.17,298.05Z",
];
const LOGO_SRC: f64 = 3840.0;
const LOGO_TRANSFORM: &str = "translate(744.90 729.60) scale(2.671645) translate(-14.05 -91.42)";
/// Side of the box the source drawing is scaled into at full size. The mark itself
/// then reaches [`LOGO_REACH`] from the middle.
const LOGO_D: f64 = 180.0;
const LOGO_REACH: f64 = 72.9;

/// The colour `t` of the way from `background` to `ink`, both `#rrggbb`.
fn blend(background: &str, ink: &str, t: f64) -> String {
    let channel = |colour: &str, i: usize| u8::from_str_radix(colour.get(i..i + 2).unwrap_or("00"), 16).unwrap_or(0) as f64;
    let mix = |i: usize| (channel(background, i) + (channel(ink, i) - channel(background, i)) * t).round() as u8;
    format!("#{:02x}{:02x}{:02x}", mix(1), mix(3), mix(5))
}

/// Inner radius of a star as drawn, as a fraction of its outer radius: that of the
/// star in the mark. (A regular pentagram has 0.382.)
pub const DRAWN_RATIO: f64 = 0.4437;
/// Radius of the arc that rounds each tip, as a fraction of the star's outer radius.
pub const TIP_ROUNDING: f64 = 0.0256;

/// The point `radius` from `center` in the direction `degrees` (clockwise from east, as SVG has it).
fn at(center: Point, radius: f64, degrees: f64) -> Point {
    let a = degrees.to_radians();
    [center[0] + radius * a.cos(), center[1] + radius * a.sin()]
}

/// The notch before limb `k` of a star, going clockwise: where its outline is nearest the centre.
fn notch(center: Point, k: usize) -> Point {
    at(center, STAR_RO * DRAWN_RATIO, -90.0 - 36.0 + 72.0 * k as f64)
}

/// Path commands from the notch before limb `k` round its tip: a line to where the
/// rounding arc leaves the first edge, and the arc to where it meets the second.
fn tip(center: Point, k: usize) -> String {
    let axis = -90.0 + 72.0 * k as f64;
    let point = at(center, STAR_RO, axis);
    let inner = STAR_RO * DRAWN_RATIO;
    // Half the angle between the two edges that meet at the tip, and from it how far
    // along each edge an arc tangent to both leaves it.
    let half = (inner * 36f64.to_radians().sin()).atan2(STAR_RO - inner * 36f64.to_radians().cos());
    let radius = TIP_ROUNDING * STAR_RO;
    let cut = radius / half.tan();
    let leave = |side: f64| {
        let corner = at(center, inner, axis + side);
        let length = (corner[0] - point[0]).hypot(corner[1] - point[1]);
        [point[0] + (corner[0] - point[0]) * cut / length, point[1] + (corner[1] - point[1]) * cut / length]
    };
    let (a, b) = (leave(-36.0), leave(36.0));
    format!("L{:.3},{:.3} A{radius:.3},{radius:.3} 0 0 1 {:.3},{:.3} ", a[0], a[1], b[0], b[1])
}

/// Limb `k` of a star centred on `center`, closed through the star's centre so that
/// a light limb drawn first tucks under the solid core without a seam.
fn limb_path(k: usize, center: Point) -> String {
    let (from, to) = (notch(center, k), notch(center, (k + 1) % 5));
    format!(
        "<path d=\"M{:.3},{:.3} {}L{:.3},{:.3} L{:.3},{:.3} Z\"/>",
        from[0],
        from[1],
        tip(center, k),
        to[0],
        to[1],
        center[0],
        center[1]
    )
}

/// The solid part of a star with this symbol: the centre pentagon plus its solid limbs.
fn solid_path(symbol: u8, center: Point) -> String {
    let mut d = String::new();
    for k in 0..5 {
        let corner = notch(center, k);
        d.push_str(&format!("{}{:.3},{:.3} ", if k == 0 { 'M' } else { 'L' }, corner[0], corner[1]));
        if limb_solid(symbol, k) {
            d.push_str(&tip(center, k));
        }
    }
    format!("<path d=\"{d}Z\"/>")
}

/// A code as an SVG document, from the symbols of each ring as
/// [`super::code::encode`] returns them.
pub fn svg(rings: &[Vec<u8>], style: &Style) -> String {
    let half = half_side(rings.len());
    let (mut light, mut solid) = (String::new(), String::new());
    for (ring, symbols) in rings.iter().enumerate() {
        for (i, &symbol) in symbols.iter().enumerate() {
            let center = star_center(i, ring);
            for k in (0..5).filter(|&k| !limb_solid(symbol, k)) {
                light.push_str(&limb_path(k, center));
            }
            solid.push_str(&solid_path(symbol, center));
        }
    }
    // The radius the centre picture reaches.
    let reach = style.centre_size.clamp(0.0, 1.0) * CENTER_R;
    let picture = match &style.centre {
        Centre::Empty => String::new(),
        Centre::Logo => {
            let k = LOGO_D / LOGO_SRC * reach / LOGO_REACH;
            let paths: String = LOGO_PATHS.iter().map(|d| format!("<path d=\"{d}\"/>")).collect();
            format!("<g transform=\"scale({k}) translate({o} {o}) {LOGO_TRANSFORM}\">{paths}</g>", o = -LOGO_SRC / 2.0)
        }
        Centre::Picture { href, reach: own } => {
            let half = reach / own.max(1e-6);
            let href = href.replace('&', "&amp;").replace('"', "&quot;").replace('<', "&lt;");
            format!(
                "<image x=\"{min}\" y=\"{min}\" width=\"{side}\" height=\"{side}\" preserveAspectRatio=\"xMidYMid meet\" href=\"{href}\"/>",
                min = -half,
                side = 2.0 * half
            )
        }
    };
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{size}\" height=\"{size}\" \
         viewBox=\"{min} {min} {side} {side}\">\
         <rect x=\"{min}\" y=\"{min}\" width=\"{side}\" height=\"{side}\" fill=\"{bg}\"/>\
         <g fill=\"{shade}\">{light}</g><g fill=\"{ink}\">{solid}{picture}</g></svg>",
        size = style.size,
        min = -half,
        side = 2.0 * half,
        bg = style.background,
        shade = blend(&style.background, &style.ink, style.shade),
        ink = style.ink,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colours_blend_between_background_and_ink() {
        assert_eq!(blend("#ffffff", "#000000", 0.0), "#ffffff");
        assert_eq!(blend("#ffffff", "#000000", 1.0), "#000000");
        assert_eq!(blend("#000000", "#ff8000", 0.5), "#804000");
    }

    #[test]
    fn every_star_is_drawn() {
        let symbols = vec![vec![31u8; 12], vec![0u8; 18]];
        let out = svg(&symbols, &Style { centre: Centre::Empty, ..Style::default() });
        // 30 solid parts, and five light limbs for each of the 18 all-light stars.
        assert_eq!(out.matches("<path").count(), 30 + 18 * 5);
        assert!(out.contains("viewBox=\"-175.5 -175.5 351 351\""));
    }

    #[test]
    fn a_centre_picture_is_sized_by_how_far_it_reaches() {
        let style = |centre| Style { centre, centre_size: 0.5, ..Style::default() };
        // A disc that fills its square, at half the most the centre allows: a square of side 73.
        let out = svg(&[vec![31u8; 12]], &style(Centre::Picture { href: "data:image/png;base64,AA==\"x".into(), reach: 1.0 }));
        assert!(out.contains("<image x=\"-36.5\" y=\"-36.5\" width=\"73\" height=\"73\""), "{out}");
        assert!(out.contains("href=\"data:image/png;base64,AA==&quot;x\""));
        // The mark, likewise: half its full size.
        let half = svg(&[vec![31u8; 12]], &style(Centre::Logo));
        let full = svg(&[vec![31u8; 12]], &Style::default());
        let k = |out: &str| -> f64 { out.split("scale(").nth(1).unwrap().split(')').next().unwrap().parse().unwrap() };
        assert!((k(&half) / k(&full) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn stars_have_the_shape_of_the_star_in_the_mark() {
        // Star 0 of ring 0, all limbs solid, is centred on (0, -100) with its first tip straight up.
        let out = svg(&[vec![31u8; 12]], &Style { centre: Centre::Empty, ..Style::default() });
        let first = out.split("<path d=\"").nth(1).expect("a star").split('"').next().expect("its outline");
        // It starts at the notch before that tip: the inner radius, 36 degrees round from the tip.
        let inner = STAR_RO * DRAWN_RATIO;
        let (x, y) = (-inner * 36f64.to_radians().sin(), -100.0 - inner * 36f64.to_radians().cos());
        assert!(first.starts_with(&format!("M{x:.3},{y:.3} ")), "{first}");
        // Five rounded tips, and the arc at the top passes just short of the pentagram's sharp tip.
        assert_eq!(first.matches(" A").count(), 5);
        let arc: Vec<f64> = first.split(" A").nth(1).expect("an arc").split([' ', ',']).filter_map(|v| v.parse().ok()).collect();
        let (radius, end) = (arc[0], [arc[5], arc[6]]);
        assert!((radius - TIP_ROUNDING * STAR_RO).abs() < 1e-3);
        assert!(end[0] > 0.0 && end[0] < 0.5 && end[1] > -100.0 - STAR_RO && end[1] < -100.0 - STAR_RO + 1.5, "{end:?}");
    }
}
