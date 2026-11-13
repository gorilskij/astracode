//! Drawing a code as SVG.
//!
//! The output is one square, [`half_side`] model units from the centre to each
//! edge, filled with the background colour. Light limbs are drawn first in a
//! colour blended between background and ink, then the solid parts over them,
//! then the centre picture.

use super::geometry::{limb_polygon, limb_solid, star_center, star_polygon, Point};
use super::{half_side, SHADE};

/// What to draw in the middle of the code. Readers ignore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Centre {
    /// The Astra Europa mark.
    #[default]
    Logo,
    /// Nothing.
    Empty,
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
}

impl Default for Style {
    fn default() -> Self {
        Style { ink: "#021639".into(), background: "#febc5c".into(), shade: SHADE, size: 512, centre: Centre::Logo }
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
/// Side of the box the source drawing is scaled into. The mark itself reaches radius 72.9.
const LOGO_D: f64 = 180.0;

/// The colour `t` of the way from `background` to `ink`, both `#rrggbb`.
fn blend(background: &str, ink: &str, t: f64) -> String {
    let channel = |colour: &str, i: usize| u8::from_str_radix(colour.get(i..i + 2).unwrap_or("00"), 16).unwrap_or(0) as f64;
    let mix = |i: usize| (channel(background, i) + (channel(ink, i) - channel(background, i)) * t).round() as u8;
    format!("#{:02x}{:02x}{:02x}", mix(1), mix(3), mix(5))
}

fn polygon(points: &[Point]) -> String {
    let list: Vec<String> = points.iter().map(|p| format!("{:.3},{:.3}", p[0], p[1])).collect();
    format!("<polygon points=\"{}\"/>", list.join(" "))
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
                light.push_str(&polygon(&limb_polygon(k, center)));
            }
            solid.push_str(&polygon(&star_polygon(symbol, center)));
        }
    }
    let picture = match style.centre {
        Centre::Empty => String::new(),
        Centre::Logo => {
            let k = LOGO_D / LOGO_SRC;
            let paths: String = LOGO_PATHS.iter().map(|d| format!("<path d=\"{d}\"/>")).collect();
            format!("<g transform=\"scale({k}) translate({o} {o}) {LOGO_TRANSFORM}\">{paths}</g>", o = -LOGO_SRC / 2.0)
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
        assert_eq!(out.matches("<polygon").count(), 30 + 18 * 5);
        assert!(out.contains("viewBox=\"-175.5 -175.5 351 351\""));
    }
}
