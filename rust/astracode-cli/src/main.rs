//! `astracode`: make and scan Astracodes from the command line.
//!
//! ```text
//! astracode scan photo.jpg                       # read the codes in an image
//! astracode encode "hello" -o code.png           # a code holding text; it gets as many rings as it needs
//! astracode encode --hex c0ffee -o code.svg      # ... or any bytes
//! astracode astra 4320880580 -o code.svg         # an Astra member number: always one ring
//! astracode batch frames.txt                     # one line of results per image, for test harnesses
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use astracode::astra::Number;
use astracode::format::code;
use astracode::format::svg::{svg, Centre, Style};
use astracode::image::Gray;
use astracode::scan::{scan_gray, scan_rgb8, Code, Options, Payload};
use clap::{Args, Parser, Subcommand};
use rayon::prelude::*;

#[derive(Parser)]
#[command(name = "astracode", version, about = "Make and scan Astracodes")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Read the codes in one or more images.
    Scan {
        images: Vec<PathBuf>,
        #[command(flatten)]
        scanning: Scanning,
        /// Also print how long each scan took.
        #[arg(long)]
        time: bool,
        /// Scan each image this many times and report the fastest. For benchmarking and profiling.
        #[arg(long, default_value_t = 1)]
        repeat: u32,
    },
    /// Scan every image listed in a file (one path per line) and print one
    /// tab-separated line per image: path, seconds, then for each code found
    /// its payload's length in bits, the payload as hex bytes (the last byte
    /// padded with zero bits; `-` if empty), rings, level, mirrored, repaired,
    /// shaped.
    Batch {
        list: PathBuf,
        #[command(flatten)]
        scanning: Scanning,
        #[command(flatten)]
        damage: Damage,
    },
    /// Draw a code holding text or bytes. It gets as many rings as that needs:
    /// one ring holds four bytes.
    Encode {
        /// The text to store, as UTF-8; with --hex, bytes written as hex digits.
        data: String,
        /// Read DATA as hex digits rather than text.
        #[arg(long)]
        hex: bool,
        /// Error-correction level of the rings beyond the first, 0 (least) to 3 (most).
        #[arg(long, default_value_t = 1)]
        level: u8,
        /// Draw this many rings even if fewer would hold the data, 1 to 24.
        #[arg(long)]
        rings: Option<usize>,
        #[command(flatten)]
        drawing: Drawing,
    },
    /// Draw the one-ring code of an Astra member number.
    Astra {
        /// The number: up to ten digits, or a 12-digit member ID.
        number: String,
        /// What the number is, 0 to 26.
        #[arg(long, default_value_t = 0)]
        metadata: u8,
        #[command(flatten)]
        drawing: Drawing,
    },
}

#[derive(Args)]
struct Scanning {
    /// Worker threads to use. Defaults to all cores.
    #[arg(long)]
    threads: Option<usize>,
    /// Look for stars in a copy shrunk to this many pixels on its longer side.
    /// Faster on big frames, but misses small stars.
    #[arg(long)]
    find_size: Option<usize>,
}

/// Ways for `batch` to spoil each image before scanning it, for experiments
/// that sweep one kind of damage over the same clean frames. Any of them makes
/// the scan work on brightness alone, as a camera preview does.
#[derive(Args)]
struct Damage {
    /// Scan the image's brightness only, without falling back to single colour channels.
    #[arg(long)]
    gray: bool,
    /// Swap dark and light: each pixel becomes darkest + lightest - itself, which
    /// turns dark ink on a light background into the same code light on dark.
    #[arg(long)]
    invert: bool,
    /// Add Gaussian noise of this standard deviation, in grey levels of 0 to 255.
    #[arg(long, default_value_t = 0.0)]
    noise: f64,
    /// Seed for the noise. The same seed gives the same noise for the same file name.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

impl Damage {
    fn any(&self) -> bool {
        self.gray || self.invert || self.noise > 0.0
    }

    /// The image's brightness plane, spoiled as asked, as 8-bit pixels.
    fn apply(&self, image: &image::RgbImage, path: &str) -> Vec<u8> {
        let mut gray: Vec<u8> = image
            .as_raw()
            .chunks_exact(3)
            .map(|p| ((p[0] as u32 * 4899 + p[1] as u32 * 9617 + p[2] as u32 * 1868 + 8192) >> 14) as u8)
            .collect();
        if self.invert {
            let (darkest, lightest) = gray.iter().fold((255u8, 0u8), |(lo, hi), &g| (lo.min(g), hi.max(g)));
            for g in &mut gray {
                *g = darkest + lightest - *g;
            }
        }
        if self.noise > 0.0 {
            // One generator per row, seeded from the file's name, the seed and the row, so that rows can
            // be done in parallel and a run can be repeated.
            let name = Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let base = name.bytes().fold(self.seed ^ 0x9E37_79B9_7F4A_7C15, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3));
            let width = image.width() as usize;
            gray.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
                let mut state = base ^ (y as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
                let mut uniform = || {
                    // splitmix64, as a number in (0, 1].
                    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    (((z ^ (z >> 31)) >> 11) as f64 + 1.0) / (1u64 << 53) as f64
                };
                for pair in row.chunks_mut(2) {
                    // Box-Muller: two uniform numbers give two independent normal ones.
                    let (radius, angle) = ((-2.0 * uniform().ln()).sqrt(), std::f64::consts::TAU * uniform());
                    for (g, normal) in pair.iter_mut().zip([radius * angle.cos(), radius * angle.sin()]) {
                        *g = (*g as f64 + self.noise * normal).round().clamp(0.0, 255.0) as u8;
                    }
                }
            });
        }
        gray
    }
}

#[derive(Args)]
struct Drawing {
    /// Where to write the code: `.svg` or `.png`.
    #[arg(short, long)]
    output: PathBuf,
    /// Ink colour.
    #[arg(long, default_value = "#021639")]
    ink: String,
    /// Background colour.
    #[arg(long, default_value = "#febc5c")]
    background: String,
    /// How inked a light limb is, in percent. Lower scans more reliably.
    #[arg(long, default_value_t = 70)]
    light: u32,
    /// Width and height in pixels.
    #[arg(long, default_value_t = 1024)]
    size: u32,
    /// Leave the centre empty instead of drawing the logo.
    #[arg(long)]
    no_logo: bool,
}

fn load(path: &Path) -> Result<image::RgbImage, String> {
    image::open(path).map(|i| i.into_rgb8()).map_err(|e| format!("{}: {e}", path.display()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(digits: &str) -> Result<Vec<u8>, String> {
    let digits: String = digits.chars().filter(|c| !c.is_whitespace()).collect();
    if digits.len() % 2 != 0 || !digits.is_ascii() {
        return Err("hex data must be pairs of hex digits".into());
    }
    (0..digits.len() / 2)
        .map(|i| u8::from_str_radix(&digits[2 * i..2 * i + 2], 16).map_err(|_| "hex data must be pairs of hex digits".to_string()))
        .collect()
}

/// What a payload holds, for a person to read.
fn describe_payload(payload: &Payload) -> String {
    if let Some(number) = Number::from_payload(payload) {
        return format!("Astra number, metadata {}, payload {:010} (member ID {})", number.metadata, number.payload, number.member_id());
    }
    match (payload.as_text(), payload.as_bytes()) {
        (Some(text), _) => format!("{text:?}"),
        (None, Some(bytes)) => format!("{} bytes, hex {}", bytes.len(), hex(bytes)),
        (None, None) => format!("{} bits: {}", payload.len(), payload.bits().map(|b| if b { '1' } else { '0' }).collect::<String>()),
    }
}

fn describe(code: &Code) -> String {
    let size = if code.rings == 1 { "1 ring".to_string() } else { format!("{} rings, level {}", code.rings, code.level) };
    let what = format!("{size}: {}", describe_payload(&code.payload));
    let mut notes = Vec::new();
    if code.mirrored {
        notes.push("mirrored".to_string());
    }
    if code.repaired > 0 {
        notes.push(format!("{} repaired", code.repaired));
    }
    if code.shaped {
        notes.push("bent surface".to_string());
    }
    if notes.is_empty() {
        what
    } else {
        format!("{what} [{}]", notes.join(", "))
    }
}

fn scan(images: &[PathBuf], options: &Options, time: bool, repeat: u32) -> Result<(), String> {
    for path in images {
        let image = load(path)?;
        let (mut codes, mut took) = (Vec::new(), f64::INFINITY);
        for _ in 0..repeat.max(1) {
            let start = Instant::now();
            codes = scan_rgb8(image.width() as usize, image.height() as usize, image.as_raw(), options);
            took = took.min(start.elapsed().as_secs_f64());
        }
        let found = if codes.is_empty() { "no code".to_string() } else { codes.iter().map(describe).collect::<Vec<_>>().join("; ") };
        if time {
            println!("{}: {found} ({took:.3}s)", path.display());
        } else {
            println!("{}: {found}", path.display());
        }
    }
    Ok(())
}

fn batch(list: &Path, options: &Options, damage: &Damage) -> Result<(), String> {
    let text = std::fs::read_to_string(list).map_err(|e| format!("{}: {e}", list.display()))?;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let image = load(Path::new(line))?;
        let (width, height) = (image.width() as usize, image.height() as usize);
        let gray = damage.any().then(|| Gray::from_luma8(width, height, &damage.apply(&image, line)));
        let start = Instant::now();
        let codes = match &gray {
            Some(gray) => scan_gray(gray, options),
            None => scan_rgb8(width, height, image.as_raw(), options),
        };
        let took = start.elapsed().as_secs_f64();
        let mut fields = vec![line.to_string(), format!("{took:.6}")];
        for code in &codes {
            fields.extend([
                code.payload.len().to_string(),
                if code.payload.is_empty() { "-".into() } else { hex(code.payload.padded_bytes()) },
                code.rings.to_string(),
                code.level.to_string(),
                (code.mirrored as u8).to_string(),
                code.repaired.to_string(),
                (code.shaped as u8).to_string(),
            ]);
        }
        println!("{}", fields.join("\t"));
    }
    Ok(())
}

fn draw(rings: &[Vec<u8>], drawing: &Drawing) -> Result<(), String> {
    let style = Style {
        ink: drawing.ink.clone(),
        background: drawing.background.clone(),
        shade: drawing.light as f64 / 100.0,
        size: drawing.size,
        centre: if drawing.no_logo { Centre::Empty } else { Centre::Logo },
    };
    let document = svg(rings, &style);
    let path = &drawing.output;
    match path.extension().and_then(|e| e.to_str()) {
        Some("svg") => std::fs::write(path, document).map_err(|e| format!("{}: {e}", path.display())),
        Some("png") => {
            let tree = resvg::usvg::Tree::from_str(&document, &resvg::usvg::Options::default()).map_err(|e| e.to_string())?;
            let mut pixmap = resvg::tiny_skia::Pixmap::new(drawing.size, drawing.size).ok_or("the image size must not be zero")?;
            resvg::render(&tree, resvg::tiny_skia::Transform::identity(), &mut pixmap.as_mut());
            pixmap.save_png(path).map_err(|e| format!("{}: {e}", path.display()))
        }
        _ => Err(format!("{}: the output must be .svg or .png", path.display())),
    }
}

/// Apply the settings shared by the scanning commands.
fn options(scanning: &Scanning) -> Result<Options, String> {
    if let Some(threads) = scanning.threads {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().map_err(|e| e.to_string())?;
    }
    Ok(Options { find_size: scanning.find_size })
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Scan { images, scanning, time, repeat } => scan(&images, &options(&scanning)?, time, repeat),
        Command::Batch { list, scanning, damage } => batch(&list, &options(&scanning)?, &damage),
        Command::Encode { data, hex, level, rings, drawing } => {
            let bytes = if hex { unhex(&data)? } else { data.into_bytes() };
            let payload = Payload::from_bytes(bytes);
            let symbols = match rings {
                Some(rings) => code::encode_with_rings(&payload, level, rings),
                None => code::encode(&payload, level),
            }
            .map_err(|e| e.to_string())?;
            draw(&symbols, &drawing)?;
            let rings = symbols.len();
            let room = code::capacity(rings, level);
            println!("{} of {room} bytes used in {rings} {}", payload.len() / 8, if rings == 1 { "ring" } else { "rings" });
            Ok(())
        }
        Command::Astra { number, metadata, drawing } => {
            let digits = number.chars().filter(char::is_ascii_digit).count();
            let number = if digits == 12 {
                Number::from_member_id(metadata, &number)
            } else {
                Number::new(
                    metadata,
                    number.trim().parse().map_err(|_| "the number must be up to ten digits, or a 12-digit member ID".to_string())?,
                )
            }
            .map_err(|e| e.to_string())?;
            let symbols = code::encode(&number.to_payload(), 0).map_err(|e| e.to_string())?;
            draw(&symbols, &drawing)?;
            println!("member ID {}", number.member_id());
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("astracode: {message}");
            ExitCode::FAILURE
        }
    }
}
