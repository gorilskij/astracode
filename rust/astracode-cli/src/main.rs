//! `astracode`: make and scan Astracodes from the command line.
//!
//! ```text
//! astracode scan photo.jpg                       # read the codes in an image
//! astracode number 4320880580 -o code.svg        # a one-ring code
//! astracode message --rings 4 "hello" -o code.png
//! astracode batch frames.txt                     # one line of results per image, for test harnesses
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use astracode::format::svg::{svg, Centre, Style};
use astracode::format::{message, ring0};
use astracode::scan::{scan_rgb8, Code, Options, Payload};
use clap::{Args, Parser, Subcommand};

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
    /// its kind (`number` or `message`), value (decimal, or hex bytes), rings,
    /// level, mirrored, repaired, shaped.
    Batch {
        list: PathBuf,
        #[command(flatten)]
        scanning: Scanning,
    },
    /// Draw a one-ring code.
    Number {
        /// Up to ten digits.
        payload: u64,
        /// What the payload is, 0 to 108.
        #[arg(long, default_value_t = 0)]
        metadata: u64,
        #[command(flatten)]
        drawing: Drawing,
    },
    /// Draw a code with several rings carrying a text message.
    Message {
        text: String,
        /// Number of rings, 2 to 24.
        #[arg(long)]
        rings: usize,
        /// Error-correction level, 0 (least) to 3 (most).
        #[arg(long, default_value_t = 1)]
        level: u8,
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

fn describe(code: &Code) -> String {
    let what = match &code.payload {
        Payload::Number(value) => {
            let (metadata, payload) = ring0::unpack(*value);
            format!("metadata {metadata}, payload {payload:010} (member ID {})", ring0::member_id(payload))
        }
        Payload::Message { level, bytes } => match std::str::from_utf8(bytes) {
            Ok(text) => format!("{} rings, level {level}: {text:?}", code.rings),
            Err(_) => format!("{} rings, level {level}: {} bytes, hex {}", code.rings, bytes.len(), hex(bytes)),
        },
    };
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

fn batch(list: &Path, options: &Options) -> Result<(), String> {
    let text = std::fs::read_to_string(list).map_err(|e| format!("{}: {e}", list.display()))?;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let image = load(Path::new(line))?;
        let start = Instant::now();
        let codes = scan_rgb8(image.width() as usize, image.height() as usize, image.as_raw(), options);
        let took = start.elapsed().as_secs_f64();
        let mut fields = vec![line.to_string(), format!("{took:.6}")];
        for code in &codes {
            let (kind, value, level) = match &code.payload {
                Payload::Number(value) => ("number", value.to_string(), 0),
                Payload::Message { level, bytes } => ("message", if bytes.is_empty() { "-".into() } else { hex(bytes) }, *level),
            };
            fields.extend([
                kind.to_string(),
                value,
                code.rings.to_string(),
                level.to_string(),
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
        Command::Batch { list, scanning } => batch(&list, &options(&scanning)?),
        Command::Number { payload, metadata, drawing } => {
            let value = ring0::pack(metadata, payload).map_err(|e| e.to_string())?;
            let symbols = ring0::encode(value).map_err(|e| e.to_string())?;
            draw(&[symbols.to_vec()], &drawing)?;
            println!("member ID {}", ring0::member_id(payload));
            Ok(())
        }
        Command::Message { text, rings, level, drawing } => {
            let symbols = message::encode(text.as_bytes(), rings, level).map_err(|e| e.to_string())?;
            draw(&symbols, &drawing)?;
            println!("{} of {} bytes used", text.len(), message::capacity(rings, level));
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
