use reqwest::blocking::Client;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use flate2::read::GzDecoder;
use tar::Archive;
use zip::ZipArchive;

/// Recursively generate a JSON value containing random numbers in the range [-1, 1]
/// following the provided `shape` (e.g. `[2, 3]` => `[[.. 3 numbers ..], [.. 3 numbers ..]]`).
///
/// An empty shape returns an empty array.
/// that returns nested lists of random `float`s.
pub fn generate_random_list(shape: &[usize]) -> serde_json::Value {
    fn build(shape: &[usize], rng: &mut impl rand::Rng) -> serde_json::Value {
        if shape.is_empty() {
            return serde_json::Value::Array(vec![]);
        }

        if shape.len() == 1 {
            let mut arr = Vec::with_capacity(shape[0]);
            for _ in 0..shape[0] {
                arr.push(serde_json::json!(rng.gen_range(-1.0..1.0)));
            }
            serde_json::Value::Array(arr)
        } else {
            let mut outer = Vec::with_capacity(shape[0]);
            for _ in 0..shape[0] {
                outer.push(build(&shape[1..], rng));
            }
            serde_json::Value::Array(outer)
        }
    }

    let mut rng = rand::thread_rng();
    build(shape, &mut rng)
}

/// Download a remote file at `url` and write it to `save_path`.
/// A progress bar is displayed in the terminal (simple percentage-based).
pub fn download_from_url(url: &str, save_path: impl AsRef<Path>, chunk_size: usize) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    let mut resp = client.get(url).send()?;
    let total_len_opt = resp.content_length();

    let mut file = File::create(&save_path)?;
    let mut downloaded: u64 = 0;

    let mut buffer = vec![0u8; chunk_size.max(1024 * 1024)];

    loop {
        let n = resp.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])?;
        downloaded += n as u64;
        if let Some(total_len) = total_len_opt {
            progress_bar(downloaded, total_len, Some("Downloading..."), None, None);
        }
    }
    Ok(())
}

/// Extract a compressed archive (tar, tar.gz, zip) into `dirpath`.
/// `.gz` files that are *not* tar archives are simply moved to the target directory.
pub fn extract_to_dir(filename: impl AsRef<Path>, dirpath: impl AsRef<Path>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let filename = filename.as_ref();
    let dirpath = dirpath.as_ref();

    println!("{}", dirpath.display());
    println!("Extracting...");

    let lower_name = filename.to_string_lossy().to_ascii_lowercase();

    if lower_name.ends_with(".tar.gz") || lower_name.ends_with(".tgz") {
        let tar_gz = File::open(filename)?;
        let decompressor = GzDecoder::new(tar_gz);
        let mut archive = Archive::new(decompressor);
        archive.unpack(dirpath)?;
    } else if lower_name.ends_with(".tar") {
        let tar = File::open(filename)?;
        let mut archive = Archive::new(tar);
        archive.unpack(dirpath)?;
    } else if lower_name.ends_with(".zip") {
        let zip_file = File::open(filename)?;
        let mut archive = ZipArchive::new(zip_file)?;
        archive.extract(dirpath)?;
    } else if lower_name.ends_with(".gz") {
        if !dirpath.exists() {
            fs::create_dir_all(dirpath)?;
        }
        let dest = dirpath.join(filename.file_name().unwrap());
        fs::rename(filename, &dest)?;
        println!(" | NOTE: gzip files are not extracted, moved to {}", dest.display());
    }

    println!(" | Done !");
    Ok(dirpath.canonicalize()?)
}

/// Display a rudimentary progress bar similar to the Python version.
/// `current_index`/`max_index` are expected to be in bytes for downloads.
pub fn progress_bar(current_index: u64, max_index: u64, prefix: Option<&str>, suffix: Option<&str>, start_time: Option<Instant>) {
    let percentage = if max_index == 0 { 0 } else { (current_index * 100 / max_index) as u64 };
    let filled = (percentage / 2) as usize;
    let loading_bar = format!("[{}{}]", "=".repeat(filled), " ".repeat(50 - filled));

    let mut display = format!("\r{}{:3}% | {}", prefix.unwrap_or(""), percentage, loading_bar);

    if let Some(suf) = suffix {
        display.push_str(&format!(" | {}", suf));
    }

    if let Some(start) = start_time {
        let elapsed = start.elapsed();
        let mins = elapsed.as_secs() / 60;
        let secs = elapsed.as_secs() % 60;
        display.push_str(&format!(" | Time: {}m {}s", mins, secs));
    }

    if current_index < max_index {
        print!("{}", display);
    } else {
        println!("{} | Done !", display);
    }
    // Make sure stdout flushes immediately.
    let _ = std::io::stdout().flush();
}

/// Helper returning elapsed minutes / seconds given two instants.
pub fn get_time(start_time: Instant, end_time: Instant) -> (u64, u64) {
    let elapsed = end_time.duration_since(start_time);
    (elapsed.as_secs() / 60, elapsed.as_secs() % 60)
}
