use std::fs;
use std::path::{Path, PathBuf};

use super::example::Example;
use super::field::Field;
use crate::utils::functions::{download_from_url, extract_to_dir};

/// Represents a collection of `Example`s along with field metadata.
///   * holding the data (`examples` and `fields`)
///   * providing helper constructors / length / indexing
///   * exposing a convenience `download` that fetches and extracts archives

#[derive(Clone, Debug)]
pub struct Dataset {
    pub examples: Vec<Example>,
    pub fields: Vec<Field>,
}

impl Dataset {
    /// Create a new dataset from examples & fields metadata.
    pub fn new(examples: Vec<Example>, fields: Vec<Field>) -> Self {
        Self { examples, fields }
    }

    /// Total number of examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Indexing helper (immutable).
    pub fn get(&self, idx: usize) -> Option<&Example> {
        self.examples.get(idx)
    }

    /// Mutable indexing helper.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut Example> {
        self.examples.get_mut(idx)
    }

    /// Replace an example at `idx`.
    pub fn set(&mut self, idx: usize, value: Example) {
        if let Some(slot) = self.examples.get_mut(idx) {
            *slot = value;
        }
    }

    pub fn splits(
        examples: Vec<Example>,
        fields: Vec<Field>,
        train_ratio: f32,
        valid_ratio: f32,
    ) -> (Dataset, Dataset, Dataset) {
        let total = examples.len();
        let train_end = (total as f32 * train_ratio) as usize;
        let valid_end = train_end + (total as f32 * valid_ratio) as usize;

        let train = examples[..train_end].to_vec();
        let valid = examples[train_end..valid_end].to_vec();
        let test = examples[valid_end..].to_vec();

        (
            Dataset::new(train.clone(), fields.clone()),
            Dataset::new(valid.clone(), fields.clone()),
            Dataset::new(test, fields),
        )
    }

    /// Download & extract helper 
    ///
    /// * `urls`: list of remote archives
    /// * `root`: destination directory
    /// * `dirname`: dataset subfolder inside `root`
    /// * `name`: final extracted directory name
    pub fn download(urls: &[&str], root: &str, dirname: &str, name: &str) -> std::io::Result<PathBuf> {
        let path_dirname = Path::new(root).join(dirname);
        let path_name = path_dirname.join(name);

        if !path_dirname.exists() {
            fs::create_dir_all(&path_dirname)?;
            for url in urls {
                let filename = Path::new(url)
                    .file_name()
                    .expect("url has no filename")
                    .to_string_lossy()
                    .to_string();
                let zpath = path_dirname.join(&filename);
                if !zpath.exists() {
                    println!("Download {} from {} to {}", filename, url, zpath.display());
                    // silent unwrap for chunk_size, we use default 1 MiB blocks
                    if let Err(e) = download_from_url(url, &zpath, 1_048_576) {
                        eprintln!("download error: {}", e);
                    }
                }
                if let Err(e) = extract_to_dir(zpath.to_string_lossy(), &path_name) {
                    eprintln!("extract error: {}", e);
                }
            }
        }

        Ok(path_name)
    }
}