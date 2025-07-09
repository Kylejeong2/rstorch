use std::fs;
use std::path::{Path, PathBuf};

use super::example::Example;
use super::field::Field;
use crate::utils::functions::{download_from_url, extract_to_dir};

/// Represents a collection of `Example`s along with field metadata.
///
/// This is a simplified Rust port of the Python `Dataset` base class displayed
/// in the comments. It focuses on:
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

    /// Rough equivalent of Python `@classmethod splits(...)` returning train /
    /// test / valid splits according to provided ratios. Currently performs a
    /// naive sequential split; in real code you'd want to shuffle first.
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

    /// Download & extract helper similar to the Python `download` classmethod.
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

// class Dataset(ABC):
//     r"""
//     Abstract Dataset class. All dataset for machine learning purposes can inherits from this architecture,
//     for convenience.
//     """
//     urls = []
//     name = ''
//     dirname = ''

//     def __init__(self, examples, fields):
//         self.examples = examples
//         self.fields = fields

//     @classmethod
//     def splits(cls, train=None, test=None, valid=None, root='.'):
//         raise NotImplementedError

//     @classmethod
//     def download(cls, root):
//         r"""Download and unzip a web archive (.zip, .gz, or .tgz).

//         Args:
//             root (str): Folder to download data to.

//         Returns:
//             string: Path to extracted dataset.
//         """
//         path_dirname = os.path.join(root, cls.dirname)
//         path_name = os.path.join(path_dirname, cls.name)
//         if not os.path.isdir(path_dirname):
//             for url in cls.urls:
//                 filename = os.path.basename(url)
//                 zpath = os.path.join(path_dirname, filename)
//                 if not os.path.isfile(zpath):
//                     if not os.path.exists(os.path.dirname(zpath)):
//                         os.makedirs(os.path.dirname(zpath))
//                     print(f'Download {filename} from {url} to {zpath}')
//                     download_from_url(url, zpath)
//                 extract_to_dir(zpath, path_name)

//         return path_name

//     def __repr__(self):
//         name = self.__class__.__name__
//         string = f"Dataset {name}("
//         tab = "   "
//         for (key, value) in self.__dict__.items():
//             if key[0] != "_":
//                 if isinstance(value, Example):
//                     fields = self.fields
//                     for (name, field) in fields:
//                         string += f"\n{tab}({name}): {field.__class__.__name__}" \
//                                   f"(transform={True if field.transform is not None else None}, dtype={field.dtype})"
//                 elif isinstance(value, norch.Tensor):
//                     string += f"\n{tab}({key}): {value.__class__.__name__}(shape={value.shape}, dtype={value.dtype})"
//                 else:
//                     string += f"\n{tab}({key}): {value.__class__.__name__}"
//         return f'{string}\n)'

//     def __getitem__(self, item):
//         return self.examples[item]

//     def __setitem__(self, key, value):
//         self.examples[key] = value

//     def __len__(self):
//         return len(self.examples)