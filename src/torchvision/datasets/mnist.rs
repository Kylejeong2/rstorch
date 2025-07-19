use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use ndarray::{Array1, Array2, Array3, s};
use flate2::read::GzDecoder;
use crate::utils::data::{Dataset, Example, Field};

/// MNIST dataset loader for handwritten digit recognition.
/// 
/// Loads training, validation, and test partitions of the MNIST dataset
/// (http://yann.lecun.com/exdb/mnist/). If the data is not already contained 
/// in the specified directory, it will try to download it.
///
/// This dataset contains 60000 training examples, and 10000 test examples of 
/// handwritten digits in {0, ..., 9} and corresponding labels. Each handwritten 
/// image has an "original" dimension of 28x28x1, and is stored row-wise as a 
/// string of 784x1 bytes. Pixel values are in range 0 to 255 (inclusive).
#[derive(Clone, Debug)]
pub struct MNIST {
    pub data: Array3<u8>,
    pub labels: Array1<u8>,
    pub transform: Option<Box<dyn Fn(&Array2<u8>) -> Array2<f32>>>,
    pub target_transform: Option<Box<dyn Fn(u8) -> u8>>,
}

impl MNIST {
    const URLS: &'static [&'static str] = &[
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", 
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
    ];
    const NAME: &'static str = "mnist-data-rs";
    const DIRNAME: &'static str = "mnist";

    /// Create a new MNIST dataset from data and label file paths.
    ///
    /// # Arguments
    /// * `path_data` - Path to the images file (e.g., train-images-idx3-ubyte)
    /// * `path_label` - Path to the labels file (e.g., train-labels-idx1-ubyte)
    /// * `transform` - Optional transform function for images
    /// * `target_transform` - Optional transform function for labels
    pub fn new<P: AsRef<Path>>(
        path_data: P,
        path_label: P,
        transform: Option<Box<dyn Fn(&Array2<u8>) -> Array2<f32>>>,
        target_transform: Option<Box<dyn Fn(u8) -> u8>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let data = Self::load_mnist_images(path_data, 16)?;
        let labels = Self::load_mnist_labels(path_label, 8)?;
        
        Ok(MNIST {
            data,
            labels,
            transform,
            target_transform,
        })
    }

    /// Load MNIST images from IDX file format (supports both gzipped and uncompressed).
    fn load_mnist_images<P: AsRef<Path>>(
        path: P,
        header_size: usize,
    ) -> Result<Array3<u8>, Box<dyn std::error::Error>> {
        let file = File::open(&path)?;
        let mut buffer = Vec::new();
        
        // Check if file is gzipped by extension
        let path_str = path.as_ref().to_string_lossy();
        if path_str.ends_with(".gz") {
            let mut decoder = GzDecoder::new(file);
            decoder.read_to_end(&mut buffer)?;
        } else {
            let mut reader = BufReader::new(file);
            reader.read_to_end(&mut buffer)?;
        }
        
        if buffer.len() < header_size {
            return Err("File too small to contain valid header".into());
        }
        
        let data = &buffer[header_size..];
        let num_images = (data.len() / (28 * 28)) as usize;
        
        let array = Array1::from_vec(data.to_vec());
        let reshaped = array.into_shape((num_images, 28, 28))?;
        
        Ok(reshaped)
    }

    /// Load MNIST labels from IDX file format (supports both gzipped and uncompressed).
    fn load_mnist_labels<P: AsRef<Path>>(
        path: P,
        header_size: usize,
    ) -> Result<Array1<u8>, Box<dyn std::error::Error>> {
        let file = File::open(&path)?;
        let mut buffer = Vec::new();
        
        // Check if file is gzipped by extension
        let path_str = path.as_ref().to_string_lossy();
        if path_str.ends_with(".gz") {
            let mut decoder = GzDecoder::new(file);
            decoder.read_to_end(&mut buffer)?;
        } else {
            let mut reader = BufReader::new(file);
            reader.read_to_end(&mut buffer)?;
        }
        
        if buffer.len() < header_size {
            return Err("File too small to contain valid header".into());
        }
        
        let data = &buffer[header_size..];
        Ok(Array1::from_vec(data.to_vec()))
    }

    /// Create training and test splits of the MNIST dataset with default filenames.
    ///
    /// # Arguments
    /// * `root` - Root directory for dataset storage
    ///
    /// # Returns
    /// A tuple containing (train_dataset, test_dataset)
    pub fn splits(root: &str) -> Result<(Self, Self), Box<dyn std::error::Error>> {
        Self::splits_with_files(
            root,
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        )
    }

    /// Create training and test splits of the MNIST dataset with custom filenames.
    ///
    /// # Arguments
    /// * `root` - Root directory for dataset storage
    /// * `train_data` - Filename for training images
    /// * `train_label` - Filename for training labels
    /// * `test_data` - Filename for test images
    /// * `test_label` - Filename for test labels
    ///
    /// # Returns
    /// A tuple containing (train_dataset, test_dataset)
    pub fn splits_with_files(
        root: &str,
        train_data: &str,
        train_label: &str,
        test_data: &str,
        test_label: &str,
    ) -> Result<(Self, Self), Box<dyn std::error::Error>> {
        let path = Path::new(root).join(Self::DIRNAME).join(Self::NAME);
        
        if !path.exists() {
            Dataset::download(Self::URLS, root, Self::DIRNAME, Self::NAME)?;
        }
        
        let train_data_path = path.join(train_data);
        let train_label_path = path.join(train_label);
        let test_data_path = path.join(test_data);
        let test_label_path = path.join(test_label);
        
        let train_dataset = MNIST::new(train_data_path, train_label_path, None, None)?;
        let test_dataset = MNIST::new(test_data_path, test_label_path, None, None)?;
        
        Ok((train_dataset, test_dataset))
    }

    /// Get an item from the dataset.
    ///
    /// # Arguments
    /// * `index` - Index of the item to retrieve
    ///
    /// # Returns
    /// A tuple containing (image_data, label)
    pub fn get_item(&self, index: usize) -> Option<(Array2<f32>, u8)> {
        if index >= self.len() {
            return None;
        }
        
        let mut image = self.data.slice(s![index, .., ..]).to_owned();
        let mut label = self.labels[index];
        
        let processed_image = if let Some(ref transform) = self.transform {
            transform(&image)
        } else {
            image.mapv(|x| x as f32 / 255.0)
        };
        
        if let Some(ref target_transform) = self.target_transform {
            label = target_transform(label);
        }
        
        Some((processed_image, label))
    }

    /// Get the length of the dataset.
    pub fn len(&self) -> usize {
        self.data.len_of(ndarray::Axis(0))
    }

    /// Check if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}