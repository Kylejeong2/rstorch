use super::example::Example;
use std::fmt;

/// Metadata about a single field of an `Example` (similar to torchtext `Field`).
/// Currently only stores a name and dtype string.
pub struct Field {
    pub name: String,
    pub dtype: String,
    // Optionally a transform function applied to the field.
    pub transform: Option<Box<dyn Fn(&mut Example)>>,
}

impl fmt::Debug for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Field")
            .field("name", &self.name)
            .field("dtype", &self.dtype)
            .field("transform", &self.transform.is_some())
            .finish()
    }
}

impl Clone for Field {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            dtype: self.dtype.clone(),
            transform: None, // Cannot clone function pointers
        }
    }
} 