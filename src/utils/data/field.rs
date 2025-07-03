use super::example::Example;

/// Metadata about a single field of an `Example` (similar to torchtext `Field`).
/// Currently only stores a name and dtype string.
#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub dtype: String,
    // Optionally a transform function applied to the field.
    pub transform: Option<Box<dyn Fn(&mut Example)>>,
} 