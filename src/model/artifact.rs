use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct ModelArtifact {
    pub gguf_path: PathBuf,
    pub source_label: String,
    pub cache_hit: bool,
}
