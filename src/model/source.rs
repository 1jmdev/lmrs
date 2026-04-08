use std::path::PathBuf;

#[derive(Clone, Debug)]
pub enum ModelSource {
    LocalPath(PathBuf),
    HuggingFace {
        repo: String,
        revision: Option<String>,
        file: Option<String>,
    },
}

impl ModelSource {
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::LocalPath(path.into())
    }

    pub fn hugging_face(repo: impl Into<String>) -> Self {
        Self::HuggingFace {
            repo: repo.into(),
            revision: None,
            file: None,
        }
    }
}
