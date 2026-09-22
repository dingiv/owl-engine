//! 模型路径容器(从 xinfer `utils/downloader.rs` 搬运,只取纯数据部分;
//! 下载器本体属 T4 服务层)。

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelPaths {
    pub tokenizer_filename: PathBuf,
    pub tokenizer_config_filename: PathBuf,
    pub config_filename: PathBuf,
    pub generation_config_filename: PathBuf,
    pub filenames: Vec<PathBuf>,
    pub auxiliary_filenames: Vec<PathBuf>,
    pub chat_template_filename: Option<PathBuf>,
}

impl std::fmt::Debug for ModelPaths {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelPaths")
            .field("tokenizer_filename", &self.tokenizer_filename)
            .field("filenames", &self.filenames)
            .finish()
    }
}
