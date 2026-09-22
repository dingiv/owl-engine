//! 服务层(T4 大件;本波只收编 runner 协议触点所需的最小类型)。

/// xinfer `server::EmbeddingStrategy` 原样搬运(MessageType::RunEmbed 载荷)。
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingStrategy {
    Mean,
    Last,
}

impl Default for EmbeddingStrategy {
    fn default() -> Self {
        Self::Mean
    }
}
