//! **图像输入面**（服务面；2026-11-16 业务下沉 P5）。
//!
//! 预处理（`smart_resize` / PIL BICUBIC / patchify / 归一化）已搬到
//! [`owl_models::vlm::preprocess`] —— 那是**视觉塔契约的产地**
//! （`preprocessor_config.json` 的语义），不是 HTTP 应用的事。
//!
//! 服务面只剩两件**输入侧**的事：
//! 1. `data:image/...;base64,...` 解码（协议面：OpenAI `image_url` 的载荷形态）；
//! 2. **内容缓存**（同一张图在 GUI agent 多轮请求里反复重发；预处理实测
//!    ~430ms/张，不缓存会直接进 TTFT）。缓存按 LRU 保留最近 8 张（≈100MB host）。
//!
//! 缓存键 = `(data URL, max_pixels)`：同一张图的 base64 逐字节相同才有意义，
//! 这正是客户端重发的形态。

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

use base64::Engine as _;
use owl_models::vlm::VlmImage;

/// **data URL → VlmImage**（`data:image/png;base64,...`；http URL 暂不支持）
pub fn from_data_url(url: &str, max_pixels: usize) -> Result<VlmImage, String> {
    let key = cache_key(url, max_pixels);
    {
        let mut c = cache()
            .lock()
            .map_err(|_| "图像缓存锁不可用".to_string())?;
        c.tick += 1;
        let tick = c.tick;
        if let Some(e) = c.map.get_mut(&key) {
            e.tick = tick;
            return Ok(e.image.clone());
        }
    }
    let bytes = decode_data_url(url)?;
    let image = owl_models::vlm::preprocess::from_bytes(&bytes, max_pixels)?;
    {
        let mut c = cache()
            .lock()
            .map_err(|_| "图像缓存锁不可用".to_string())?;
        c.tick += 1;
        let tick = c.tick;
        if c.map.len() >= IMAGE_CACHE_CAP {
            if let Some(old) = c
                .map
                .iter()
                .min_by_key(|(_, e)| e.tick)
                .map(|(k, _)| *k)
            {
                c.map.remove(&old);
            }
        }
        c.map.insert(
            key,
            CachedImage {
                image: image.clone(),
                tick,
            },
        );
    }
    Ok(image)
}

/// `data:image/...;base64,...` → 原始字节（协议载荷解析；不做任何图像处理）
fn decode_data_url(url: &str) -> Result<Vec<u8>, String> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| "图像 URL 只支持 data:image/...;base64,...(http 暂未接)".to_string())?;
    let (meta, b64) = rest
        .split_once(',')
        .ok_or_else(|| "data URL 缺逗号分隔的 base64 体".to_string())?;
    if !meta.contains(";base64") {
        return Err(format!("data URL 只支持 base64 编码(头 = {meta:?})"));
    }
    base64::engine::general_purpose::STANDARD
        .decode(b64.as_bytes())
        .map_err(|e| format!("data URL base64 解码失败: {e}"))
}

const IMAGE_CACHE_CAP: usize = 8;

struct CachedImage {
    image: VlmImage,
    tick: u64,
}

#[derive(Default)]
struct ImageCache {
    map: HashMap<u64, CachedImage>,
    tick: u64,
}

fn cache() -> &'static Mutex<ImageCache> {
    static CACHE: OnceLock<Mutex<ImageCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(ImageCache::default()))
}

fn cache_key(url: &str, max_pixels: usize) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    url.hash(&mut h);
    max_pixels.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// data URL 头/编码的拒法（协议面；预处理已搬到 models）
    #[test]
    fn data_url_rejects_http_and_non_base64() {
        assert!(from_data_url("https://example.com/a.png", 0).is_err());
        assert!(from_data_url("data:image/png,abc", 0).is_err());
    }

    /// 坏 base64 响亮拒（不静默当空图）
    #[test]
    fn bad_base64_is_rejected() {
        assert!(from_data_url("data:image/png;base64,!!!!", 0).is_err());
    }
}
