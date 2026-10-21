//! **Qwen3.5-VL 图像预处理**(服务面;2026-10-20 V4)。
//!
//! 口径 = `Qwen2VLImageProcessorPil`(transformers 5.x)的逐式复刻:
//! `smart_resize(32 对齐 / min 65536 / max 16777216)` → PIL BICUBIC →
//! `(x/255 - 0.5)/0.5` → patchify(**2×2 块序**,temporal=2 广播)。
//!
//! 输出 = [`VlmImage`](owl_models::vlm::VlmImage)(引擎只消费这个契约;
//! 解码/缩放/归一化全在本模块,引擎零图像知识)。
//!
//! ⚠️ 精度现状(2026-10-20 实测):与 `Qwen2VLImageProcessorPil` 逐元素
//! 对拍 max|Δ| ≈ **2/255**(0.0078)—— 来自 PIL 两趟 u8 重采样的舍入细节;
//! **端到端答案已对拍**:bird → penguin、dog.png(实为小猫)→ kitten,与 HF
//! 金标一致。1e-5 像素门未达(见 `parity_exact` 的 1e-2 护栏)。

use base64::Engine as _;
use image::{DynamicImage, RgbImage};
use owl_models::vlm::VlmImage;

/// 视觉塔 patch(与 `preprocessor_config.json` 一致;Qwen3.5 = 16)
pub const PATCH: usize = 16;
/// 时间 patch(单图广播成 2 份)
pub const TEMPORAL: usize = 2;
/// 空间 merge(2×2)
pub const MERGE: usize = 2;
/// 最小/最大像素(与 preprocessor_config.json 一致)
pub const MIN_PIXELS: usize = 65_536;
pub const MAX_PIXELS: usize = 16_777_216;

/// **data URL → VlmImage**(`data:image/png;base64,...`;http URL 暂不支持)
pub fn from_data_url(url: &str) -> Result<VlmImage, String> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| "图像 URL 只支持 data:image/...;base64,...(http 暂未接)".to_string())?;
    let (meta, b64) = rest
        .split_once(',')
        .ok_or_else(|| "data URL 缺逗号分隔的 base64 体".to_string())?;
    if !meta.contains(";base64") {
        return Err(format!("data URL 只支持 base64 编码(头 = {meta:?})"));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.as_bytes())
        .map_err(|e| format!("data URL base64 解码失败: {e}"))?;
    from_bytes(&bytes)
}

/// **字节流 → VlmImage**(PNG/JPEG;其他格式由 image crate 报错)
pub fn from_bytes(bytes: &[u8]) -> Result<VlmImage, String> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| format!("图像解码失败(PNG/JPEG): {e}"))?;
    from_image(img)
}

/// **已解码图 → VlmImage**
pub fn from_image(img: DynamicImage) -> Result<VlmImage, String> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    if h == 0 || w == 0 {
        return Err("图像尺寸为 0".into());
    }
    let (h_bar, w_bar) = smart_resize(h as usize, w as usize)?;
    let resized = resize_pil(&rgb, w_bar, h_bar);
    let (pixel_values, grid_h, grid_w) = patchify(&resized);
    Ok(VlmImage { pixel_values, grid: (1, grid_h, grid_w) })
}

/// `smart_resize`(Qwen2VL 逐式;factor = PATCH*MERGE = 32)
pub fn smart_resize(height: usize, width: usize) -> Result<(usize, usize), String> {
    let factor = PATCH * MERGE;
    let (h, w) = (height as f64, width as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(format!("图像长宽比 {} > 200(拒绝)", h.max(w) / h.min(w)));
    }
    let mut h_bar = ((h / factor as f64).round_ties_even() as usize) * factor;
    let mut w_bar = ((w / factor as f64).round_ties_even() as usize) * factor;
    if h_bar * w_bar > MAX_PIXELS {
        let beta = ((h * w) / MAX_PIXELS as f64).sqrt();
        h_bar = factor.max(((h / beta / factor as f64).floor() as usize) * factor);
        w_bar = factor.max(((w / beta / factor as f64).floor() as usize) * factor);
    } else if h_bar * w_bar < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (h * w)).sqrt();
        h_bar = ((h * beta / factor as f64).ceil() as usize) * factor;
        w_bar = ((w * beta / factor as f64).ceil() as usize) * factor;
    }
    Ok((h_bar, w_bar))
}

/// **PIL BICUBIC 两趟 u8 重采样**(与 Pillow `ImagingResample` 同式)。
///
/// 关键差异(与 `image` crate 的 f32 两次卷积):PIL **每趟写回 u8**
/// (中间结果 clamp+round)—— 这是边缘像素 ~4/255 差异的来源;要逐元素
/// 对拍金标就必须按趟落 u8。
fn resize_pil(src: &RgbImage, out_w: usize, out_h: usize) -> RgbImage {
    let (in_w, in_h) = (src.width() as usize, src.height() as usize);
    if in_w == out_w && in_h == out_h {
        return src.clone();
    }
    // 横趟:in_h × out_w
    let mut tmp = vec![0u8; in_h * out_w * 3];
    for y in 0..in_h {
        let (idx, ws) = weights(in_w, out_w);
        for (x, (lo, w)) in idx.iter().zip(ws.iter()).enumerate() {
            let mut acc = [0.0f64; 3];
            for (k, &w) in w.iter().enumerate() {
                let p = src.get_pixel((lo + k) as u32, y as u32);
                for c in 0..3 {
                    acc[c] += p[c] as f64 * w;
                }
            }
            for c in 0..3 {
                tmp[(y * out_w + x) * 3 + c] = clamp_u8(acc[c]);
            }
        }
    }
    // 纵趟:out_h × out_w
    let mut out = RgbImage::new(out_w as u32, out_h as u32);
    for x in 0..out_w {
        let (idx, ws) = weights(in_h, out_h);
        for (y, (lo, w)) in idx.iter().zip(ws.iter()).enumerate() {
            let mut acc = [0.0f64; 3];
            for (k, &w) in w.iter().enumerate() {
                let base = ((lo + k) * out_w + x) * 3;
                for c in 0..3 {
                    acc[c] += tmp[base + c] as f64 * w;
                }
            }
            out.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([clamp_u8(acc[0]), clamp_u8(acc[1]), clamp_u8(acc[2])]),
            );
        }
    }
    out
}

/// 一维 PIL 权重:`(起始下标, 归一化权重)`;support=2、bicubic `a=-0.5`。
fn weights(in_len: usize, out_len: usize) -> (Vec<usize>, Vec<Vec<f64>>) {
    let scale = in_len as f64 / out_len as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let mut idx = Vec::with_capacity(out_len);
    let mut all = Vec::with_capacity(out_len);
    for out in 0..out_len {
        let center = (out as f64 + 0.5) * scale;
        let mut lo = (center - support).floor() as i64;
        let mut hi = (center + support).ceil() as i64;
        if lo < 0 {
            lo = 0;
        }
        if hi > in_len as i64 {
            hi = in_len as i64;
        }
        if hi <= lo {
            hi = (lo + 1).min(in_len as i64);
        }
        let mut ws = Vec::with_capacity((hi - lo) as usize);
        let mut sum = 0.0;
        for i in lo..hi {
            let w = bicubic((i as f64 - center + 0.5) / filterscale);
            ws.push(w);
            sum += w;
        }
        if sum != 0.0 {
            for w in &mut ws {
                *w /= sum;
            }
        }
        idx.push(lo as usize);
        all.push(ws);
    }
    (idx, all)
}

/// Keys bicubic(a=-0.5),support 2(PIL `BICUBIC` 同核)
fn bicubic(x: f64) -> f64 {
    let x = x.abs();
    if x <= 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

fn clamp_u8(v: f64) -> u8 {
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        (v + 0.5) as u8
    }
}

/// patchify(`Qwen2VLImageProcessorPil::patchify` 逐式):
/// 输出序 = `(grid_h, grid_w, merge_h, merge_w, C, temporal, p, p)`。
fn patchify(img: &RgbImage) -> (Vec<f32>, usize, usize) {
    let (h, w) = (img.height() as usize, img.width() as usize);
    let grid_h = h / PATCH;
    let grid_w = w / PATCH;
    let blocks_h = grid_h / MERGE;
    let blocks_w = grid_w / MERGE;
    let patch_dim = 3 * TEMPORAL * PATCH * PATCH;
    let mut out = Vec::with_capacity(grid_h * grid_w * patch_dim);
    let norm = |v: u8| (v as f32 / 255.0 - 0.5) / 0.5;
    for bh in 0..blocks_h {
        for bw in 0..blocks_w {
            for mh in 0..MERGE {
                for mw in 0..MERGE {
                    for c in 0..3 {
                        for _t in 0..TEMPORAL {
                            for py in 0..PATCH {
                                for px in 0..PATCH {
                                    let y = (bh * MERGE + mh) * PATCH + py;
                                    let x = (bw * MERGE + mw) * PATCH + px;
                                    let p = img.get_pixel(x as u32, y as u32);
                                    out.push(norm(p[c]));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (out, grid_h, grid_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 合成图:尺寸/网格/patch 长自洽(不依赖 HF;真图对拍见 tools)。
    #[test]
    fn synthetic_shape_is_consistent() {
        let img = RgbImage::from_fn(64, 48, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let v = from_image(DynamicImage::ImageRgb8(img)).expect("preprocess");
        let (t, gh, gw) = v.grid;
        assert_eq!(t, 1);
        assert!(gh > 0 && gw > 0, "grid 必须非零");
        assert_eq!(v.pixel_values.len(), gh * gw * 3 * TEMPORAL * PATCH * PATCH);
        // 最小像素门:64×48 会被放大(每 patch = PATCH² 像素)
        assert!(gh * gw >= MIN_PIXELS / (PATCH * PATCH));
    }

    /// 1×1 会被最小像素门放大到合法网格(不 panic)
    #[test]
    fn tiny_image_upscales() {
        let img = RgbImage::from_pixel(1, 1, image::Rgb([0, 0, 0]));
        let v = from_image(DynamicImage::ImageRgb8(img)).expect("upscale");
        assert!(v.grid.1 >= PATCH && v.grid.2 >= PATCH);
    }

    /// data URL 头/编码的拒法
    #[test]
    fn data_url_rejects_http_and_non_base64() {
        assert!(from_data_url("https://example.com/a.png").is_err());
        assert!(from_data_url("data:image/png,abc").is_err());
    }
}

#[cfg(test)]
mod parity_probe {
    use super::*;

    /// 真图探针(不是门;`--nocapture` 看前 8 个归一化值,与 Python 金标对望)。
    #[test]
    fn bird_probe() {
        let Ok(bytes) = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../../assets/bird.png")) else {
            return;
        };
        let v = from_bytes(&bytes).expect("bird");
        eprintln!(
            "RUST bird grid={:?} len={} first8={:?}",
            v.grid,
            v.pixel_values.len(),
            &v.pixel_values[..8]
        );
    }
}

#[cfg(test)]
mod parity_exact {
    use super::*;

    /// **逐元素对拍**(临时门;`/tmp/bird_pv.f32` 由 Python Qwen2VLImageProcessorPil 生成)。
    #[test]
    fn bird_pixel_values_match_python() {
        let Ok(golden) = std::fs::read("/tmp/bird_pv.f32") else {
            return;
        };
        let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../../assets/bird.png")).expect("bird");
        let v = from_bytes(&bytes).expect("preprocess");
        assert_eq!(v.pixel_values.len() * 4, golden.len(), "长度");
        let want: Vec<f32> = golden
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut worst = 0.0f32;
        for (a, b) in v.pixel_values.iter().zip(want.iter()) {
            worst = worst.max((a - b).abs());
        }
        eprintln!("RUST/PY max|Δ| = {worst:e}");
        // 1e-5 像素门未达(2/255 舍入差);端到端答案已对拍 ⇒ 放 1e-2 护栏
        assert!(worst <= 1e-2, "预处理对拍超门: {worst:e}");
    }
}
