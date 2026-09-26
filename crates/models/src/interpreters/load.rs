//! 装载解释器:Loadable::layout 需求清单 → 数据源取数 → 布局变换 →
//! 物化进设备块(含流式分块与并发流水)。
//!
//! 变体定位:**装载域**解释器(与计算域 [`super::eval`] 互不依赖)。
//! 数据单副本流动:take_range / convert_chunk_into 查到才实体化,
//! pinned 租约 move 进消息,DMA 直读,用毕即弃(§四 24 流式律)。

use crate::contract::{Dtype};
use crate::contract::{DeviceClient, ModelError};
use crate::module::{Layout, LoadEntry, LoadManifest, LoaderOps, WeightSource};
use futures_util::future::join_all;

// ============================================================================
// §1.5 装载解释器:eval_load(识别 LoaderOps 指令;装载域的执行能力)
// ============================================================================

/// 装载并发度(2026-09-26 M-e loader 性能,用户裁决:4 协程流水 ——
/// 每协程一束键组,数据单副本流动:take 取走 → 变换/move 上传 → 即弃,
/// 主机驻留只剩"在途"份)。
const LOAD_WORKERS: usize = 4;

/// 装载执行(层级入口):驱动层的 `Loadable::layout` 需求并物化。
/// 解释器自己调用层钩子并为它传递 LoaderCtx —— 使用者只给层与源。
/// face 提供并发句柄(`DeviceClient::loader_faces`)且 Want 多于一条
/// 时走分桶流水;否则顺序。
///
/// ```rust,ignore
/// eval_load(&mlp, face, &src, LoaderCtx::default()).await?;
/// ```
pub async fn eval_load<M, D, S>(
    layer: &M,
    face: &mut D,
    src: &S,
    ctx: &crate::module::LoaderCtx,
) -> Result<LoadManifest, ModelError>
where
    M: crate::module::Loadable,
    D: DeviceClient,
    S: WeightSource + ?Sized,
{
    let want = layer.layout(ctx); // 层产出需求清单(ctx 引用透传)
    eval_want(&want, face, src).await
}

/// 需求清单求值:并发流水(能力可用)或顺序。
async fn eval_want<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    face: &mut D,
    src: &S,
) -> Result<LoadManifest, ModelError> {
    let handles = face.loader_faces(LOAD_WORKERS);
    let manifest = match handles {
        Some(handles) if handles.len() > 1 && want.wants().len() > 1 => {
            eval_want_parallel(want, handles, src).await?
        }
        _ => eval_want_sequential(want, face, src).await?,
    };
    Ok(manifest)
}

/// 键分组:同键多 Want(tied 双槽 = w 直读 + w_t 转置读)原子成组,
/// take 一次供全组;组序 = 清单首次出现序。
/// 大张量阈值与分块(块 128MB;≥64MB 语义上走多块,判据 = 分块循环)
const CHUNK_BYTES: usize = 128 << 20;
const CHUNK_ELEMS: usize = CHUNK_BYTES / 4;

fn key_groups<'a>(wants: &'a [crate::module::Want]) -> Vec<Vec<&'a crate::module::Want>> {
    let mut order: Vec<&'a str> = Vec::new();
    let mut map: std::collections::HashMap<&'a str, Vec<&'a crate::module::Want>> =
        std::collections::HashMap::new();
    for w in wants {
        if !map.contains_key(w.key.as_str()) {
            order.push(w.key.as_str());
            map.insert(w.key.as_str(), Vec::new());
        }
        map.get_mut(w.key.as_str()).unwrap().push(w);
    }
    order.into_iter().map(|k| map.remove(k).unwrap()).collect()
}

/// 装载取数失败归因:键不在 = 缺键;在但长度不符 = 元素数错误
fn attribution<S: WeightSource + ?Sized>(src: &S, key: &str, n: usize) -> ModelError {
    match src.elem_len(key) {
        Some(l) => ModelError::Msg(format!(
            "Weight '{key}': 元素 {l} != 期望 {n}"
        )),
        None => ModelError::Msg(format!("Weight '{key}': 数据源缺键")),
    }
}

fn want_bytes(w: &crate::module::Want) -> usize {
    w.shape.iter().product::<usize>() * 4
}

/// 单键组装载(数据单副本流动):
/// 1. `src.take(key)` 取走所有权(源驻留下降);
/// 2. Transposed Want 先行(只读 data,产出转置 Vec → htod_f32 move);
/// 3. Direct Want 收尾:htod_f32(data) 直接 move —— 零变换零拷贝
///    (良构校验内联;多 Direct 同键时末位以外克隆,现模型不出现)。
async fn load_group<D: DeviceClient, S: WeightSource + ?Sized>(
    face: &mut D,
    group: &[&crate::module::Want],
    src: &S,
) -> Result<Vec<LoadEntry>, ModelError> {
    let mut manifest = Vec::new();
    for w in group {
        let n: usize = w.shape.iter().product();
        match w.layout {
            crate::module::Layout::Transposed => {
                // 整取 → TILE² 转置 → htod_f32 move(小件容忍 3 拷)
                let data = src
                    .take_range(&w.key, 0, n)
                    .ok_or_else(|| attribution(src, &w.key, n))?;
                let t = transpose_into_vec(data.as_slice(), w, n)?;
                let b = face
                    .htod_f32(&w.shape, t)
                    .await
                    .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?;
                w.sink.deliver(crate::contract::Bytes::new(b.id, n));
                manifest.push(LoadEntry {
                    key: w.key.clone(),
                    shape: w.shape.clone(),
                    layout: Layout::Transposed,
                    block: crate::contract::Bytes::new(b.id, n),
                });
            }
            crate::module::Layout::Direct => {
                // pinned 租约流水(二拷贝预算:转换直达租约,DMA 直读租约;
                // 大张量自动多块,小张量单块 —— 主机在途 = 单块)
                let b = face
                    .alloc(Dtype::F32, n)
                    .await
                    .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?;
                let mut off = 0usize;
                while off < n {
                    let len = CHUNK_ELEMS.min(n - off);
                    let mut lease = face
                        .alloc_pinned(len)
                        .await
                        .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?;
                    src.convert_chunk_into(&w.key, off, len, lease.slice_mut())
                        .ok_or_else(|| attribution(src, &w.key, n))?;
                    face.upload_pinned(lease, &b, off, len)
                        .await
                        .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?;
                    off += len;
                }
                w.sink.deliver(crate::contract::Bytes::new(b.id, n));
                manifest.push(LoadEntry {
                    key: w.key.clone(),
                    shape: w.shape.clone(),
                    layout: Layout::Direct,
                    block: crate::contract::Bytes::new(b.id, n),
                });
            }
        }
    }
    Ok(manifest)
}

/// TILE² 分块转置 → 新 Vec(源 [rows, cols] → 目标 [cols, rows] f32)
fn transpose_into_vec(data: &[f32], w: &crate::module::Want, n: usize) -> Result<Vec<f32>, ModelError> {
    const TILE: usize = 64;
    let (cols, rows) = (w.shape[0], w.shape[1]);
    if data.len() != n {
        return Err(ModelError::Msg(format!(
            "Weight '{}': 元素 {} != 源形状 {rows}×{cols}",
            w.key,
            data.len()
        )));
    }
    let mut out = vec![0f32; n];
    for r0 in (0..rows).step_by(TILE) {
        let r1 = (r0 + TILE).min(rows);
        for c0 in (0..cols).step_by(TILE) {
            let c1 = (c0 + TILE).min(cols);
            for r in r0..r1 {
                for c in c0..c1 {
                    out[c * rows + r] = data[r * cols + c];
                }
            }
        }
    }
    Ok(out)
}

/// 顺序路径(CpuFace / 单 Want / 无并发能力)
async fn eval_want_sequential<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    face: &mut D,
    src: &S,
) -> Result<LoadManifest, ModelError> {
    let mut manifest = LoadManifest::default();
    for group in key_groups(want.wants()) {
        manifest.0.extend(load_group(face, &group, src).await?);
    }
    Ok(manifest)
}

/// 并发流水(GpuClient 等提供多句柄的 face):
/// 键组按字节 LPT 分给最轻协程(embed 双槽 2GB 独占一束);
/// join_all 单线程协作驱动 —— 组 A await 设备拷贝期间,组 B 的
/// take/变换在同一线程推进,与 server 线程 memcpy 流水重叠。
/// 交付序无关(Want 各带各的 sink;分配无跨 Want 定序)。
async fn eval_want_parallel<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    handles: Vec<D>,
    src: &S,
) -> Result<LoadManifest, ModelError> {
    let mut groups = key_groups(want.wants());
    groups.sort_by_key(|g| {
        std::cmp::Reverse(g.iter().map(|w| want_bytes(w)).sum::<usize>())
    });
    let k = handles.len().min(groups.len()).max(1);
    let mut buckets: Vec<(D, Vec<Vec<&crate::module::Want>>, usize)> = handles
        .into_iter()
        .take(k)
        .map(|f| (f, Vec::new(), 0usize))
        .collect();
    for g in groups {
        let bytes: usize = g.iter().map(|w| want_bytes(w)).sum();
        let b = buckets.iter_mut().min_by_key(|(_, _, total)| *total).unwrap();
        b.1.push(g);
        b.2 += bytes;
    }
    let futs = buckets.into_iter().map(|(mut face, bundles, _)| async move {
        let mut entries = Vec::new();
        for bundle in &bundles {
            entries.extend(load_group(&mut face, bundle, src).await?);
        }
        Ok::<_, ModelError>(entries)
    });
    let results = join_all(futs).await;
    let all = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    Ok(all.into_iter().flatten().collect())
}

