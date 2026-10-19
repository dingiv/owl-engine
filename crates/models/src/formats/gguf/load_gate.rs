//! **整模型装载门**(2026-10-19 刀 7c;设计 §六 I-3 / I-5,§七 刀 7)。
//!
//! 判据不是"没报错",而是**字节账**:
//!
//! | # | 判据 | 挡住什么 |
//! |---|---|---|
//! | ① | 每个量化键的槽 dtype = `U8` 且 `shape.product() == GGUF 文件 nbytes` | 悄悄解量化成 f16(显存 ×3.5)、块布局被重排(I-5) |
//! | ② | 原生键槽 dtype = 目标 f16(恰半字节) | cast 臂走错 / 精度口径漂移 |
//! | ③ | 设备总账 ≤ 文件大小 | "27B f16 = 54G 装不下"的算术前提(I-3) |
//! | ④ | 装载条目数 == 声明数 | 静默跳过某个键(最阴的一类) |
//!
//! 为什么必须真上卡:`eval_load` 的形状/分片/物化三条路都在设备面收口,
//! 只在 host 验 `Model::layout` 抓不到"槽声明对了但装错"。

#[cfg(test)]
mod tests {
    use crate::contract::Dtype;
    use crate::formats::gguf::{spec_from_gguf, Gguf, GgufConvention, GgufPlan, GgufSource};
    use crate::formats::load::{load_from_dir, LoadOpts};
    use crate::model::Model;
    use crate::module::{Loadable, LoaderCtx, PlanSource, PlanTable, SourceFacts};
    use owl_macros::owl_test;
    use owl_shared::testkit::{resource, resource_ok};
    use std::path::PathBuf;

    /// 4B 全模型真装载(GPU;`GGUF` 起点 + `Keyed` 计划表 + tied 量化表)
    #[owl_test(gpu)]
    async fn gpu_load_whole_4b_is_compressed() {
        if !resource_ok("gguf:4b") {
            eprintln!("skip: 资源清单缺 \"gguf:4b\"");
            return;
        }
        let path = PathBuf::from(resource("gguf:4b").expect("resource_ok 已过"));
        let g = Gguf::open(&path).expect("open 4B");
        let (spec, _rep) = spec_from_gguf(&g).expect("spec_from_gguf");
        let keys = GgufConvention::new();
        let plan = GgufPlan::new(&g);
        let src = PlanSource::keyed(&plan, &keys, SourceFacts::GGUF);
        let model = Model::new(&spec, GgufConvention::new(), src, src).expect("Model::new(键面全覆盖)");
        assert!(plan.take_unresolved().is_empty(), "计划表仍有未解析键");

        // 声明数(装载前的事实面)
        let ctx = LoaderCtx { dtype: Dtype::F16, ..Default::default() };
        let n_want = model.layout(&ctx).wants().len();

        let mut gpu = crate::testkit::gpu_client().await;
        let opts = LoadOpts::plain(Dtype::F16);
        let manifest = load_from_dir(&path, &mut gpu, &model, opts)
            .await
            .unwrap_or_else(|e| panic!("load_from_dir(4B GGUF): {e}"));
        let entries = manifest.entries();

        // ④ 条目数 == 声明数(每个 want 都落地)
        assert_eq!(entries.len(), n_want, "装载条目数 ≠ 声明数(有键被静默跳过?)");

        // ① ② 逐键字节账
        let mut quant_bytes = 0usize;
        let mut native_bytes = 0usize;
        let mut quant_n = 0usize;
        let mut native_n = 0usize;
        let mut perm_n = 0usize;
        // 期望的列序表(与源同源:同一公式单一产地)
        let exp_perm = crate::formats::gguf::transform::vhead_q8_blocks(16, 32, 128).expect("列序表");
        for e in entries {
            // **虚拟键**(源生成,文件里没有):`blk.N.ssm_out.perm` —— 列序表
            // (A2 落点的伴随件;内容与 `vhead_q8_blocks` 逐字相同 = 端到端验证
            // “源合成 → 上卡”这一路没走歪)
            if e.key.ends_with(".ssm_out.perm") {
                use owl_iface::contract::DeviceClient as _;
                assert_eq!(e.dtype, Dtype::U32, "`{}` 列序表应为 U32 槽", e.key);
                assert_eq!(e.shape, vec![exp_perm.len()], "`{}` 表长 ≠ k/32", e.key);
                let mut buf = vec![0u8; exp_perm.len() * 4];
                gpu.dtoh(&e.block, &mut buf).await.expect("dtoh 列序表");
                let got: Vec<u32> = buf
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                assert_eq!(
                    got, exp_perm,
                    "`{}` 表内容 ≠ `vhead_q8_blocks`(源合成 → 上卡 这一路走歪了?)",
                    e.key
                );
                perm_n += 1;
                continue;
            }
            let Some(ti) = g.get_tensor(&e.key) else {
                panic!("装载条目 `{}` 不在 GGUF 索引里(键算错了?)", e.key);
            };
            let n: usize = e.shape.iter().product();
            let elem_bytes = match e.dtype {
                Dtype::U8 => 1usize,
                Dtype::F16 | Dtype::BF16 => 2,
                Dtype::F32 => 4,
                other => panic!("`{}` 槽 dtype {other:?} 不在 GGUF 装载面的词汇里", e.key),
            };
            let got = n * elem_bytes;
            if ti.ty.native_dtype().is_none() {
                // ① 量化键:**压缩驻留**(零解量化)
                assert_eq!(
                    e.dtype,
                    Dtype::U8,
                    "`{}` 是 {} 量化块却装成 {:?}(应为 U8 块槽 —— I-3)",
                    e.key,
                    ti.ty.name(),
                    e.dtype
                );
                assert_eq!(
                    got, ti.nbytes as usize,
                    "`{}`({}) 槽字节 {got} ≠ 文件 nbytes {}(I-5 块布局零重排)",
                    e.key,
                    ti.ty.name(),
                    ti.nbytes
                );
                quant_bytes += got;
                quant_n += 1;
            } else {
                // ② 原生键:f16 直读(恰半字节)
                assert_eq!(e.dtype, Dtype::F16, "`{}` 原生键应为 f16 槽", e.key);
                assert_eq!(
                    n,
                    ti.elems() as usize,
                    "`{}` 元素数 {} ≠ 文件 {}(形状翻转律失守)",
                    e.key,
                    n,
                    ti.elems()
                );
                native_bytes += got;
                native_n += 1;
            }
        }

        // ③ 总账:压缩驻留 ⇒ 设备字节**低于**文件大小(原生 F32 转 f16 减半)
        let file_bytes = std::fs::metadata(&path).expect("stat").len() as usize;
        let total = quant_bytes + native_bytes;
        assert!(perm_n > 0, "一个 `ssm_out.perm` 虚拟键都没装载 —— A2 列序表没接线?");
        eprintln!(
            "[gguf-load] 4B 装载绿:{} 条(量化 {quant_n} = {:.2} GiB / 原生 {native_n} = {:.2} GiB / 列序表 {perm_n} 张 = {:.1} KiB)\n\
             [gguf-load] 设备总账 {:.2} GiB vs 文件 {:.2} GiB(比 {:.3})",
            entries.len(),
            quant_bytes as f64 / (1u64 << 30) as f64,
            native_bytes as f64 / (1u64 << 30) as f64,
            (perm_n * exp_perm.len() * 4) as f64 / 1024.0,
            (total + perm_n * exp_perm.len() * 4) as f64 / (1u64 << 30) as f64,
            file_bytes as f64 / (1u64 << 30) as f64,
            (total + perm_n * exp_perm.len() * 4) as f64 / file_bytes as f64
        );
        let total = total + perm_n * exp_perm.len() * 4;
        assert!(
            total <= file_bytes,
            "设备总账 {total}B 超过文件 {file_bytes}B —— 量化权重被解量化驻留了?(I-3)"
        );
        // 量化张量必须占绝对多数字节(否则说明大量量化键走了原生臂)
        assert!(
            quant_bytes > native_bytes,
            "量化字节 {quant_bytes} 未占多数(原生 {native_bytes})—— 计划表大面积落 F16?"
        );
        // tied 模型:表是以量化槽装上的(否则本门等于没测 tied 臂)
        let embed = entries
            .iter()
            .find(|e| e.key == "token_embd.weight")
            .expect("tied 模型必须有 token_embd 槽");
        assert_eq!(embed.dtype, Dtype::U8, "tied 量化表必须走 U8 块槽(§11.5 ②)");
        // 源侧事实:同一份文件二次打开一致(反变换描述符一致;零 I/O 病)
        let src2 = GgufSource::open(&path).expect("二次 open");
        assert_eq!(src2.dims(), GgufSource::open(&path).expect("三次 open").dims());
    }
}
