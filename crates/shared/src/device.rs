//! **声明式选卡**(2026-10-16):`device = { bdf | find_name | uuid }`。
//!
//! 语义(用户裁决):
//! - 三字段**各自可选**;**同时给出 = 交集(AND)**;
//! - **多命中全部收进列表**(为多卡 / 张量并行预留;单卡期取首张);
//! - 整个 spec 为空(配置里不写 device)⇒ **全部设备都是候选**(上层再按
//!   "未被占用"过滤 ⇒ 等价"自动挑一张空闲卡");
//! - 找卡**不用 ordinal**(CUDA 序会漂),用 bdf / 名字 / uuid 这类稳定身份。
//!
//! 本模块只做**纯匹配**(可 CPU 单测);枚举设备(ordinal/name/bdf/uuid)由
//! 后端提供,占用过滤由 `crate::gpu_lock` 提供。
//!
//! # 附:peer 额度常量(TP2 S1)
//!
//! 单笔 peer 可映射缓冲的三个硬约束常量(判据出处见各常量注释)。
//! **单源**在此:通信策略层(`owl_cuda::comm::peer` 的 re-export)与执行层
//! (`owl-cuda` 的 server 侧硬门)读同一份 —— 两边不许各写一套。

/// 单笔 peer 可映射缓冲的**硬上限**(192 MiB)。
///
/// 由来:3080 档 BAR1 256 MiB − 64 MiB 驱动预留 = 192 MiB;超限时驱动
/// 预算检查硬拒(`docs/llama.cpp/v3基线不通-立案排查.md` §10.5 约束 1,
/// 原件:`docs/llama.cpp/vmm-p2p-3080会战-测试报告.md` 坑 3/7)。
/// 纪律(2026-10-16 用户裁决):**peer 单笔 ≤ 192MB**。
pub const PEER_ALLOC_HARD_CAP: u64 = 192 << 20;

/// Window 档的 BAR1 预留量(驱动侧 `nv_dynbar1_reserve_mb` 的对账口径)。
pub const WINDOW_BAR1_RESERVE: u64 = 64 << 20;

/// 判为 Static(宽 BAR1,不走动态窗口)所需的 BAR1 下限:2 GiB。
///
/// 实测对照(本机):3090 Ti = 32768 MiB(Static)· 3080 20G 魔改 = 256 MiB(Window)。
pub const STATIC_BAR1_MIN: u64 = 2 << 30;

/// 一台设备的事实(由后端枚举填;uuid 稳定,ordinal 只作最终下发给引擎的序号)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceRow {
    /// CUDA ordinal(仅用于最终 `cudaSetDevice`;不参与匹配)
    pub ordinal: usize,
    /// 设备名(如 `NVIDIA GeForce RTX 3090 Ti`)
    pub name: String,
    /// PCI BDF(`0000:42:00.0`;允许只写 `42:00.0` 或 `42.0`)
    pub bdf: String,
    /// 设备 UUID(如 `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)
    pub uuid: String,
}

/// UUID → 规范显示串:`GPU-` + 8-4-4-4-12 小写十六进制(与 NVIDIA 官方写法一致)
pub fn uuid_display(bytes: &[u8; 16]) -> String {
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[0..8], &hex[8..12], &hex[12..16], &hex[16..20], &hex[20..32]
    )
}

/// UUID 归一(**匹配端与显示端同一套**):去 `GPU-`/`gpu-` 前缀、去连字符、小写
pub fn uuid_norm(s: &str) -> String {
    let t = s.trim();
    let t = t.strip_prefix("GPU-").or_else(|| t.strip_prefix("gpu-")).unwrap_or(t);
    t.chars().filter(|c| c.is_ascii_hexdigit()).map(|c| c.to_ascii_lowercase()).collect()
}

/// 解析 `nvidia-smi --query-gpu=uuid,memory.free,memory.total --format=csv,noheader,nounits`
/// 的输出 → `uuid → 空闲占比`(行形如 `GPU-xxxx, 24148, 24564`;非法行跳过)。
/// 用途:auto-pick 时避让**非 owl 进程/未持锁实例**占用的卡(锁只认 owl)。
pub fn parse_nvidia_smi_mem(text: &str) -> std::collections::HashMap<String, f64> {
    let mut out = std::collections::HashMap::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 3 || parts[0].is_empty() {
            continue;
        }
        if let (Ok(f), Ok(t)) = (parts[1].parse::<f64>(), parts[2].parse::<f64>()) {
            if t > 0.0 {
                out.insert(parts[0].to_string(), f / t);
            }
        }
    }
    out
}

/// **测试设备门控值**(`OWL_TEST_DEVICE`;2026-10-18 用户裁决:**唯一形态 = BDF**,
/// CUDA 序号禁用)。未设/空串 = `None`(GPU 用例据此 skip)。
///
/// 为什么是 BDF:CUDA 序号是 FASTEST_FIRST 序,随驱动/换卡/重启漂移;本机已有
/// 多次"数字序钉错卡"判例(见 AGENTS.md)。BDF 稳定,且与 `DeviceSpec.bdf` 同一
/// 套归一/匹配(`bdf_norm`)。
pub fn test_device_bdf() -> Option<String> {
    let s = crate::env_reader::str("OWL_TEST_DEVICE")?;
    let t = s.trim();
    (!t.is_empty()).then(|| t.to_string())
}

/// 解析 `nvidia-smi -q -d MEMORY` 输出 → `BDF(规范) → BAR1 bytes`。
///
/// 形如(每个设备一段;BDF 行 `GPU 00000000:41:00.0` + 段内
/// `BAR1 Memory Usage / Total : 32768 MiB`)。用途:TP 分档
/// (`owl_cuda::comm::peer::PeerCaps.bar1_bytes`)—— CUDA 无 BAR1 属性、
/// NVML 不在依赖面,而 `--query-gpu` **没有** `bar1.total` 字段(实测
/// 该字段在 615 驱动上返回 rc=2)⇒ 只能走详查面。
/// 非法/缺段行跳过(缺哪张卡由调用方按 BDF 响亮拒)。
pub fn parse_nvidia_smi_bar1(text: &str) -> std::collections::HashMap<String, u64> {
    let mut out = std::collections::HashMap::new();
    let mut cur: Option<String> = None;
    let mut in_bar1 = false;
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("GPU ") {
            cur = Some(bdf_norm(rest));
            in_bar1 = false;
            continue;
        }
        if t.starts_with("BAR1 Memory Usage") {
            in_bar1 = true;
            continue;
        }
        if in_bar1 {
            if let Some(v) = t.strip_prefix("Total") {
                let v = v.trim_start_matches([' ', ':', '\t']);
                let mib = v.split_whitespace().next().and_then(|x| x.parse::<u64>().ok());
                if let (Some(bdf), Some(mib)) = (cur.as_ref(), mib) {
                    out.insert(bdf.clone(), mib << 20);
                }
                in_bar1 = false;
            }
        }
    }
    out
}

/// 选卡规格(全 `None` = 不指定 ⇒ 全部候选)
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bdf: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub find_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uuid: Option<String>,
}

impl DeviceSpec {
    /// 未指定任何字段
    pub fn is_empty(&self) -> bool {
        self.bdf.is_none() && self.find_name.is_none() && self.uuid.is_none()
    }

    /// 单条设备是否命中(**已给的字段必须全部满足** = 交集)
    pub fn matches(&self, row: &DeviceRow) -> bool {
        if let Some(b) = &self.bdf {
            if bdf_norm(b) != bdf_norm(&row.bdf) {
                return false;
            }
        }
        if let Some(n) = &self.find_name {
            if !row.name.to_lowercase().contains(&n.to_lowercase()) {
                return false;
            }
        }
        if let Some(u) = &self.uuid {
            if uuid_norm(u) != uuid_norm(&row.uuid) {
                return false;
            }
        }
        true
    }

    /// 全部命中(保持输入顺序;空 spec ⇒ 全部)
    pub fn select<'a>(&self, rows: &'a [DeviceRow]) -> Vec<&'a DeviceRow> {
        rows.iter().filter(|r| self.matches(r)).collect()
    }
}

/// **单串 → DeviceSpec**(`[parallel] device = ["42:00.0", "GPU-…", "3090"]`)。
///
/// 分类规则(纯语法,不做存在性判断):
/// - `GPU-…` 或 32 位十六进制(可带连字符)→ **uuid**;
/// - 形如 `[domain:]bus:dev.func`(全十六进制,`:`/`.` 分隔)→ **bdf**;
/// - 其余 → **find_name**(名字子串,大小写不敏感);
/// - **纯十进制整数 = 拒**:CUDA 序号会漂(AGENTS.md 多次判例),卡位必须
///   用 BDF / UUID / 名字。
pub fn parse_device_spec(raw: &str) -> Result<DeviceSpec, String> {
    let t = raw.trim();
    if t.is_empty() {
        return Err("空串(卡位必须给 BDF / UUID / 名字子串)".into());
    }
    // 纯数字:**0..=99 视为 CUDA 序号**(本机 ordinal 恒在 0..16)⇒ 禁用;
    // 4 位型号(3090/3080/4090…)是**名字子串**,放行。
    // 判据出处:AGENTS.md 多起"数字序钉错卡"判例 —— 能写死成 BDF 就写死。
    if t.chars().all(|c| c.is_ascii_digit()) {
        if t.len() <= 2 {
            return Err(format!(
                "\"{t}\" 是 CUDA 序号 —— 禁用(FASTEST_FIRST 序会随驱动/换卡漂移);\
                 请写 BDF(如 41:00.0)或 UUID、或名字子串(如 3090)"
            ));
        }
        return Ok(DeviceSpec { find_name: Some(t.to_string()), ..Default::default() });
    }
    // UUID:GPU- 前缀,或 32 位十六进制(允许 8-4-4-4-12 连字符)
    let hex_only: String = t.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    if t.starts_with("GPU-") || t.starts_with("gpu-") || t.starts_with("MIG-")
        || (hex_only.len() == 32 && t.chars().all(|c| c.is_ascii_hexdigit() || c == '-'))
    {
        return Ok(DeviceSpec { uuid: Some(t.to_string()), ..Default::default() });
    }
    // BDF:[domain:]bus:dev[.func] —— 两段或三段、全十六进制、分隔符只有 : 和 .
    let bdf_like = {
        let parts: Vec<&str> = t.split([':', '.']).filter(|p| !p.is_empty()).collect();
        (parts.len() == 2 || parts.len() == 3 || parts.len() == 4)
            && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_hexdigit()))
            && t.contains(':')
    };
    if bdf_like {
        return Ok(DeviceSpec { bdf: Some(t.to_string()), ..Default::default() });
    }
    Ok(DeviceSpec { find_name: Some(t.to_string()), ..Default::default() })
}

impl DeviceSpec {
    /// **单串解析**(`[parallel] device` 的每项;语义见 [`parse_device_spec`])
    pub fn parse(raw: &str) -> Result<Self, String> {
        parse_device_spec(raw)
    }

    /// 规范显示(日志/错误信息用;空 spec = `<auto>`)
    pub fn describe(&self) -> String {
        if let Some(u) = &self.uuid {
            u.clone()
        } else if let Some(b) = &self.bdf {
            b.clone()
        } else if let Some(n) = &self.find_name {
            format!("name~{n}")
        } else {
            "<auto>".into()
        }
    }
}

/// BDF 规范化:`0000:42:00.0` / `42:00.0` / `42:00` / `42.0` 一律可比
/// (去 domain、大小写不敏感、`:` 与 `.` 等价、去前导零)
pub fn bdf_norm(s: &str) -> String {
    let low = s.trim().to_ascii_lowercase();
    let body = match low.split_once(':') {
        // 带 domain ⇒ 去 domain。两源写法都要认:驱动枚举给 `0000:42:00.0`
        // (4 位),`nvidia-smi -q` 给 `00000000:41:00.0`(**8 位**)—— 2026-10-17
        // K2 探测当场踩到(两源归一后必须可比:TP 分档按 BDF 对账 BAR1)。
        Some((dom, rest))
            if (dom.len() == 4 || dom.len() == 8)
                && dom.chars().all(|c| c.is_ascii_hexdigit()) =>
        {
            rest
        }
        _ => low.as_str(),
    };
    // 只比 `bus:dev`(GPU 功能号恒 0,忽略)· 统一分隔符 · 去前导零
    let parts: Vec<String> = body
        .split([':', '.'])
        .filter(|p| !p.is_empty())
        .take(2)
        .map(|p| {
            let t = p.trim_start_matches('0');
            if t.is_empty() { "0".to_string() } else { t.to_string() }
        })
        .collect();
    parts.join(":")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<DeviceRow> {
        vec![
            DeviceRow { ordinal: 0, name: "NVIDIA GeForce RTX 3090 Ti".into(), bdf: "0000:41:00.0".into(), uuid: "GPU-aaaa".into() },
            DeviceRow { ordinal: 1, name: "NVIDIA GeForce RTX 3090 Ti".into(), bdf: "0000:42:00.0".into(), uuid: "GPU-bbbb".into() },
            DeviceRow { ordinal: 2, name: "NVIDIA GeForce RTX 3080".into(), bdf: "0000:01:00.0".into(), uuid: "GPU-cccc".into() },
            DeviceRow { ordinal: 3, name: "NVIDIA GeForce RTX 3080".into(), bdf: "0000:c1:00.0".into(), uuid: "GPU-dddd".into() },
        ]
    }

    /// 空 spec = 全部候选(上层再剔被占 ⇒ "自动挑空闲卡")
    #[test]
    fn empty_spec_selects_all() {
        let r = rows();
        let s = DeviceSpec::default();
        assert!(s.is_empty());
        assert_eq!(s.select(&r).len(), 4);
    }

    /// BDF 三种写法等价(带/不带 domain、`:`↔`.`)
    #[test]
    fn bdf_forms_are_equivalent() {
        let r = rows();
        for q in ["0000:42:00.0", "42:00.0", "42:0.0", "42:00", "42.0"] {
            let s = DeviceSpec { bdf: Some(q.into()), ..Default::default() };
            let hit = s.select(&r);
            assert_eq!(hit.len(), 1, "{q} 应命中恰 1 台,得 {}", hit.len());
            assert_eq!(hit[0].ordinal, 1, "{q} 应命中 42:00 那台");
        }
    }

    /// find_name 子串 + 大小写不敏感;多命中全收
    #[test]
    fn find_name_substring_multi_hit() {
        let r = rows();
        let s = DeviceSpec { find_name: Some("3090".into()), ..Default::default() };
        assert_eq!(s.select(&r).len(), 2);
        let s2 = DeviceSpec { find_name: Some("rtx 3080".into()), ..Default::default() };
        assert_eq!(s2.select(&r).len(), 2, "大小写不敏感");
    }

    /// 多字段 = 交集(AND);uuid 归一后精确(GPU- 前缀/连字符/大小写不敏感)
    #[test]
    fn fields_are_conjunctive() {
        let r = rows();
        let s = DeviceSpec { find_name: Some("3090".into()), bdf: Some("42:00.0".into()), ..Default::default() };
        let hit = s.select(&r);
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].uuid, "GPU-bbbb");
        let s2 = DeviceSpec { uuid: Some("gpu-CCCC".into()), ..Default::default() };
        assert_eq!(s2.select(&r)[0].ordinal, 2);
        // 交集为空 ⇒ 零命中(上层响亮拒绝)
        let s3 = DeviceSpec { find_name: Some("3080".into()), uuid: Some("GPU-aaaa".into()), ..Default::default() };
        assert!(s3.select(&r).is_empty());
    }

    /// UUID 显示/归一同源:128 位转规范串,再按各种写法都能匹配
    #[test]
    fn uuid_display_norm_roundtrip() {
        // 合成 UUID(**非**真实卡):01234567-89ab-cdef-0123-456789abcdef
        let bytes: [u8; 16] = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd,
            0xef,
        ];
        let disp = uuid_display(&bytes);
        assert_eq!(disp, "GPU-01234567-89ab-cdef-0123-456789abcdef");
        let row = DeviceRow { ordinal: 2, name: "RTX 3090 Ti".into(), bdf: "0000:42:00.0".into(), uuid: disp.clone() };
        for q in [disp.clone(), disp.to_uppercase(), disp.replace('-', ""), "01234567-89ab-cdef-0123-456789abcdef".into()] {
            let s = DeviceSpec { uuid: Some(q.clone()), ..Default::default() };
            assert!(s.matches(&row), "写法 {q:?} 应命中");
        }
        let other = DeviceSpec { uuid: Some("GPU-00000000".into()), ..Default::default() };
        assert!(!other.matches(&row));
    }

    /// nvidia-smi 内存输出解析:合法行入表,脏行跳过(不炸)
    #[test]
    fn nvidia_smi_mem_parse() {
        let m = parse_nvidia_smi_mem("GPU-aaaa, 24148, 24564\nGPU-bbbb, 18, 24564\n垃圾行\n");
        assert_eq!(m.len(), 2);
        assert!((m["GPU-aaaa"] - 24148.0 / 24564.0).abs() < 1e-9);
        assert!(m["GPU-bbbb"] < 0.01, "空卡占比≈0 被占");
        assert!(parse_nvidia_smi_mem("").is_empty());
    }

    /// nvidia-smi BAR1 解析(`-q -d MEMORY` 详查面):BDF → bytes
    #[test]
    fn nvidia_smi_bar1_parse() {
        let text = "==============NVSMI LOG==============\n\
                    Attached GPUs : 2\n\
                    GPU 00000000:41:00.0\n\
                    \x20   FB Memory Usage\n\
                    \x20       Total : 24564 MiB\n\
                    \x20   BAR1 Memory Usage\n\
                    \x20       Total : 32768 MiB\n\
                    \x20       Used : 15 MiB\n\
                    GPU 00000000:01:00.0\n\
                    \x20   BAR1 Memory Usage\n\
                    \x20       Total : 256 MiB\n";
        let m = parse_nvidia_smi_bar1(text);
        assert_eq!(m.len(), 2, "两张卡各一条");
        // 键 = `bdf_norm` 规范形(驱动枚举与 nvidia-smi 两源同归一)
        assert_eq!(m[&bdf_norm("00000000:41:00.0")], 32768 << 20, "3090 Ti = 32GiB");
        assert_eq!(m[&bdf_norm("00000000:01:00.0")], 256 << 20, "3080 魔改 = 256MiB");
        // 两源可比(8 位 domain vs 4 位 domain;2026-10-17 K2 探测踩坑回归)
        assert_eq!(bdf_norm("00000000:41:00.0"), bdf_norm("0000:41:00.0"));
        assert!(parse_nvidia_smi_bar1("").is_empty());
    }
}
