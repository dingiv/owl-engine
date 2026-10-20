//! **KernelTagIndex** —— 按 tag 反查算子元数据的运行时索引（2026-10-20）。
//!
//! 单一真源 = 每个算子的 `KernelSpec::TAGS`（经 `#[kernel_tags]` 注入）+
//! build.rs 生成的 `KERNEL_TAGS` 词表。本模块**不持有静态全局**：由
//! boot/工具面显式构造，随调用链传递（项目纪律：不用全局变量）。
//!
//! 用途：
//! - 白名单从 tag 派生（`feat.vision` / `kv.main.fp8` / `quant.w4a16` …）；
//! - boot 报告可审计「本档实际需要哪些 tag、每个 tag 命中哪些核」；
//! - 新算子忘打 tag ⇒ [`KernelTagIndex::from_metas`] 结构化拒（不静默）。

use crate::spec::{KernelMeta, KernelTag, KERNEL_TAGS, KERNEL_TAG_COUNT};

/// tag 索引构造错误（响亮、可指名）
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TagIndexError {
    /// 算子的 tag 不在 `tags.toml` 词表（理论上被宏编译期拦住；防御手工 TAGS）
    UnknownTag { op: &'static str, tag: &'static str },
    /// 生产算子无 tag（审计盲区）
    EmptyTagSet { op: &'static str },
    /// 同一 entry 出现两次（总表/扫描器故障）
    DuplicateOp { entry: &'static str },
}

impl std::fmt::Display for TagIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownTag { op, tag } => {
                write!(f, "算子 `{op}` 的 tag `{tag}` 不在 tags.toml 词表")
            }
            Self::EmptyTagSet { op } => write!(f, "算子 `{op}` 没有 tag（生产算子必须至少 1 个）"),
            Self::DuplicateOp { entry } => write!(f, "算子 `{entry}` 在总表里重复"),
        }
    }
}

impl std::error::Error for TagIndexError {}

/// tag → 算子元数据的只读倒排索引
#[derive(Clone, Debug)]
pub struct KernelTagIndex {
    /// 下标 = `KernelTag::index()`；长度 = `KERNEL_TAG_COUNT`
    by_tag: Vec<Vec<&'static KernelMeta>>,
    /// 建索引时的全量算子（顺序 = 总表顺序）
    all: Vec<&'static KernelMeta>,
}

impl KernelTagIndex {
    /// 从生产总表建索引（boot/工具面入口）
    pub fn build() -> Result<Self, TagIndexError> {
        Self::from_metas(crate::ops::all())
    }

    /// 从任意 meta 迭代器建索引（测试/白名单子集用）
    pub fn from_metas(
        metas: impl IntoIterator<Item = &'static KernelMeta>,
    ) -> Result<Self, TagIndexError> {
        let mut by_tag: Vec<Vec<&'static KernelMeta>> = vec![Vec::new(); KERNEL_TAG_COUNT];
        let mut all: Vec<&'static KernelMeta> = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for m in metas {
            if !seen.insert(m.entry) {
                return Err(TagIndexError::DuplicateOp { entry: m.entry });
            }
            if m.tags.is_empty() {
                return Err(TagIndexError::EmptyTagSet { op: m.entry });
            }
            for t in m.tags {
                let i = t.index();
                if i >= KERNEL_TAG_COUNT || KERNEL_TAGS[i].tag != *t {
                    return Err(TagIndexError::UnknownTag { op: m.entry, tag: t.as_str() });
                }
                by_tag[i].push(m);
            }
            all.push(m);
        }
        Ok(Self { by_tag, all })
    }

    /// 某 tag 命中的算子（顺序 = 总表顺序）
    pub fn tagged(&self, tag: KernelTag) -> &[&'static KernelMeta] {
        &self.by_tag[tag.index()]
    }

    /// 按 tag id 查询（未知 id = `None`，不 panic；调用方决定是否响亮拒）
    pub fn tagged_id(&self, id: &str) -> Option<&[&'static KernelMeta]> {
        crate::spec::tag_by_id(id).map(|t| self.tagged(t))
    }

    /// 全量算子（顺序 = 总表顺序）
    pub fn all(&self) -> &[&'static KernelMeta] {
        &self.all
    }

    /// 每个 tag 的命中数（`(tag, count)`；0 命中也列出）
    pub fn counts(&self) -> Vec<(KernelTag, usize)> {
        KERNEL_TAGS
            .iter()
            .map(|info| (info.tag, self.by_tag[info.tag.index()].len()))
            .collect()
    }

    /// tag 集 → 算子 entry 集（去重；白名单派生的最小单元）
    pub fn entries_for(&self, tags: &[KernelTag]) -> Vec<&'static str> {
        let mut out = std::collections::BTreeSet::new();
        for t in tags {
            for m in self.tagged(*t) {
                out.insert(m.entry);
            }
        }
        out.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_table_is_complete_and_unique() {
        let n = crate::ops::all().count();
        assert_eq!(n, crate::ops::OP_COUNT, "生成总表 OP_COUNT 与 all() 不一致");
        assert!(n >= 200, "总表只有 {n} 张，扫描器失效");
        let mut seen = std::collections::BTreeSet::new();
        for m in crate::ops::all() {
            assert!(seen.insert(m.entry), "总表重复 entry:{}", m.entry);
        }
    }

    #[test]
    fn kv_tags_correspond_to_declared_pool_access() {
        // tag 是选核声明:带 kv.* 的算子必须真的声明了 KV 池访问;
        // 没有 POOL 的通用 norm/rope 不得因为 dtype 后缀被误打 kv tag
        // (否则 fp8 档的 tag filter 会误杀它们)。
        let mut bad = Vec::new();
        for m in crate::ops::all() {
            let has = |prefix: &str| m.tags.iter().any(|t| t.as_str().starts_with(prefix));
            let main_pool = m.pool.iter().any(|p| p.pool == "kv.main");
            let draft_pool = m.pool.iter().any(|p| p.pool == "kv.draft");
            if main_pool != has("kv.main.") || draft_pool != has("kv.draft.") {
                bad.push((
                    m.entry,
                    main_pool,
                    draft_pool,
                    m.pool.iter().map(|p| p.pool).collect::<Vec<_>>(),
                    m.tag_ids(),
                ));
            }
        }
        assert!(bad.is_empty(), "kv.main/kv.draft tag 与 POOL 不一致: {bad:#?}");
    }

    #[test]
    fn graph_tag_matches_capture_evidence() {
        // `feat.graph` 不是手抄：宏只在 impl 的 CAPTURE 声明为
        // `NoHostWork`（契约 = 可进图）时注入；这里做逐核双向锁。
        for m in crate::ops::all() {
            let tag = m.has_tag(crate::spec::KernelTag::FEAT_GRAPH);
            let cap = m.capture == crate::spec::CaptureEvidence::NoHostWork;
            assert_eq!(tag, cap, "{}: feat.graph={tag} 但 CAPTURE={:?}", m.entry, m.capture);
        }
    }

    #[test]
    fn tag_index_rejects_untagged_and_unknown() {
        let idx = KernelTagIndex::build().expect("生产总表应可建 tag 索引");
        assert_eq!(idx.all().len(), crate::ops::OP_COUNT);
        // 每个生成 tag 都在词表里
        for m in idx.all() {
            for t in m.tags {
                assert_eq!(KERNEL_TAGS[t.index()].tag, *t, "{}: tag 位错位", m.entry);
            }
        }
        // 竖切阶段先锁已覆盖数；全量 tag 完成后本断言改为「零未打 tag」
        let mut untagged = Vec::new();
        for m in idx.all() {
            if m.tags.is_empty() {
                untagged.push(m.entry);
            }
        }
        eprintln!("[tag-index] 已打 tag {} / 总 {}（未打 {}）", idx.all().len() - untagged.len(), idx.all().len(), untagged.len());
        assert!(untagged.is_empty(), "仍有 {} 个算子未打 tag: {:?}", untagged.len(), &untagged[..untagged.len().min(20)]);
    }
}
