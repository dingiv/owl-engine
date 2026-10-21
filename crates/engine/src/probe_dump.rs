//! **诊断 dump 路径的唯一构造点**。
//!
//! 原实现把 `/tmp/owl_dflash_mem.bin`、`/tmp/kv-{tag}.bin`、`/tmp/owl-prompt-*.txt`
//! 等 10 处散在各文件里 —— 换根目录(容器/CI 只读 /tmp)就得全仓搜。现在:
//! 名字由调用方给,**根只在这里**。

/// dump 根目录(后续可由配置注入;当前与历史行为一致 = `/tmp`)。
pub(crate) const PROBE_ROOT: &str = "/tmp";

/// 拼一个 dump 路径(**唯一入口**)。
pub(crate) fn probe_path(name: &str) -> String {
    format!("{PROBE_ROOT}/{name}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_joins_under_single_root() {
        assert_eq!(probe_path("owl_x.bin"), "/tmp/owl_x.bin");
        assert!(probe_path("kv-a.bin").starts_with(PROBE_ROOT));
    }
}
