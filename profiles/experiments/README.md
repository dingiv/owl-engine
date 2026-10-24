# experiments —— 实验/取证档（可随时改，不作为生产入口）

这些档保留历史矩阵，路径变了：例如
`profiles/qwen38_27b-3090-w4a16-dflash-longctx.toml`
→ `profiles/experiments/qwen38_27b-3090-w4a16-dflash-longctx.toml`。

生产入口只保留在 `profiles/` 顶层；新实验优先用
`-p profiles/<canonical>.toml --set key=value`，不要为每个变体新建文件。
