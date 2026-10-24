# 组件目录（不是可直接启动的完整档）。
#
# leaf 的 extends 顺序建议：
#   full.toml
#   → parts/model/<model>.toml
#   → parts/feature/<feature>.toml ...
#   → parts/hardware/<hw>.toml
#   → parts/pool/<pool>.toml
#   → parts/kernel/<kernel>.toml
# 最后 leaf 只写站点键（server.bind / runtime.max_seq / 路径）。
