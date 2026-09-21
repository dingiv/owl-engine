# Owl

从架构层重新设计的推理引擎。graph 与 TP 通信为一等公民(公理见
`docs/arch/charter.md` §二)。生产引擎仍是 vLLM;本项目按里程碑慢养。

- 立项总设:`docs/arch/charter.md`
- 需求基线:`packages/xinfer/docs/arch/requirements.md`(冻结继承)
- 判例库:`packages/xinfer`(冻结存档)/ `docs/mistral.rs` / `docs/bench`
