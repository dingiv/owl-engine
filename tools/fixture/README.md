# tools/fixture

从 `tools/` 根收拢的可复现工具。按用途分：

| 类别 | 文件 |
|---|---|
| 金标 / 质量 | `dflash2_golden.py`、`vision_golden.py`、`vlm_e2e_golden.py`、`quality_eval.py`、`needle_eval.py` |
| 性能 / 分账 | `ctxladder.py`、`prefill-account.sh`、`prefill_account.py`、`nsys-decode-account.sh`、`nsys_decode_account.py`、`tp-ab-std.sh`、`vram_timeline.py` |
| KVMem 门 | `kvmem_visibility_gate.py`、`kvmem_ttft_gate.py`、`kvmem_material_gate.py`、`kvmem_repro_sparse_prefill.sh`、`kv_footprint.py` |
| GPU 探针 | `ar-*.cu`、`p2p-*.cu`、`vmm-*.cu`、`gdn-rec-probe.cu`、`nccl-ar-probe.cu` |
| 启动 / 跑批 | `serve-w4a16-3090.sh`、`run-agent-workload.sh`、`run-dflash-longctx-probe.sh`、`run-nsys-dflash.sh` |
| 辅助 | `fixedtail_prompt.py`、`trunc_ckpt_layers.py`、`select_fingerprint.py`、`gen_third_party_licenses.py` |

> 这些脚本的 `ROOT` 以 `tools/fixture/` 为基准上移两级；从 owl 根调用时路径不变。
