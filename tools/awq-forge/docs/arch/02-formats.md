# 输入 / 输出格式

## 输入：BF16 dense

- `config.json` 无 `quantization_config`，或 `quant_method` 非 fp8/compressed-tensors。
- safetensors 张量 `*.weight` 为 BF16/F16。
- 0.8B / 4B 首期都走这条。

## 输入：FP8 blockwise

以 `models/Qwen/Qwen3.8-27B-FP8` 为准：

```json
{
  "quant_method": "fp8",
  "fmt": "e4m3",
  "activation_scheme": "dynamic",
  "weight_block_size": [128, 128]
}
```

- `*.weight`：`F8_E4M3`，形状 `[out, in]`。
- `*.weight_scale_inv`：F32，形状 `[ceil(out/128), ceil(in/128)]`。
- 反量化：`w_bf16 = w_fp8.float() * scale_inv`，按 128×128 块广播；
  边缘块按实际形状裁剪。
- `*.input_scale` 等激活量化参数忽略：输出是 W4A16，激活保持 BF16。

禁止：`F8_E4M3FNUZ`、MXFP8、per-tensor 标量 scale 未声明的情形；
`SourceProbe` 必须显式拒绝，不能猜。

## 输出：compressed-tensors pack-quantized

以 `models/cyankiwi/Qwen3.8-27B-AWQ-INT4` 为准。

### config.json

```json
{
  "quantization_config": {
    "config_groups": {
      "group_0": {
        "format": "pack-quantized",
        "targets": ["Linear"],
        "weights": {
          "actorder": null,
          "block_structure": null,
          "dynamic": false,
          "group_size": 32,
          "num_bits": 4,
          "observer": "mse",
          "observer_kwargs": {},
          "scale_dtype": null,
          "strategy": "group",
          "symmetric": false,
          "type": "int",
          "zp_dtype": "torch.int8"
        }
      }
    },
    "format": "pack-quantized",
    "ignore": ["..."],
    "kv_cache_scheme": null,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "sparsity_config": {},
    "transform_config": {}
  }
}
```

### 每个量化 Linear 的四件套

| 键 | dtype | shape | 语义 |
|---|---|---|---|
| `weight_packed` | I32 | `[out, in/8]` | 8 个 4bit 沿 input 维打包 |
| `weight_scale` | BF16 | `[out, in/32]` | 每组一个 scale |
| `weight_zero_point` | I32 | `[out/8, in/32]` | 8 个 zp 沿 output 维打包 |
| `weight_shape` | I64 | `[2]` | `[out, in]` |

示例（27B layer0 in_proj_qkv）：

```
weight_packed       I32 [10240, 640]     # in=5120
weight_scale        BF16 [10240, 160]    # in/32
weight_zero_point   I32 [1280, 160]      # out/8
weight_shape        I64 [2] = [10240, 5120]
```

### 打包纪律

- **不手写 bit 序**。AutoAWQ GEMM → CT 用
  `compressed_tensors.entrypoints.convert.converters.AutoAWQConverter`；
  它负责 AWQ 的 intra-int32 反序、signed 偏移、`pack_to_int32`。
- 如果将来直接产出 CT，也必须调用
  `compressed_tensors.compressors.pack_quantized.helpers.pack_to_int32`。
- 写入前做 round-trip：`unpack_from_int32` → dequant → 与 AWQ 中间态逐元素对拍；
  worst error 超过阈值直接失败。

### ignore 面

至少包含：

```
lm_head
model.language_model.embed_tokens
model.language_model.layers.*.linear_attn.in_proj_a
model.language_model.layers.*.linear_attn.in_proj_b
model.language_model.layers.*.linear_attn.norm
model.language_model.layers.*.linear_attn.conv1d
model.visual.*
mtp.*
*normalization*
```

具体以 `SourceProbe.build_quant_targets()` 的清单为准，并在 manifest 里落盘。

## 引擎侧期望

- vLLM / sglang 读到 `quant_method=compressed-tensors`，按 `pack-quantized` 装载。
- owl 读到同一布局时，`AwqSource` 接受 `(4,32,false)`。
- 如果引擎只认 AutoAWQ `qweight/qzeros/scales`，用导出的 AutoAWQ 中间态作为 fallback，
  但首期发布物以 CT 为准。
