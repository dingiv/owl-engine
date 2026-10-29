# 量化流水线

## 端到端

```
awq-forge run --job job.toml
  load_job
  inspect_source
  build_plan
  prepare_workdir
  backend.quantize
  export_ct
  verify
  publish
```

## 0.8B 冒烟链路

```
run(job)
  JobSpec.load(job)                      # 源、输出、scheme、calib、device
  SourceProbe.inspect(source)
    read config.json
    read safetensors index
    classify_tensors()                   # BF16 / FP8 / packed
    build_quant_targets()                # Linear + ignore 面
  BackendSession.open(spec)
    llm-compressor oneshot(
      model = AutoModelForImageTextToText,
      recipe = [
        AWQModifier(n_grid=10, offload_device=None),
        QuantizationModifier(
          config_groups = g32-asym,
          ignore = visual/mtp/in_proj_a/b/...,
        ),
      ],
      pipeline = "sequential",
      sequential_offload_device = "cuda",
      batch_size = 8,
      max_seq_length = 2048,
    )
      trace/split decoder layers
      per-layer AWQ scale search
      compress directly to CT
  CtArtifact.save(.work/ct)
    save_pretrained(save_compressed=True)
    copy preprocessor_config.json
  Verifier.run(.work/ct)
    structural checks
    dequant sample parity
    engine smoke (optional)
  publish(.work/ct, output)
```

## 4B 验收链路

与 0.8B 相同，只改：

- 校准 128×2048；
- `batch_size=8`，`n_grid=10`，`seq_offload=cuda:1`（第二张卡接激活）；
- 实测 229s / 17.2 GiB；`cuda`（同卡）时 batch8 OOM；
- 质量门 `GSM8K 200 / needle 16k / std 15/15`；
- 质量门失败 → 不发布，保留报告。

## 27B FP8 链路（P1）

```
run(job)
  SourceProbe.inspect(source)
    detect fp8 blockwise 128×128
    verify weight_scale_inv coverage
  BackendSession.open(spec)
    GPTQModel.load(source, AWQConfig(...))
      native_floatx_source = fp8
      AutoModuleDecoderConfig(target_dtype=bf16)
      offload_to_disk=True               # 只落源/壳，不落 54GB dense
      LazyTurtle                        # 按模块读 checkpoint
  quantize(calib)
    for layer:
      materialize current layer          # FP8 -> BF16，仅当前层
      AWQ scale search / quantize
      release dense layer
  export CT
  verify
```

关键：**不调用** `compressed_tensors.convert_checkpoint` 全模型反量化；
它会在磁盘上生成 54GB dense。P1 优先验证 GPTQModel 的 module-local decoder
（只借它的 FP8 解码，不借它的 Qwen3.5 AWQ 映射）；若不可用，再实现
`compressed-tensors` `FP8BlockDequantizer` 的逐层封装。

## 失败语义

| 阶段 | 失败动作 |
|---|---|
| inspect | 非零退出，打印缺哪个键/哪个 scale |
| quantize | 保留 `.work/`，不写 output |
| export | 保留 AutoAWQ 中间态，便于重跑 |
| verify | 保留产物到 `.work/rejected/`，manifest 标记原因 |
| publish | 原子 rename；失败回滚到旧 output |

## 进程与并发

- 首期单进程；GPU 0 一张卡。
- 27B 双卡时按层分配，不引入多进程；先验证单卡逐层。
- 所有日志写 `.work/run.log`，stdout 只留阶段摘要。
