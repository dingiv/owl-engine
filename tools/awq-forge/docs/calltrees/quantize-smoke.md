# 0.8B 冒烟 calltree

```
awq-forge run --job jobs/qwen35-08b.toml
  JobSpec.load()
    parse TOML
    resolve source/output/calib/device
    validate paths
  SourceProbe.inspect(source)
    read config.json
    read model.safetensors.index.json
    classify_tensors()
      BF16 dense
      vision / mtp / norm / embed
    build_quant_targets()
      full-attn: q/k/v/o
      GDN: in_proj_qkv / in_proj_z / out_proj
      MLP: gate/up/down
      ignore: in_proj_a / in_proj_b / conv1d / norm / visual / mtp
  BackendSession.open()
    llm-compressor oneshot(
      model = AutoModelForImageTextToText,
      recipe = [
        AWQModifier(n_grid=10, offload_device=None),
        QuantizationModifier(g32-asym, ignore=...),
      ],
      pipeline = "sequential",
      sequential_offload_device = "cuda",
      batch_size = 8,
    )
      trace/split decoder layers
      per-layer AWQ scale search
      compress directly to CT
  save(.work/ct)
    save_pretrained(save_compressed=True)
    copy preprocessor_config.json
  Verifier.run()
    keys/dtypes/shapes
    dequant sample vs AutoAWQ
    optional engine load
  publish(output)
```
