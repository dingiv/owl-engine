import json, os, re, struct, sys, shutil
SRC = "./models/cyankiwi/Qwen3.8-27B-AWQ-INT4"
DST_ROOT = "./packages/owl-worktrees/w4a8-p1/target/t27b"
N = int(sys.argv[1])
DST = f"{DST_ROOT}-{N}"
os.makedirs(DST, exist_ok=True)
def keep(name):
    m = re.match(r"model\.language_model\.layers\.(\d+)\.", name)
    return True if not m else int(m.group(1)) < N
copied = 0
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith(".safetensors"): continue
    src = f"{SRC}/{fn}"
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data_off = 8 + n
        meta = hdr.pop("__metadata__", None)
        keep_t = {k: v for k, v in hdr.items() if keep(k)}
        if not keep_t:
            continue
        # 重排 data_offsets(按原序紧密排布)
        new = {}
        cur = 0
        blobs = []
        for k, v in keep_t.items():
            s, e = v["data_offsets"]
            ln = e - s
            new[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [cur, cur + ln]}
            blobs.append((s, ln))
            cur += ln
        out_hdr = dict(new)
        if meta is not None: out_hdr["__metadata__"] = meta
        hb = json.dumps(out_hdr, separators=(",", ":")).encode()
        pad = (8 - len(hb) % 8) % 8
        hb += b" " * pad
        with open(f"{DST}/{fn}", "wb") as g:
            g.write(struct.pack("<Q", len(hb)))
            g.write(hb)
            for (s, ln) in blobs:
                f.seek(data_off + s)
                g.write(f.read(ln))
        copied += 1
        print(f"  {fn}: 保留 {len(keep_t)} 张量 / {cur/1e9:.2f}GB")
for fn in ("config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","chat_template.jinja","generation_config.json"):
    d = f"{DST}/{fn}"
    if not os.path.lexists(d): os.symlink(f"{SRC}/{fn}", d)
print(f"N={N} → {DST}({copied} 个文件)")
