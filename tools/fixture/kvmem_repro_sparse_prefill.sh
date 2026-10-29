#!/bin/bash
# KVMem 稀疏 prefill 硬塌 —— 最小复现(2026-10-28)
#
# 病灶:prefill 走稀疏视图 ⇒ 模型对「文档末尾一条事实」也答不出(out=1, `:` 后 EOS);
#       同一 ws 下 prefill_view=on-hole(全链)则正常。全程无下沉、无洞、无回灌。
#
# 用法: tools/kvmem_repro_sparse_prefill.sh [ws...]   缺省 7680(活) 7168(死)
# 前置: 仓库根构建好 target/release/server(单张空闲 3090 Ti,端口 8140)
set -u
cd "$(dirname "$0")/.."
WSS=${*:-"7680 7168"}
python3 - <<'PY' $WSS
import json, re, sys, time, subprocess, urllib.request
OWL="."; PORT=8140
GPU="GPU-89dffb90-d810-7ce7-f649-60d781508f21"   # 换卡请改这里(必须空闲)
WSS=[int(x) for x in sys.argv[1:]]

def stop():
    out=subprocess.run(["ss","-ltnp"],capture_output=True,text=True).stdout
    for line in out.splitlines():
        if f":{PORT}" in line:
            m=re.search(r"pid=(\d+)",line)
            if m:
                import os,signal; os.kill(int(m.group(1)),signal.SIGINT); time.sleep(6); return

DOCS = [
    "roadmap.local/26-10-27-KVMem-a8-重叠与视图口径.md",
    "roadmap.local/26-10-27-KVMem-缺陷清单与后续路线.md",
    "roadmap.local/26-10-19-store回收完全体-立案.md",
    "roadmap.local/26-10-19-decode稳定性与GDN小刀.md",
    "roadmap.local/26-10-17-模块审查-次轮.md",
]

def build():
    """⚠️ **逐字节固定**的 8.5k 文档(prompt 8570 tok / 268 块)。

    换文档 ⇒ 换掉的是**哪批块被丢** ⇒ 生死会翻转(实测同一 ws 下
    8292 tok 的变体能答、8570 tok 的不能)⇒ 复现必须用这份构造,别"顺手简化"。
    """
    import os
    buf,n=[],0
    for p in DOCS:
        t=open(p,encoding="utf-8").read()
        buf.append(f"\n\n=== 资料 {p.rsplit('/',1)[-1]} ===\n"+t); n+=len(t)
        if n>=15000: break
    doc="".join(buf)[:15000]
    doc+="\n\n【尾部事实 T-9】本项目最终交付日期是 2026 年 12 月 31 日，负责人代号是 蓝鲸七号。\n"
    q="资料最后一条【尾部事实 T-9】里写的负责人代号是什么？只输出代号。"
    return doc,q

doc,q=build()
for ws in WSS:
    stop()
    cmd=["./target/release/server","-p","profiles/measurement/qwen38_27b-3090-kvmem.toml",
         "--set",f'parallel.device=["{GPU}"]',"--set","kvmem.prefill_view=always",
         "--set",f"kvmem.workspace_per_req={ws}","--log-level","warn"]
    log=f"/tmp/kvmem_repro_{ws}.log"
    with open(log,"w") as f:
        subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(180):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",timeout=2) as r:
                if json.loads(r.read()).get("ok"): break
        except Exception: time.sleep(1)
    else:
        print(f"ws={ws} BOOT FAIL(见 {log})"); continue
    body={"model":"qwen3.8-27b","messages":[{"role":"system","content":"你是严谨的技术助手，只依据给定资料回答。"},{"role":"user","content":doc+"\n\n"+q}],"temperature":0,"max_tokens":64,"stream":False,"reasoning_effort":"none"}
    req=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=3600) as r: j=json.loads(r.read())
    ch=j["choices"][0]; txt=(ch["message"]["content"] or "")
    print(f"ws={ws:5d}: prompt={j['usage']['prompt_tokens']} out={j['usage']['completion_tokens']:3d} | {txt[:48]!r}", flush=True)
stop()
PY
