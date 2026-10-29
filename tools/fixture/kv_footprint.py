import sys
def rd(p): return open(p,'rb').read()
try:
    a=rd('/tmp/kv-before.bin'); b=rd('/tmp/kv-after.bin')
except Exception as e:
    print("ERR", e); sys.exit(1)
meta=open('/tmp/kv-meta.txt',encoding='utf-8').read().strip().split('\n')
print("meta:", meta[-2:])
n=len(a); assert len(b)==n, (len(a), len(b))
diff=[i for i in range(n) if a[i]!=b[i]]
print("页字节数:", n, "改动字节:", len(diff))
if not diff:
    print("⇒ 该页未被写(槽不在这一页?)"); sys.exit(0)
# 落在这页里的那一行的页内偏移(从 meta 取 off)
off=int([l for l in meta if 'tag=after' in l][-1].split('off=')[1].split()[0])
lo,hi=min(diff),max(diff)
print("改动区间: [%d, %d]  页内行偏移 off=%d" % (lo,hi,off))
# 假设 A: page-major ⇒ [off*hkv*hd, +hkv*hd) 一整段连续
# 假设 B: x-packed  ⇒ 每 head 每 d/8 块 8B,块间跨 page*8
hd=256; hkv=4; page=32
rowA=off*hkv*hd
A = all(rowA <= i < rowA+hkv*hd for i in diff)
print("假设 A(page-major, 连续 %d B @%d): %s" % (hkv*hd, rowA, "命中" if A else "不符"))
# B: 对每个 head h、每个 d8: [h*(hd/8)*page*8 + d8*page*8 + off*8, +8)
Bs=set()
for h in range(hkv):
    for d8 in range(hd//8):
        s0=h*(hd//8)*page*8 + d8*page*8 + off*8
        Bs.update(range(s0,s0+8))
B = all(i in Bs for i in diff) and len(diff)>0
cov = len([i for i in diff if i in Bs])/len(diff)
print("假设 B(x-packed, 8B×%d 块 strided %dB): %s (覆盖 %.2f)" % (hkv*hd//8, page*8, "命中" if B else "不符", cov))
# 也报一下"改动落在哪几个 8B 块"的形态
blk_ids=sorted({i//8 for i in diff})
print("改动的 8B 块数:", len(blk_ids), "前 6 块:", blk_ids[:6], "…" if len(blk_ids)>6 else "", "块间距样本:", [blk_ids[i+1]-blk_ids[i] for i in range(min(5,len(blk_ids)-1))])
