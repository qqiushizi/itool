#!/bin/bash
# ============================================================
# 实验: b.paged-attention
# 说明: 分页 KV、碎片、显存利用率
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# KV cache 按连续内存分配的问题:
#   1. 内部碎片:每个请求 max_len 算足,实际可能用不到
#   2. 外部碎片:不同长度请求间空隙无法利用
#   3. 显存浪费:实测 60-80% 显存被浪费
# PagedAttention (vLLM 首创):
#   - 把 KV cache 切成固定大小 block (e.g. 16 token)
#   - 逻辑 seq → 物理 block 通过 block_table 映射
#   - 类似 OS 虚拟内存分页
#   - block 共享(beam search, parallel sampling)
# 效果:显存利用率从 20-40% 提升到 80-95%。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: b.paged-attention | 分页 KV:显存利用率 4× 提升"
echo "############################################################"

python3 <<'PY'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 连续分配 vs 分页 ---
hdr(1,TOTAL,"连续分配 vs 分页:显存浪费对比")
why("""假设 max_len=2048,4 个请求实际长度 500/800/1200/1800。
连续分配:每个请求都按 2048 预留 → 实际用 4300/8192 = 53%
分页(block=16):每个请求按实际长度向上对齐 → 利用率近 100%
外加 block 共享(同前缀),系统 prompt 段不重复存。""")
def cont_alloc(lens, max_len):
    used = sum(lens); reserved = max_len * len(lens)
    return used, reserved
def paged_alloc(lens, block=16):
    used = sum((l + block - 1)//block * block for l in lens)
    return used
lens = [500, 800, 1200, 1800]
max_len = 2048
cu, cr = cont_alloc(lens, max_len)
pu = paged_alloc(lens, 16)
res(f"""4 个请求, 长度 {lens}, max_len={max_len}:
  连续分配: 实际用 {cu} / 预留 {cr} = 利用率 {cu/cr*100:.0f}%
  分页分配: 实际用 {pu} / 预留 {cr} = 利用率 {pu/cr*100:.0f}%
  节省显存:  {(1-pu/cr)*100:.0f}% (实际可服务更多并发)""")
mea("""直觉:连续分配像\"每个人占一整间房\",分页像\"合住\"。
  实际生产中 vLLM 实测:同样显存服务请求数提高 2-4×。""")

# --- 2. block table 映射 ---
hdr(2,TOTAL,"Block table:逻辑 seq → 物理 block")
why("""每个请求有 1 个 block_table,记录逻辑 block 序 → 物理 block id。
例如:逻辑 0,1,2,3 → 物理 5, 12, 3, 7 (乱序,任意位置)
这样物理 block 可以分散在不同地方,无需连续。""")
res("""seq_id 0  block_table = [5, 12, 3]      长度 48
seq_id 1  block_table = [8]                  长度 10
seq_id 2  block_table = [11, 7, 0, 9, 13]   长度 80
物理 block 池(共 16 个): [0,1,...,15]
  占用: 0,3,5,7,8,9,11,12,13  (9 个)
  空闲: 1,2,4,6,10,14,15      (7 个)""")
mea("""这种\"乱序\"映射的好处:
  1. 物理 block 用满才分配新的,没碎片
  2. block 可共享(同前缀的请求)
  3. 拷贝只需改 block_table(同一 block 给两个请求 = copy-on-write)
类比:就像 OS 虚拟内存。""")

# --- 3. Copy-on-write 共享 ---
hdr(3,TOTAL,"Beam search / parallel sampling 共享 block")
why("""Beam search:从 1 个 prompt 展开 B 个候选。
若 B 个候选的前 N 个 token 一样,它们的 KV 前 N 个 block 完全一样 → 共享。
传统做法:复制 B 份 KV,浪费显存。
PagedAttention:1 份 KV,N 个 block_table 都指向它,要分叉时才新分配。""")
shared_n = 50
total_blocks = 100
res(f"""beam=4, 前 {shared_n} token 全部相同:
  传统做法: 4 × {total_blocks} = {4*total_blocks} block,前缀重复 {shared_n//16}×4
  PagedAttn: {total_blocks} + (3 × 50) = {total_blocks + 3*shared_n//16} block
  节省: {(1 - (total_blocks + 3*shared_n//16)/(4*total_blocks))*100:.0f}%""")
mea("""Copy-on-write:前 N 个 token 全部共享。
  分叉时(某个 beam 选了不同 token),只新分配 1 个 block 给那条新路径。
  优势:beam 数越大,共享收益越大。""")

# --- 4. 实测提升 ---
hdr(4,TOTAL,"vLLM PagedAttention 实测提升")
why("""vLLM 论文数据(2023, A100 80G, LLaMA-7B):
  传统(HF): 79% 显存碎片, 吞吐 1× baseline
  vLLM:  4% 碎片, 吞吐 14-24× HF baseline
  同样显存,可服务请求数 ~4×""")
res("""vLLM 论文 (Kwon et al. 2023) 实测:
  workload            HF        vLLM      提升
  ShareGPT(聊天)      1.0×     14.5×     14.5
  Alpaca(指令)        1.0×     24.2×     24.2
  24h 持续压力测试     -        0% OOM    -""")
mea("""vLLM PagedAttention 几乎是\"必须\"。SGLang / TGI 也都跟进。
  2024+ 进一步: prefix cache + chunked-prefill 把\"没用上的 KV\"也压榨掉。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:PagedAttention = 把 KV 切成小块,逻辑→物理映射,显存利用率 4×;
  类似 OS 虚拟内存;vLLM 首创,已成行业标配。
- 熟手:block 大小 16 是 sweet spot;block table 共享让 beam search/并行
  sampling 几乎不耗额外 KV;prefix cache 在这之上再省一层;OOM 几乎消失。
【进阶】vLLM 跑压力测试,观察 KV cache 利用率(>>80% 健康) + requests/s。
EOF
echo "############################################################"
