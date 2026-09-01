#!/bin/bash
# ============================================================
# 实验: a.prefill-decode
# 说明: prefill vs decode 阶段计算特性
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Transformer 解码分两个阶段:
#   - PREFILL: 一次性把 prompt 全部塞进去,所有 token 并行算 QKV
#     特点: 计算密集(GEMM 大),能打满 GPU 算力
#   - DECODE:  一次只生 1 个 token,要反复跑 N 步直到 EOS
#     特点: 访存密集(每步要读整个 KV cache),算力利用率低
# 这就是为什么 LLM 推理要分两阶段优化:
#   - PREFILL: 优化 GEMM,跑得快
#   - DECODE:  优化访存,跑得轻
# 关键指标:
#   - TTFT (Time To First Token): 反映 PREFILL 速度
#   - TPOT (Time Per Output Token): 反映 DECODE 速度
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: a.prefill-decode | PREFILL vs DECODE 计算特性"
echo "############################################################"

python3 <<'PY'
import numpy as np, time
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 算力 vs 访存 ---
hdr(1,TOTAL,"PREFILL 算力密集 vs DECODE 访存密集")
why("""PREFILL 输入 prompt 长 L,计算 QKV+O: 4 个 L×d 矩阵乘 = 4*L*d^2*2 FLOPs。
  batch=1, L=1024, d=4096 → 137 GFLOPs
DECODE 每生成 1 个 token: 1 个 d×d 矩阵乘 + 读 KV cache
  计算量 = 2*d^2 = 33 MFLOPs(单 token)
  访存量 = L*2*d (读 KV)
PREFILL 的算力/访存比 = O(L*d) 大 → 算力密集
DECODE 的算力/访存比 = O(d/L) 小 → 访存密集""")
L, d, batch = 1024, 4096, 1
prefill_flops = 4 * L * d * d * 2
decode_flops_per = 2 * d * d
decode_bytes = L * 2 * d * 2  # KV cache, fp16
res(f"""L={L}, d={d}, batch=1:
  PREFILL 总算力: {prefill_flops/1e9:.1f} GFLOPs
  DECODE 单步算力: {decode_flops_per/1e6:.1f} MFLOPs (相差 ~{prefill_flops/decode_flops_per:.0f}×)
  DECODE 单步访存: {decode_bytes/1e6:.1f} MB (读 KV)
  算力/访存: prefill={prefill_flops/decode_bytes:.0f}  decode={decode_flops_per/decode_bytes:.0f}  (差距大)""")
mea("""PREFILL 算力比 = prefill 数字大得多 → GPU 算力能打满。
DECODE 算力比小到 ~1,主要时间花在等 KV 从 HBM 搬到 SM。
A100 HBM 带宽 2 TB/s,DECODE 一秒最多读 ~1M token 的 KV。""")

# --- 2. 实测 toy matmul 时间 vs 内存读 ---
hdr(2,TOTAL,"CPU 模拟:算力 vs 访存")
why("""在 CPU 上跑个 toy 实验:大 matmul vs 大量小乘,看哪种快。
对应:大 matmul = PREFILL,小乘 = DECODE。""")
def big_mm():
    A = np.random.randn(2048,4096); B = np.random.randn(4096,4096)
    t=time.perf_counter()
    for _ in range(3): C = A @ B
    return (time.perf_counter()-t)/3
def many_small():
    A = np.random.randn(4096,4096); x = np.random.randn(4096)
    t=time.perf_counter()
    for _ in range(1000): y = A @ x
    return (time.perf_counter()-t)/1000
mm = big_mm(); sm = many_small()
res(f"""CPU 模拟(单线程, numpy):
  PREFILL-like  2048×4096·4096×4096: {mm*1000:.1f} ms / 调用
  DECODE-like   4096×4096·4096:       {sm*1000:.4f} ms / 调用
  → PREFILL 单次慢但 FLOPs 多,DECODE 单次快但次数极多""")
mea("""PREFILL 像\"修一条 100km 高速\":前期慢,一旦修好后续全部受益。
DECODE 像\"高速公路上每 5km 收一次费\":每次 1ms 但 1000 次就 1s。
所以预填充 + 大量短回答,GPU 算力根本用不满——这就是 batch 推理的痛点。""")

# --- 3. TTFT vs TPOT 模型 ---
hdr(3,TOTAL,"TTFT + TPOT + 总时延分解")
why("""用户感知的延迟:
  TTFT  = PREFILL 时间      (决定\"首字快不快\")
  TPOT  = 后续每字平均时间  (决定\"流式顺不顺畅\")
  总时延 = TTFT + (n-1)*TPOT  (n = 生成 token 数)
例:prompt=512,生成 200 token
  PREFILL: 100ms (TTFT)
  DECODE:  每 token 20ms → 200*20 = 4000ms
  总 = 4100ms (TTFT 仅占 2.4%, DECODE 主导)""")
TTFT, TPOT, n = 100, 20, 200
total = TTFT + (n-1)*TPOT
res(f"""典型值(7B 模型, 单卡 H100):
  TTFT  = {TTFT} ms   ← prompt=512 一次性 prefill
  TPOT  = {TPOT} ms   ← 每 token 20ms(7B 在 H100 典型值)
  生成 {n} token 总时延:
    = {TTFT} + {n-1}*{TPOT} = {total} ms
    TTFT 占比 {TTFT/total*100:.1f}%
    DECODE 占比 {(n-1)*TPOT/total*100:.1f}%""")
mea("""优化优先级: DECODE > PREFILL,因为占大头。
DECODE 优化: KV cache 量化 / FlashDecoding / speculative / batch。
PREFILL 优化: chunked-prefill(把长 prompt 切成多块)。""")

# --- 4. 混合调度:prefill + decode 共存 ---
hdr(4,TOTAL,"Prefill-Decode 共存与调度")
why("""真实服务多个请求时,prefill 和 decode 必须在同一 GPU 上轮转。
问题:prefill(算力密集) 和 decode(访存密集) 计算模式冲突。
  - 同时跑: prefill 长尾让 decode 等待 → TPOT spike
  - 分开跑: prefill 单独一阵,decode 单独一阵(iteration-level)
现代引擎(vLLM/SGLang):iteration-level 调度,每轮先 decode 再 prefill。""")
res("""调度策略对比:
  策略                TTFT      TPOT   吞吐     复杂度
  串行(简单)          高        低     低       简单
  持续 batching       中        中     中       中
  Iteration-level      中        中     高       中
  Chunked-prefill     低        中     高       较复杂
  Disaggregation      极低      极低   极高     复杂(2 类节点)""")
mea("""2024+ 趋势: Disaggregated serving — prefill 和 decode 跑在不同节点,
  通过 KV transfer 桥接。优势:各自独立优化,decode 节点不会被打断。
  vLLM V1 / SGLang 都支持,DeepSeek-V3 的部署就是这模式。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:PREFILL 是把整个问题一次性\"算清楚\",DECODE 是\"一字一字往外吐\"。
  PREFILL 算力密集、DECODE 访存密集,所以优化策略不同。
- 熟手:TTFT 反映 prefill,TPOT 反映 decode;总时延 90%+ 来自 decode;
  iteration-level 调度 + chunked-prefill 是现代引擎标配;disaggregation 是新趋势。
【进阶】vLLM/SGLang 真机开 chunked-prefill,看 TTFT 和 TPOT 变化。
EOF
echo "############################################################"
