#!/bin/bash
# ============================================================
# 实验: c.chunked-prefill
# 说明: 分块预填充、TTFT/TPOT 优化
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 长 prompt (32K+) PREFILL 一次算要十几秒,导致:
#   1. 短请求 decode 被卡住,TPOT spike
#   2. GPU 算力被这一个长请求独占
#   3. 用户 TTFT 不可控
# Chunked-prefill (vLLM V0.4+, SGLang):
#   - 把长 prompt PREFILL 切成小块(e.g. 4K token 一块)
#   - 每块和其他 decode 一起跑一个 iteration
#   - 长 PREFILL 不再独占,decode 持续有算力
# 收益:
#   - TTFT 不再 worst-case (长 prompt 变 8 次短 prefill)
#   - TPOT 不 spike (decode 不被长时间阻塞)
#   - 调度公平,GPU 利用率更平滑
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: c.chunked-prefill | 长 prompt 分块,TTFT/TPOT 双优化"
echo "############################################################"

python3 <<'PY'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 问题:长 prefill 阻塞一切 ---
hdr(1,TOTAL,"长 PREFILL 阻塞问题:TTFT vs TPOT")
why("""某请求 prompt=32K,生成 200 token。
  普通 prefill: 一次性算 32K,耗时 800ms(8 卡 A100 估)
  decode 步: 20ms × 200 = 4000ms
  其他并发请求的 TPOT 全部被卡 800ms(等 prefill 完)
  → 用户看到 TPOT spike = 体感卡""")
res(f"""长 prompt (32K) 一次性 prefill:
  当前请求: TTFT=800ms, 然后 TPOT=20ms
  其他 32 个并发请求: 全部 TPOT spike +800ms
  总体验:\"打字打字,卡 0.8 秒,又打字\"
  SLO 违约率: 高""")
mea("""问题的核心:prefill 和 decode 计算模式不兼容,放一起就慢。
  chunked-prefill 思路:不要\"一锤子买卖\",把 32K 拆成 8×4K。
  8 个 iteration 各 100ms,decode 也分到这 8 步里跑。""")

# --- 2. chunked prefill 机制 ---
hdr(2,TOTAL,"chunked prefill:每步只算一块")
why("""vLLM --enable-chunked-prefill 行为:
  每 iteration:
    1. waiting 队列选 prefill 请求(限块大小, e.g. 4096 token)
    2. running 队列所有 decode 请求
    3. 拼成 1 个 mixed batch,统一跑
  同一 GPU kernel,无额外调度成本""")
res("""iteration 序列(prompt=32K, chunk=4K):
  iter 1: prefill 4K + decode 32 req
  iter 2: prefill 4K + decode 32 req
  ...
  iter 8: prefill 4K + decode 32 req
  iter 9: decode 32 req (prefill 完)
  iter 10: ... 第一批 decode 完
  → prefill 总耗时还是 800ms(分 8 次),但 decode 不再被卡 800ms""")
mea("""每步 100ms 混合 vs 一次 800ms 独占,后者 decode 集体等 800ms。
  chunked 后 decode 一直在跑,只是每步 100ms 里多担一点 prefill 算力。
  体感:TTFT 从 800ms 变 100ms(首块即可生成),TPOT 几乎不 spike。""")

# --- 3. TTFT 实际分布 ---
hdr(3,TOTAL,"TTFT 实际分布对比")
why("""设 prompt 长度服从 1K~32K 均匀分布,chunked vs unchunked:
  unchunked:  TTFT ≈ P50=400ms, P99=800ms
  chunked:    TTFT ≈ P50=100ms, P99=100ms
因为 chunked 的 TTFT = \"首块 prefill 时间\",与总长度无关""")
res("""prompt 长度均匀分布 1K~32K, prefill 速度 50ms/K token:
  策略       P50 TTFT    P99 TTFT    P99.9
  unchunked  400ms       800ms       800ms
  chunked    50ms        50ms        50ms   (每块 1K)
  chunked    100ms       100ms       100ms  (每块 4K)""")
mea("""chunked 让 TTFT 与 prompt 长度解耦,这是稳定性巨大提升。
  缺点:总 prefill 时间仍 = 800ms(8 × 100),但\"全跑完\"对用户不可见。
  用户体验:输入长 prompt 也能\"秒出第一字\"。""")

# --- 4. 实践 ---
hdr(4,TOTAL,"vLLM 启用方法 + 副作用")
why("""vLLM 启用:
  vllm serve --enable-chunked-prefill --max-num-batched-tokens 4096
  --max-num-batched-tokens: 每 iteration 最多 token 数
SGLang: 默认开 chunked-prefill,无需开关。""")
res("""配置项                    效果                推荐
  --enable-chunked-prefill  启用                 True(长 prompt 场景)
  --max-num-batched-tokens  每 iter token 上限    4096-8192
  --max-num-seqs            每 iter 序列上限     256
  副作用:
    - 总 prefill 时间略增(每次有 kernel launch 开销)
    - decode 每 iter 算力减少(要给 prefill 让位)
  但用户体验提升远大于理论性能损失""")
mea("""经验:除非 prompt 都 < 2K(无意义开),否则必开。
  chunked-prefill + continuous batching + prefix cache = 现代 LLM 服务三件套。
  2024+ 进一步: disaggregation 把 prefill 节点独立出来,无需妥协。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:长 prompt 一次性 prefill 会卡住所有其他请求;chunked-prefill 把
  长 prompt 切成小块混进 decode,TTFT 从秒级降到 100ms 内,体感顺滑。
- 熟手:--max-num-batched-tokens 是关键调参;chunked + continuous +
  prefix cache 是现代 LLM 服务标配;SGLang 默认开, vLLM 一行启用。
【进阶】vLLM 实测 8K/32K/128K prompt 长度下,开/关 chunked-prefill 的 TTFT。
EOF
echo "############################################################"
