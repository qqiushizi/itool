#!/bin/bash
# ============================================================
# 实验: a.continuous-batching
# 说明: 连续批处理、调度、吞吐
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 批处理是 GPU 利用率的命门。三种调度:
#   - static batching: 等一批都完成,再处理下一批
#       短板: 短请求被长请求拖累(gpu 空转等)
#   - dynamic batching: 每步组一次 batch (相对老, 每步重 pad)
#   - continuous batching (vLLM 首创):
#       每 decode step 重新组 batch:
#         - 已完成的请求 → 退出
#         - 新来的请求 → 加入
#         - 还在跑的请求 → 继续
#       优势: 短请求不被等,GPU 几乎满载
# 实测: continuous 比 static 吞吐高 2-10×,尤其请求长度差异大时。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: a.continuous-batching | 连续批处理:不停 batch,GPU 满载"
echo "############################################################"

python3 <<'PY'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 三种调度对比 ---
hdr(1,TOTAL,"三种 batching 调度策略")
why("""模拟 4 个请求,长度 10/20/30/40 token,同一 batch 跑 50 步。
static 短板:每步都按最长请求的 KV 算,短请求浪费。
continuous:每步把完成的剔除,新进的加入。""")
# 简化模拟:decode 一步要 1 单位时间,假设 4 卡的 max batch=4
req_lens = [10, 20, 30, 40]
# static:全部 40 步
static_total = max(req_lens) * 4
# continuous:每步只算还活着的请求
# 步 1-10: 4 个都活 → 4 个并行
# 步 11-20: 3 个 (req0 完成)
# 步 21-30: 2 个 (req1 完成)
# 步 31-40: 1 个 (req2 完成)
# 总 token 算力 = 10*4 + 10*3 + 10*2 + 10*1 = 100
cont_tokens = 10*4 + 10*3 + 10*2 + 10*1
static_tokens = 40*4
res(f"""4 个请求,长度 {req_lens}:
  static batching:
    每步算 4 个请求(按最长对齐) × 40 步 = {static_tokens} token-steps
    但 req0,1,2 早早完成,白白跑了 30/20/10 步 = 浪费 {1 - (10+20+30+40)/static_tokens*100:.0f}%
  continuous batching:
    总 token-steps = {cont_tokens}
    节省: {(1 - cont_tokens/static_tokens)*100:.0f}%""")
mea("""上面是简化:实际 GPU 不按 token 算,按 batch×seq 算。
  但思想一样:continuous 让短请求不被长请求拖累,GPU 始终有活干。""")

# --- 2. 模拟一次 mini scheduler ---
hdr(2,TOTAL,"mini scheduler 模拟:每步调度的请求变化")
why("""每步:
  1. 找出已完成的 req (生成到 max_len 或 EOS)
  2. 回收它们的 KV
  3. 从 waiting queue 取新 req 加入
  4. 重新 pad,组成本步 batch""")
np.random.seed(0)
N = 20    # 20 个请求
lens = np.random.randint(5, 50, N)
lens[0:3] = [40, 35, 30]   # 故意加几个长的
max_bs = 8
finished_at = {}
active = []
wait = list(range(N))
t = 0
total_steps = 0
while wait or active:
    bs = len(active)
    if bs < max_bs and wait:
        n = min(max_bs - bs, np.random.randint(0, 3))
        for _ in range(n):
            if wait: active.append(wait.pop(0))
    # advance all
    for r in active:
        lens[r] -= 1
    total_steps += max(1, len(active))
    t += 1
    new_active = []
    for r in active:
        if lens[r] <= 0:
            finished_at[r] = t
        else:
            new_active.append(r)
    active = new_active
    if t > 200: break
res(f"""20 个请求, max_batch=8, 长度随机 5~50:
  完成 {len(finished_at)}/{N} 个请求, 总 step={t}
  每步平均 batch 大小 ≈ {total_steps/t:.1f}
  vs static (全等最长): {max(lens_orig:=np.array([40,35,30]+list(np.random.randint(5,50,17))))} 步 × 20 个 = {max(lens_orig)*20} token-steps
  vs continuous: {total_steps} token-steps
  节省: {(1 - total_steps/(max(lens_orig)*20))*100:.0f}%""")
mea("""注意:连续 batching 不直接减少\"时间\",它减少的是\"被浪费的算力\"。
  在 GPU 上 = 每秒能服务更多请求 = 吞吐更高。""")

# --- 3. iteration-level scheduling ---
hdr(3,TOTAL,"Iteration-level 调度:prefill/decode 共存")
why("""现代 vLLM 调度是 iteration-level:
  每步(scheduler tick):
    1. 先处理 waiting 队列的 prefill(有限制,避免长 prompt 阻塞)
    2. 再处理 running 队列的 decode
  这样 prefill 和 decode 都能持续,decode TPOT 不被 prefill 拖累太久。""")
res("""调度循环:
  while True:
    1. 处理 waiting 中 prefill 请求
       (每 iteration 最多 N 个 prefill, 长 prompt chunked)
    2. 处理 running 中 decode 请求
       (可能全部, 也有调度策略)
    3. 完成 → remove
    4. 等待队列空且 running 空 → 退出""")
mea("""关键参数:
  - --max-num-seqs: 同时跑多少个 seq
  - --max-model-len: 单 seq 最大长度
  - chunked-prefill 长 prompt 分块大小
vLLM V1 + SGLang + TGI 都用类似策略。""")

# --- 4. 性能对比经验值 ---
hdr(4,TOTAL,"吞吐对比(经验值)")
why("""同 7B 模型 + 单 A100 + 64 并发 + 多变 prompt 长度:
  static:        ~800 tokens/s
  dynamic:       ~1500 tokens/s
  continuous:    ~3000-4000 tokens/s
  continuous + chunked-prefill: ~3500-5000 tokens/s""")
res("""7B 模型, 64 并发:
  调度策略              吞吐(tokens/s)   提升
  static batching        800              1.0×
  dynamic batching       1500             1.9×
  continuous batching    3500             4.4×
  + chunked prefill      4200             5.3×
  + prefix caching       5000             6.3×  (system prompt 复用)
  + speculative          6000             7.5×  (decode 加速)""")
mea("""连续 batching 几乎是\"必须\"——不上吞吐低一个数量级。
  叠加 prefix caching + chunked-prefill 是现代 LLM 服务标配三件套。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:连续 batching = 每步重新组 batch,完成的走、新进的来,GPU 不空转;
  短请求不被长请求拖累,吞吐提升 3-5×。
- 熟手:iteration-level 调度 + chunked-prefill + prefix cache 是 vLLM/SGLang
  标配;--max-num-seqs 决定并发上限;调度器复杂度高,bug 容易出死锁。
【进阶】vLLM 实测关/开 continuous batching,看 requests/s 差距。
EOF
echo "############################################################"
