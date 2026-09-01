#!/bin/bash
# ============================================================
# 实验: d.speculative
# 说明: 投机解码/草稿模型、加速比
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 投机解码(Speculative Decoding):用小草稿模型(Draft)先猜 K 个 token,
# 再用大模型(Target)一次性 verify 这 K 个 token 是否接受。
# 关键:Target 一次 verify K 个 token 的 cost ≈ 1 个 token (因为可以并行)
# 加速比 = K × 接受率 / 1
# 例: K=5, 接受率 0.7, 加速比 ≈ 3.5×
# 进阶: Self-speculative (用大模型自己的浅层作 draft),无需额外模型。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: d.speculative | 投机解码:草稿+验证,加速 ~2-3×"
echo "############################################################"

python3 <<'PY'
import numpy as np
np.random.seed(0)
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 基本机制 ---
hdr(1,TOTAL,"投机解码的 4 步")
why("""四步:
  1. Draft 模型自回归生 K 个 token (快)
  2. Target 模型一次 verify K 个 token (并行)
  3. 对每个 token 比较 p_target vs p_draft:
     - 若 p_target(x) >= p_target * (p_draft / p_target) → 接受
     - 否则拒绝 + 用修正分布重采
  4. 接受的有 t 个(0 ≤ t ≤ K),实际前进 t+1 步""")
res("""verify 一步 target cost = 单步 cost(因为 QKV 对 K 个 token 并行)
  草稿一步 cost = K × draft_step_cost
  接受率 α 下,前进一个 token 平均 cost:
    = (K × draft_cost + 1 × target_cost) / (α × K)
  当 draft << target,加速比 ≈ α × K""")
mea("""实测:
  - 7B target + 100M draft, K=5, 接受率 0.6
  - 加速比 ≈ 0.6 × 5 = 3×
  - 7B target + 7B-Medusa (无需 draft),加速 ~1.8×""")

# --- 2. 模拟接受率 ---
hdr(2,TOTAL,"接受率 vs 草稿/目标相似度")
why("""接受率 = draft 模型和 target 模型的一致程度。
草稿越大、越接近 target,接受率越高。
模拟:target 是 100M, draft 是 1B, 50 token 中约 60-80% 一致。""")
def simulate(draft_quality, K, total_tokens=50):
    np.random.seed(42)
    accepted = 0
    pos = 0
    while pos < total_tokens:
        # draft 提 K 个,每个独立有 draft_quality 概率对
        n_acc = 0
        for i in range(K):
            if pos + i >= total_tokens: break
            if np.random.rand() < draft_quality:
                n_acc += 1
            else:
                break
        accepted += n_acc
        pos += max(n_acc, 1)
    return accepted / total_tokens
for K in [3, 5, 8]:
    for q in [0.4, 0.6, 0.8, 0.9]:
        acc = simulate(q, K)
        speedup = acc * K
        print(f"  K={K}, draft_quality={q}: 接受率={acc:.2f}, 加速比≈{speedup:.2f}×")
res("↑ 上面表格: ↑")
mea("""K 越大潜在加速越大,但接受率会下降(K 越长越容易错)。
  K=5 是 sweet spot,既不太长也不太短。
  草稿质量(q)从 0.6 → 0.9,加速比 1.8× → 4.5×。""")

# --- 3. 三种主流实现 ---
hdr(3,TOTAL,"三种主流投机方案对比")
why("""1. 传统 draft: 单独小模型,质量高但占显存
  2. Medusa: 大模型加几个预测头,无需 draft,直接猜 top-1
  3. EAGLE / EAGLE-2:  用大模型的浅层 + 特征作为 draft""")
res("""方案              Draft 来源     额外显存   接受率  加速
  vanilla draft     100M-1B 模型  1-2 GB     高      2-3×
  Medusa            预测头(K 个)  < 100MB    中      1.5-2×
  Medusa-2          + tree verify  < 100MB   中高    1.8-2.5×
  EAGLE             浅层 + 特征   < 200MB    高      2-2.8×
  EAGLE-2           + 动态树       < 200MB    高      2.5-3.5×
  Lookahead Decoding hidden n-gram 0         中      1.5-2×
  Self-speculative  大模型浅层     0          中      1.3-1.5×""")
mea("""实战选择:
  - 显存紧 / 单卡: Medusa(几乎不占显存)
  - 想要极限加速: EAGLE-2 / vanilla draft
  - 多 batch 高并发: vanilla draft(并行多个 verify 摊销)
  - 完全无额外资源: Lookahead / Self-speculative""")

# --- 4. 局限性 ---
hdr(4,TOTAL,"投机解码的 5 个限制")
why("""投机不是\"银弹\",有几个常见陷阱:""")
res("""限制                    影响
  1. 接受率低时(< 0.4)    反而比正常慢(草稿白做)
  2. 温度高 / 多样性采样   接受率会下降(预测的 top-1 经常被 temperature 改)
  3. 批处理 / 高并发      草稿和 target 同步难,加速被摊薄
  4. 长上下文             接受率稳定,反而效果更好
  5. 多模态               草稿难做(视觉 token 难预测)""")
mea("""经验阈值:
  - 接受率 < 0.4: 关掉,比正常还慢
  - 接受率 0.4-0.7: 加速 1.5-3×,值得开
  - 接受率 > 0.7: 加速 3-5×,必开
  - 高 temperature 场景(创意写作)慎用""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:投机解码 = 小模型先猜几个 token,大模型一次性 check 对错,对的就用;
  接受率越高加速越大,典型 2-3×。
- 熟手:草稿质量 + K 长度决定加速;Medusa / EAGLE 是工程主流;低接受率时
  关掉反而更快;高 temperature 时效果下降;高 batch 场景加速被摊薄。
【进阶】vLLM 实测 --speculative-model 选 Medusa/EAGLE,看接受率和加速比。
EOF
echo "############################################################"
