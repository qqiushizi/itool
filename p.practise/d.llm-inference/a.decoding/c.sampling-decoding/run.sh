#!/bin/bash
# ============================================================
# 实验: c.sampling-decoding
# 说明: 采样解码对比、重复惩罚
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# LLM 输出 logits → 概率分布 → 选 token。策略:
#   - greedy: argmax(logits) — 确定性,但重复
#   - top-k:  只在 top-k 个里采样
#   - top-p:  累计概率 ≥ p 的最小集合
#   - temperature: 调整分布锐度
#   - repetition_penalty: 出现过的 token 概率除以 penalty
#   - beam search: 维护 top-B 个序列
# 经验组合: top_p=0.9, top_k=20, temperature=0.7, rep_penalty=1.05
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: c.sampling-decoding | 解码策略对比 + 重复惩罚"
echo "############################################################"

python3 <<'PY'
import numpy as np
np.random.seed(0)
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 各种采样策略 ---
hdr(1,TOTAL,"5 种采样策略对比")
why("""模拟 LLM 输出 logits,看不同策略选出什么 token。""")
logits = np.array([2.0, 5.0, 1.0, 4.0, 0.5, 3.5, 2.5, 4.5])   # 8 个候选
vocab = ["the","cat","is","on","a","mat","with","hat"]
# softmax
def softmax(x): e = np.exp(x - x.max()); return e/e.sum()
p = softmax(logits)
res(f"""logits:    {dict(zip(vocab, logits.round(2)))}
  概率:      {dict(zip(vocab, p.round(3)))}
  greedy:    {vocab[np.argmax(logits)]}    (取最大)
  top-1:     {vocab[np.argmax(logits)]}    (同 greedy)
  top-2:     按概率在 {vocab[1]} 和 {vocab[7]} 中随机
  top-3:     在 {vocab[1]}, {vocab[7]}, {vocab[3]} 中随机
  top-p=0.7: 累计概率 ≥ 0.7 的集合 = {vocab[1]}, {vocab[7]} (累计 0.79)""")
mea("""Top-p(nucleus)比 top-k 更稳:动态选集合,大词表时不容易采到怪词。
实践中: top_p=0.9~0.95, top_k=20~50, temperature=0.6~0.8 是常用 sweet spot。""")

# --- 2. temperature 影响 ---
hdr(2,TOTAL,"Temperature:分布锐度调节")
why("""Temperature T: 把 logits / T 再 softmax。
  T→0:  分布变尖(几乎 greedy)
  T=1:  原分布
  T>1:  分布变平(更随机,易胡言)
  T<1:  分布变尖(更确定,可能重复)""")
logits = np.array([2.0, 5.0, 1.0, 4.0])
for T in [0.1, 0.5, 1.0, 2.0]:
    p = softmax(logits / T)
    p_str = ", ".join(f"{x:.3f}" for x in p)
    res_str = f"T={T}: [{p_str}]  选 {vocab[np.argmax(p)]}"
    print(res_str)
res("""↑ 见上 ↑""")
mea("""T=0.1 时基本只选概率最高的;T=2 时\"cat\"和\"is\"概率接近,易混乱。
  经验:
    数学/代码(求准):  T=0.0~0.2
    通用对话:          T=0.6~0.8
    创意写作/故事:     T=0.9~1.2""")

# --- 3. 重复惩罚:repetition penalty ---
hdr(3,TOTAL,"Repetition penalty:出过的不让它再出")
why("""LLM 解码时容易陷入\"循环重复\"(尤其 greedy + 长上下文)。
  repetition_penalty: 已出现 token 的 logits 除以 penalty
  - penalty > 1: 抑制重复
  - penalty < 1: 鼓励重复(没意义)
  - 不出现过的 token 不变""")
logits = np.array([2.0, 5.0, 1.0, 4.0, 0.5, 3.5, 2.5, 4.5])
generated = [vocab.index("the"), vocab.index("cat"), vocab.index("is")]
penalty = 1.5
new_logits = logits.copy()
for idx in generated:
    if new_logits[idx] > 0: new_logits[idx] /= penalty
    else: new_logits[idx] *= penalty
p = softmax(new_logits)
res(f"""原始:  {dict(zip(vocab, logits.round(2)))}
  生成历史: {generated} = {[vocab[i] for i in generated]}
  penalty={penalty} 后:  {dict(zip(vocab, new_logits.round(2)))}
  原选: {vocab[np.argmax(logits)]}    惩罚后: {vocab[np.argmax(new_logits)]}""")
mea("""Repetition penalty 默认 1.0(不惩罚),1.05~1.1 是 sweet spot。
  太小没用,太大模型开始\"乱说话\"(把正常词也压低)。
  现代推理框架(HF text-generation)直接 --repetition_penalty 1.05。""")

# --- 4. Beam search 与多样性 ---
hdr(4,TOTAL,"Beam search + 多样性:Diverse Beam / Contrastive")
why("""Beam search: 每步保留 B 个最优序列,展开时选 B² 个里取 B 个。
  优势: 全局最优
  缺点: 输出多样性差(都长一样)
  解法:
    - Diverse beam: 强制不同 beam 选不同 token
    - Contrastive search: 选与历史相似度低的 token
    - 多次采样: temperature 1.0 跑 N 次,取最优""")
# 模拟
res("""方法                    质量    多样性    计算量
  greedy                    中      差        1×
  beam=4                    高      差        4×
  beam=4 + diverse          高      中        4.5×
  sampling (T=0.7)          中      高        1×
  sampling + best-of-N      高      中        N×
  contrastive (α=0.5)        中      高        1×""")
mea("""经验:
  - 数学/代码:  beam=4 或 greedy
  - 对话/翻译:  sampling T=0.7
  - 故事/营销:  sampling T=0.9 + rep_penalty 1.1
  - 真正创意:   top_p 0.95 + 多样性 sampling""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:解码 = 选 token;greedy 选最大,采样加随机;top-p/top-k 控制随机范围;
  temperature 控\"创造力\";rep_penalty 防止车轱辘话。
- 熟手:组合 top_p=0.9 + top_k=20 + T=0.7 + rep_penalty=1.05 是默认甜点;
  数学题用 greedy,故事用 T=0.9;beam search 仅在求全局最优时用。
【进阶】同一 prompt 用不同参数跑 5 次,人工评估质量+多样性。
EOF
echo "############################################################"
