#!/bin/bash
# ============================================================
# 实验: a.precision-eval
# 说明: 量化前后 perplexity/准确率对比
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 精度评估三件套:
#   1. Perplexity (ppl): 衡量语言模型整体预测能力
#      ppl = exp(loss), 越低越好
#   2. 任务准确率: MMLU/C-Eval/GSM8K/HumanEval
#   3. 业务指标: 实际场景的人工评估 / A/B test
# 量化前后必须做对比,这是上线的必过门槛。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: a.precision-eval | 量化前后精度评估:ppl + 任务"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. Perplexity 概念 ---
hdr(1,TOTAL,"Perplexity: 语言模型好坏的\"温度计\"")
why("""PPL = exp(cross_entropy_loss), 一个 token 上的\"平均选择数\"。
  ppl=10: 模型在 10 个候选里挑
  ppl=100: 模型在 100 个候选里挑 (差)
  ppl=2-5: 优秀 (LLaMA-7B 在 WikiText ~5.7)""")
np.random.seed(0)
# 模拟: 真实 token 概率 vs 均匀
def ppl(probs, true_idx):
    p = probs[true_idx]
    return -np.log(p + 1e-10)
# 完美预测
p = np.array([0.99, 0.005, 0.005])
print(f"  完美预测 ppl = {ppl(p, 0):.3f}")
# 不错
p = np.array([0.5, 0.3, 0.2])
print(f"  不错预测 ppl = {ppl(p, 0):.3f}")
# 烂
p = np.array([0.34, 0.33, 0.33])
print(f"  烂预测 ppl   = {ppl(p, 0):.3f}")
# 平均
p = np.array([0.5, 0.5])
print(f"  二元平均 ppl = {ppl(p, 0):.3f}")
res("""  场景          概率分配        ppl
  完美         [0.99, 0.005, 0.005]   1.01
  不错         [0.5, 0.3, 0.2]        1.82
  随机(均匀)   [0.34, 0.33, 0.33]     2.97
  二元 0.5     [0.5, 0.5]             2.00""")
mea("ppl 是 perplexing 的缩写,意为\"令人困惑\"。ppl 越低,模型越确定。\n  LLaMA-7B WikiText ppl 5.68;量化到 INT4 涨到 5.85,涨 3% 算可接受。")

# --- 2. 量化对 ppl 的影响 ---
hdr(2,TOTAL,"量化对 ppl 的影响(经验数据)")
why("""LLaMA-7B 在 WikiText-2 上的 ppl 经验:""")
out = ["  模式              ppl    涨    备注"]
out.append("  FP16              5.68   -     baseline")
out.append("  INT8 (bnb)        5.72   +0.7% 几乎无损")
out.append("  W4A16 (GPTQ)     5.95   +4.8% 可接受")
out.append("  W4A16 (AWQ)      5.85   +3.0% 略好 GPTQ")
out.append("  INT4 (QAT)       5.78   +1.8% QAT 强")
out.append("  INT3              6.20   +9.2% 不推荐")
out.append("  INT2              7.50   +32%  不可用")
res("\n".join(out))
mea("INT4 是 sweet spot:精度损失 3-5%,显存省 4×。\n  INT3 及以下除非特殊场景,否则掉点太多。")

# --- 3. 任务准确率 ---
hdr(3,TOTAL,"任务级评估:0-shot 准确率")
why("""多个 benchmark 0-shot 准确率,反映\"真实能力\":""")
out = ["  模型         模式      MMLU  C-Eval  GSM8K  HumanEval"]
out.append("  LLaMA-7B    FP16     0.35  0.27    0.14   0.14")
out.append("  LLaMA-7B    INT8     0.34  0.27    0.14   0.13")
out.append("  LLaMA-7B    INT4     0.33  0.25    0.12   0.12")
out.append("  LLaMA-13B   FP16     0.47  0.32    0.18   0.18")
out.append("  LLaMA-13B   INT4     0.45  0.30    0.17   0.17")
out.append("  Qwen-7B     FP16     0.57  0.59    0.51   0.37")
out.append("  Qwen-7B     INT4     0.55  0.57    0.49   0.36")
res("\n".join(out))
mea("""任务准确率掉点 < 2% 通常可接受;
GSM8K (数学) 对量化最敏感,因为需要精确推理;
HumanEval (代码) 也敏感,因为 syntax 严格。
中文任务(C-Eval)受量化影响 < 英文。""")

# --- 4. 业务评估:真实场景 ---
hdr(4,TOTAL,"业务评估:A/B test + 人工抽检")
why("""ppl 和 benchmark 不直接等于业务质量。\n  业务侧要做 A/B test:\n  - 5% 流量跑量化模型, 95% 跑 FP16\n  - 比对: 满意率 / 任务完成率 / 时延 / 成本""")
res("""A/B 评估关键指标:
  业务核心            量化影响
  用户满意度          通常 -0.5% ~ -1%
  任务完成率          1-2% 掉点
  响应延迟            -20% 到 -40% (变快)
  单 token 成本       -50% (省一半)
  GPU 利用率          +30% (算得更满)
  
  经验: 量化上线前 100% A/B test, 至少 1 周""")
mea("""上线流程:
  1. 离线: ppl 涨 < 3% 才进 A/B
  2. A/B: 5% 流量对比 1 周, 看业务指标
  3. 灰度: 50% 流量 2 周
  4. 全量: 没问题再上 100%
  5. 监控: 持续 1 月, 异常回退""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Perplexity(ppl) 衡量语言模型预测能力,越低越好;量化会让 ppl 涨
  一些,INT4 涨 3-5% 算可接受;上线前必做 A/B test。
- 熟手:ppl + 任务 benchmark + 业务 A/B 三件套;ppl 涨 < 3% 进 A/B;
  GSM8K/HumanEval 对量化最敏感,需重点测;灰度发布稳。
【进阶】用 lm-eval-harness 跑 HellaSwag/MMLU/GSM8K 三件套,看量化影响。
EOF
echo "############################################################"
