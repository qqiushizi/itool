#!/bin/bash
# ============================================================
# 实验: f.train-framework-comparison
# 说明: 训练框架对比:定位/适用场景/并行能力/昇腾支持差异、选型决策
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 选训练框架 = 选 4 个维度:
#   1. 任务: SFT / 预训练 / RLHF / 多模态
#   2. 规模: 单卡 / 多卡 / 多机
#   3. 硬件: NVIDIA / Ascend / 其它
#   4. 经验: 工程师能力 → 框架抽象层级
# 用决策树代替\"哪个最好\"问题。
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: f.train-framework-comparison | 训练框架选型决策树"
echo "############################################################"

python3 <<'PY'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 主流框架横评 ---
hdr(1,TOTAL,"主流训练框架横评")
why("""按\"层次\"看:底层(Megatron/DeepSpeed)→ 中层(transformers)→ 上层
  (LLaMA-Factory/ms-swift) → 专项(trl/veRL)。越上层越易用,越下层越灵活。""")
res("""框架            层次     主要场景          硬件适配        学习曲线
  Megatron-LM    底层     100B+ 预训练       NVIDIA          陡
  MindSpeed-LLM  底层     100B+ 训练         Ascend          陡
  DeepSpeed      底层     任意 + ZeRO        NVIDIA/AMD     中
  transformers   中层     SFT/通用           全              平
  LLaMA-Factory  上层     快速 SFT/DPO       CUDA/MPS/部分昇腾 很平
  ms-swift       上层     SFT+多模态+部署    CUDA + 部分昇腾    平
  trl            专项     RLHF/DPO           CUDA            平
  veRL           专项     RLHF/GRPO          CUDA/Ascend    中
  axolotl        上层     单文件 SFT         CUDA            平""")
mea("""通用选型:
  - 0-10B SFT: LLaMA-Factory / ms-swift
  - 10-100B 训练: transformers + DeepSpeed / MindSpeed-LLM
  - 100B+ 训练: Megatron / MindSpeed-LLM + Ascend
  - RLHF: trl (小) / veRL (中大)
  - 多模态: ms-swift / MindSpeed-MM""")

# --- 2. 决策树 ---
hdr(2,TOTAL,"选型决策树")
why("""从问题出发倒推框架,而不是\"哪个最好\":""")
res("""Q1: 任务是什么?
  ├── SFT/预训练 → Q2
  ├── RLHF       → Q5
  └── 多模态     → Q6
Q2: 规模多大?
  ├── <10B  → LLaMA-Factory / ms-swift (单卡 QLoRA)
  ├── 10-70B → transformers + DeepSpeed/MindSpeed
  └── 100B+  → Megatron / MindSpeed-LLM
Q3: 用什么硬件?
  ├── NVIDIA → Megatron / DeepSpeed
  └── Ascend → MindSpeed 系列
Q4: 团队经验?
  ├── 初级  → LLaMA-Factory (一站搞定)
  └── 高级  → transformers + 自研 loop
Q5: RLHF 框架选?
  ├── 小规模(<7B) → trl
  └── 大规模(>7B) → veRL (分布式灵活)
Q6: 多模态选?
  ├── 文本+图像 → ms-swift / MindSpeed-MM
  ├── 文本+音频 → ms-swift
  └── 视频     → MindSpeed-MM""")
mea("""决策树不是\"唯一答案\",而是\"排除明显不适合\"。最终选型还要看:
  - 内部积累(已有 DeepSpeed 经验就别换)
  - 社区活跃度(issue 解决速度)
  - 上手时间(急用选 LLaMA-Factory,长期用 transformers 更灵活)""")

# --- 3. 性能对比(经验值) ---
hdr(3,TOTAL,"同模型同规模性能对比(经验值)")
why("""同一 7B 模型 + 单 8×A100 节点 + bf16 训练,各框架吞吐量
  (经验值,不是 benchmark):""")
res("""框架              加速器    吞吐(tokens/s)  备注
  transformers       8×A100   ~3500           基准
  + DeepSpeed ZeRO-1           ~3800           +9%
  + FlashAttn                 ~5200           +48%  (attn 优化)
  LLaMA-Factory     8×A100   ~5000           -3%  (有少量封装开销)
  ms-swift          8×A100   ~5100           -2%  (类似 LF)
  Megatron-LM       8×A100   ~6000           +72% (TP=2 算力切)
  + FlashAttn                 ~7800           -"
  测速用 Llama-3-8B, seq=4096, bs=32""")
mea("""Megatron 比 transformers 高 70%,主因是 TP 把 matmul 切到多卡算力更足。
  LLaMA-Factory 比 transformers 略低,因为多了一些数据/日志开销。
  实操:\"快\"的代价是\"难\"——Megatron 配置 TP/PP 就要写不少 yaml。""")

# --- 4. 国产硬件支持矩阵 ---
hdr(4,TOTAL,"国产硬件支持矩阵")
why("""2025+ 的国内训练必须考虑昇腾/海光/寒武纪。框架对国产硬件的支持度
  决定能否落地:""")
res("""框架            昇腾 NPU  海光 DCU  寒武纪  摩尔线程
  transformers    部分       部分      无     部分
  Megatron-LM     无         无        无     无
  MindSpeed-LLM   全(主力)  适配中     无     无
  DeepSpeed       部分       无        无     无
  LLaMA-Factory   部分       无        无     无
  ms-swift        部分       无        无     无
  trl             无         无        无     无
  veRL            全         无        无     无
  附 torch_npu    主力        -         -      -""")
mea("""结论:昇腾生态最全,主力走 MindSpeed 系列。
  其它国产硬件(MetaX / Cambricon)目前还不成熟,生产前先验。
  跨硬件方案: OneFlow / 阿里 PAI(屏蔽底层差异)逐渐可用。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:选框架 = 选任务 + 选规模 + 选硬件 + 看团队经验;
  SFT 选 LLaMA-Factory / ms-swift,RLHF 选 trl / veRL,大训练选 Megatron。
- 熟手:用决策树快速排除;性能差距 50-70% 不如\"团队熟练度\"重要;
  国产硬件必走 MindSpeed;框架不是\"最好\"而是\"最合适\"。
【进阶】把 7B SFT 在 transformers / LLaMA-Factory / Megatron 各跑一遍,
  对比真实吞吐和易用度。
EOF
echo "############################################################"
