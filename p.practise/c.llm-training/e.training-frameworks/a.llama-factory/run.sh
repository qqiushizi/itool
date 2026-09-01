#!/bin/bash
# ============================================================
# 实验: a.llama-factory
# 说明: Llama-Factory 架构:统一微调框架、数据模板/数据流、WebUI、模块化的 LoRA/全参/DPO/RLHF 支持设计
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 微调框架的核心问题:
#   1. 怎么把任意数据集(token/对话/多模态)转成统一 batch?
#   2. 怎么把任意模型(LLaMA/Qwen/GLM/Baichuan)挂上同一套训练 loop?
#   3. 怎么切换 LoRA/全参/DPO/RLHF/奖励建模?
# Llama-Factory 思路:
#   - 数据侧: Template + Dataset 抽象,内置 100+ 模板
#   - 模型侧: patcher 动态挂 LoRA/全参,内置 200+ 模型
#   - 训练侧: 统一 Trainer(基于 transformers),覆盖 SFT/DPO/PPO/KTO/IPO/RM
#   - 入口: CLI / WebUI(LLaMA Board)/ Python API
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: a.llama-factory | 统一微调框架:数据/模型/训练三件套"
echo "############################################################"

python3 <<'PY'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 框架定位 ---
hdr(1,TOTAL,"Llama-Factory 在生态中的位置")
why("""LLaMA-Factory 定位:\"单卡友好 / 模板丰富 / 上手快\"的 SFT 框架。
类似但侧重不同的框架:
  - HuggingFace transformers Trainer:最底层,灵活但要写 loop
  - LLaMA-Factory:封装好,内置数据/模型/训练方法,适合快速实验
  - ms-swift:魔搭出品,多模态扩展点更全
  - MindSpeed-LLM:昇腾专用,大模型 + 多模态
  - trl:HuggingFace 官方 RLHF/DPO 库
  - axolotl:单文件配置,适合极简 SFT
  - LLaMA-Factory ≈ 上述三者的\"中文社区合订本\"""")
res("""框架对照:
  框架           易用  多模态  分布式  RLHF/DPO  硬件适配
  transformers   低    中       强      中        全
  LLaMA-Factory  高    中       中      高        CUDA/MPS/昇腾(部分)
  ms-swift       高    高       中      中        CUDA + 部分国产
  trl            中    低       中      极高      CUDA
  axolotl        高    中       中      低        CUDA
  MindSpeed-LLM  中    高       强      中        昇腾""")
mea("""选型逻辑:
  - 单卡/小规模 SFT: LLaMA-Factory,上手最快
  - 大规模预训练: Megatron-LM / MindSpeed-LLM
  - 纯 RLHF: trl
  - 多模态: ms-swift / MindSpeed-MM
  - 国内昇腾: MindSpeed 系列""")

# --- 2. 数据流:Template + Dataset ---
hdr(2,TOTAL,"数据流:Template 如何把\"对话\"变 token 序列")
why("""每种模型有自己偏好的对话格式(Qwen/LLaMA/ChatGLM 各不同)。LF 用
\"Template\"把多轮对话渲染成字符串,再用 tokenizer 转 token。
例如 Qwen: <|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n
这种\"格式字符串\"就是 Template。""")
templates = {
    "qwen": "<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n{assistant}<|im_end|>",
    "llama3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n{assistant}<|eot_id|>",
    "chatglm": "[gMASK]sop<|user|>\\n{user}<|assistant|>\\n{assistant}",
    "yi": "<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n{assistant}<|im_end|>",
}
convo = {"user": "什么是LoRA?", "assistant": "LoRA是低秩适配,通过低秩矩阵近似权重更新。"}
res("对话: 什么是 LoRA? / LoRA 是低秩适配...")
for name, tmpl in templates.items():
    rendered = tmpl.format(**convo)
    print(f"\n  [{name}]:\n    {rendered[:120]}{'...' if len(rendered)>120 else ''}")
mea("""Template 不只是格式,它还影响 loss 掩码:
  - SFT: 只算 assistant token 的 loss(其他位置 -100)
  - Pre-training: 算所有 token
  - 多轮: 多个 assistant 段都算
LF 的 data_collator 自动处理这个 mask,无需手写。""")

# --- 3. 训练方法矩阵 ---
hdr(3,TOTAL,"LF 支持的训练方法与算法")
why("""LF 把\"训练方法\"抽象为 4 个维度,4 选 1 即可启动训练:""")
res("""训练方法        用途                  显存         效果
  full           全参微调              高           基准
  freeze         只训练最后几层        低           一般
  lora           LoRA 低秩             低(主)      良好
  qlora          4bit 基座 + LoRA      极低         良好
  galore         全参 + 低秩梯度       中           接近全参
  dojo           新型 optimizer        -           实验
  ──
  dpo            直接偏好优化          中           替代 PPO
  ipo/kto/rmpo   DPO 变体              中           类似 DPO
  ppo            经典 RLHF            高           强
  rm             奖励模型训练          中           PPO 第一阶段
  kto            Kahneman-Tversky     中           单条数据即可""")
mea("""SFT 起点: lora(显存友好);上线想再涨: qlora → dpo → ppo。
LF 一行切: --stage sft --finetuning_type lora → --stage dpo。
代码层面 LF 内部把这些映射到 peft / trl 的对应类。""")

# --- 4. WebUI(LLaMA Board)----
hdr(4,TOTAL,"LLaMA Board WebUI:零代码微调")
why("""LF 提供 Streamlit WebUI:
  1. 选择模型(下拉,内置 200+)
  2. 选择数据集(内置 50+, 支持上传)
  3. 配置参数(LoRA rank、lr、batch)
  4. 一键启动训练 + 实时看 loss 曲线
  5. 训完直接 chat 评测""")
res("""典型 WebUI 配置项(全部可视化):
  模型: LLaMA-3-8B-Instruct (4bit 加载)
  数据集: alpaca_zh (5 万条中文 SFT)
  微调方式: LoRA  (rank=8, alpha=16)
  学习率:   1e-4
  Epochs:   3
  Batch:    4 + grad accum 4 = effective 16
  量化:     4bit (QLoRA, 用 bitsandbytes)
  FlashAttn: 开
  → 8G 显存即可微调 8B 模型""")
mea("""WebUI 适合\"非工程师\"快速验证想法。生产环境仍走 CLI/API。
真实工程团队多: 写 yaml 配置文件 → llamafactory-cli train cfg.yaml。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:LF = 选模型 + 选数据 + 选训练方法,一条命令搞定 SFT/DPO/PPO;
  Template 自动处理多模型对话格式;WebUI 适合零代码体验。
- 熟手:用 yaml 配置 + CLI 启动;QLoRA 4bit 加载 70B 也能跑(多卡);
  与 trl/peft 配合做高级 RLHF 实验;国产硬件支持看 ms-swift / MindSpeed。
【进阶】clone https://github.com/hiyouga/LLaMA-Factory 跑一个 QLoRA SFT。
EOF
echo "############################################################"
