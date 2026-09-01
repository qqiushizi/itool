#!/bin/bash
# ============================================================
# 实验: d.mindspeed-mm
# 说明: MindSpeed-MM 架构:多模态训练框架、模态融合管线、与 MindSpeed-LLM 的复用关系
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 多模态 = 文本 LLM + 其他模态编码器(视觉/音频/视频)+ projector。
# 训练 = 文本 SFT 套路 + 模态对齐阶段。MindSpeed-MM 是 MindSpeed-LLM
# 的多模态扩展,设计:
#   - 模态 encoder 抽象: VisionEncoder / AudioEncoder / VideoEncoder
#   - Projector 抽象: MLP / Q-Former / Resampler
#   - 与 LLM 主干复用: MindSpeed-LLM 的 TP/PP/CP/EP 直接继承
#   - 数据流: 各模态独立 collator → 合并成多模态 batch
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: d.mindspeed-mm | 多模态训练:encoder+projector+LLM 三段式"
echo "############################################################"

python3 <<'PY'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 架构层次 ---
hdr(1,TOTAL,"MindSpeed-MM 在 MindSpeed 体系中的位置")
why("""MindSpeed 家族:
  - MindSpeed-LLM: 纯文本 LLM 训练(基础)
  - MindSpeed-MM:   多模态训练(基于 LLM)
  - MindSpeed-RL:   强化学习(基于 LLM)
三者共享底层加速算子 + 分布式策略,只在上层有差异。""")
res("""MindSpeed 家族关系:
  ┌──────────────────────────────────────────┐
  │ MindSpeed-LLM  (LLM 主干)                 │
  │   ├── MindSpeed-MM  (多模态扩展)          │
  │   ├── MindSpeed-RL  (RL 训练)             │
  │   └── MindSpeed-Core (算子库 / fused)     │
  ├──────────────────────────────────────────┤
  │ torch_npu + Ascend C 算子                 │
  ├──────────────────────────────────────────┤
  │ HCCL                                       │
  └──────────────────────────────────────────┘""")
mea("""这种\"基座 + 扩展\"设计:LLM 部分升级时,MM 自动受益。
代价:多模态特有的优化(visual tokenizer)必须各框架自研,不能直接复用。""")

# --- 2. 三段式架构 ---
hdr(2,TOTAL,"三段式:Encoder + Projector + LLM")
why("""几乎所有多模态 LLM 都是这个结构:
  1. Encoder: 图像/音频 → 特征向量(ViT/Whisper)
  2. Projector: 特征向量 → LLM 词向量空间(MLP/Q-Former)
  3. LLM: 拼接多模态 tokens + 文本 → 输出
训练三阶段:①只训 projector ②放开部分 LLM ③全量微调(可选)""")
res("""典型多模态模型(全部 MindSpeed-MM 支持):
  模型              Encoder          Projector      LLM
  LLaVA             CLIP-ViT         MLP            LLaMA
  Qwen-VL           CLIP-ViT         MLP            Qwen
  InternVL          InternViT        MLP            InternLM
  VideoLLaMA        CLIP-ViT + AvgPool MLP          LLaMA
  Qwen-Audio        Whisper          MLP            Qwen
  CogVLM            EVA-CLIP         MLP            CogLM""")
mea("""Projector 是关键:它是把视觉/音频特征\"翻译\"到 LLM 语言的关键。
轻量(MLP 2 层)→ 训练快但表达力有限;重型(Q-Former)→ 表达力强但慢。""")

# --- 3. 训练阶段 ---
hdr(3,TOTAL,"训练三阶段:对齐 → 微调 → 全量")
why("""多模态训练不能一上来全量微调,会破坏 LLM 已学到的知识。
三阶段:
  1. 模态对齐: 只训 projector(LLM 冻结),让视觉特征映射到文本空间
  2. 指令微调: 训 projector + LLM,数据是图文 QA
  3. 全量微调: 解冻 vision encoder 一起训(可选,需要大量数据)""")
res("""阶段         可训练参数      显存(7B)   数据量    训练时间
  1 对齐        ~10M (proj)    20G        1M 图文对  几小时
  2 指令微调    ~7B            80G        100K QA   1-2 天
  3 全量微调    ~8B            120G+      1M+       数天""")
mea("""阶段 1 必须先做,否则 LLM 看到视觉特征就\"懵了\"。
阶段 3 不一定做:LLaVA-1.5 只做 1+2,效果就很好;只有数据足够才做 3。""")

# --- 4. 显存/性能特征 ---
hdr(4,TOTAL,"显存与性能:vision encoder 才是大头")
why("""直觉上 LLM 应该是显存大头,实际上多模态训练里:
  - Vision encoder (ViT-L/14): ~300M 参数,每图 256 tokens
  - LLM (7B): 7B 参数
看似 LLM 大,但 batch=1 时 7B 推理只 ~14GB,ViT-L + 图像预处理反而占
几 GB。batch 大时 ViT 一次前向算得多,显存涨得快。""")
res("""多模态 batch 显存分配(以 7B LLM + ViT-L/14 为例):
  组件                1×224² 图     4×224² 图     16×224² 图
  ViT-L/14 特征       0.3 GB         1.2 GB         4.8 GB
  LLM 激活            8 GB           30 GB          80 GB
  优化器              28 GB          28 GB          28 GB
  ────────────────────────────────────────────────
  合计                ~36 GB         ~60 GB         ~115 GB""")
mea("""优化:
  - vision encoder 冻结(--freeze_vit)省一半激活
  - 图像分辨率从 448→224 省 4× token
  - 多图训练用 packed samples 节省 padding
  - 图像 token 切到视觉端做 KV cache 复用(复杂场景)""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:多模态 = Encoder(看/听)+ Projector(翻译)+ LLM(理解);
  训练分 3 阶段(对齐/微调/全量),不一次全训。
- 熟手:与 MindSpeed-LLM 共享 TP/PP/CP,MM 只需选 encoder/projector;
  ViT 冻结 + 图像分辨率控制是多模态显存关键;packed samples 提吞吐。
【进阶】跑一个 LLaVA-1.5 风格 7B 微调,冻结 ViT,只训 projector+LLM。
EOF
echo "############################################################"
