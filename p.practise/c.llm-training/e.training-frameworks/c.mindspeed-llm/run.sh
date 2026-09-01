#!/bin/bash
# ============================================================
# 实验: c.mindspeed-llm
# 说明: MindSpeed-LLM 架构:基于 Megatron-LM 的昇腾训练加速、并行策略封装、HCCL/算子适配层
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 大模型(>10B)训练无法单卡完成,需要:
#   1. 模型并行:TP/PP/EP 把模型切到多卡
#   2. 数据并行:DP/DDP/ZeRO 把数据切到多卡
#   3. 流水线调度:1F1B / interleaved 减少气泡
# NVIDIA 生态用 Megatron-LM + NCCL;昇腾生态 = MindSpeed-LLM + HCCL。
# MindSpeed-LLM 在 Megatron-LM 基础上:
#   - 替换通信原语: NCCL → HCCL(昇腾集合通信库)
#   - 算子适配: GEMM/LayerNorm/RMSNorm/RotaryEmbedding → Ascend C 高性能实现
#   - 加速插件: MindSpeed-LLM 自研 FlashAttn/Rotary/MoE 算子
#   - 兼容 Megatron API: 大部分 Megatron 脚本可直接跑
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: c.mindspeed-llm | 昇腾 LLM 训练:Megatron + HCCL + 加速库"
echo "############################################################"

python3 <<'PY'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 整体定位 ---
hdr(1,TOTAL,"MindSpeed-LLM 定位:Ascend 上的 Megatron-LM")
why("""NVIDIA GPU 上训 70B/175B 用 Megatron-LM + NCCL;
Ascend NPU 上训 70B/175B 用 MindSpeed-LLM + HCCL。
两者 API/模型定义兼容,改 --backend 就能切。""")
res("""层次结构:
  ┌──────────────────────────────────────────┐
  │ MindSpeed-LLM (本框架)                    │
  │   - 自研 FlashAttn / Rotary / MoE 算子   │
  │   - TP/PP/EP/CP/Zero 加速策略            │
  ├──────────────────────────────────────────┤
  │ Megatron-LM (核心, NVIDIA 原作)          │
  │   - 并行策略(改自研)                    │
  │   - Transformer 块定义                   │
  ├──────────────────────────────────────────┤
  │ PyTorch + torch_npu (Ascend 适配)        │
  │   - torch_npu 把 torch.* 算子映射到 NPU  │
  ├──────────────────────────────────────────┤
  │ HCCL (昇腾集合通信, 类比 NCCL)            │
  │   - AllReduce/AllGather/ReduceScatter    │
  ├──────────────────────────────────────────┤
  │ Ascend NPU (Atlas 800/9000)             │
  └──────────────────────────────────────────┘""")
mea("""这套分层好处:换硬件不用重写模型,只换 adapter。
劣势:NCCL 有的特性(比如 GIN),HCCL 不一定有,要等版本。""")

# --- 2. 加速插件速查 ---
hdr(2,TOTAL,"MindSpeed 自研加速插件")
why("""MindSpeed 在 Megatron 之上做了 30+ 性能优化,通过 --开关 启闭:""")
res("""开关(选摘)               作用                     典型收益
  --use-flash-attn         Flash Attention            1.5-2× 长序列
  --use-fused-rmsnorm      RMSNorm 融合               5-10%
  --use-fused-rotary       RoPE 融合                 3-5%
  --use-fused-swiglu       SwiGLU 融合               2-4%
  --use-distributed-opt    分布式优化器               1.2×
  --swap-attention         注意力卸载到 CPU          显存 ↓
  --recompute-*            各种粒度重算               显存 ↓
  --use-nd-matmul          Ascend Cube 大 GEMM       算力 ↑
  --use-mc2                矩阵通信融合(MatMul+AllReduce) 通信 ↓""")
mea("""实战配置:在 yaml 里把这些开关配上,跑 prof 验证。一般 70B 训练
开满所有 fused 算子 + flash-attn 加速比 1.4-1.8×。""")

# --- 3. 并行策略组合 ---
hdr(3,TOTAL,"并行策略:TP×PP×DP×CP×EP 怎么排")
why("""单个 70B 训练需要组合多种并行:TP 切 matmul,PP 切层,DP 切 batch,CP 切 seq。
常用组合(以 64 卡 NPU 训 70B 为例):
  4-way TP × 8-way PP × 2-way DP = 64 卡
  4-way TP × 4-way PP × 2-way DP × 2-way CP = 64 卡 (长序列)""")
res("""典型 70B 训练配置:
  模型       训练 token  硬件            并行策略
  LLaMA-70B  1T         8×8 NPU(64)   TP=4 PP=8 DP=2
  Qwen-72B   2T         16×8 NPU(128) TP=8 PP=8 DP=2
  DeepSeek-V3 14T        数百卡         TP=8 EP=32 PP=2
  7B/13B     <1T        1×8 NPU(8)    TP=1 PP=1 DP=8 (纯 DDP)
  7B/13B 长序  32K seq   1×8 NPU(8)    TP=1 PP=1 DP=4 CP=2""")
mea("""选并行策略经验:
  - 7B/13B 单机能装: DDP (TP=1)
  - 30-70B: TP×PP×DP
  - 200B+: 加 EP(专家并行,Moe 友好)
  - 长序列 32K+: 必加 CP
  - 显存不够: 加 ZeRO-1/2/3 (但会通信大)""")

# --- 4. 与 Megatron-LM 兼容度 ---
hdr(4,TOTAL,"兼容度:几乎平替,但有 5 个差异")
why("""MindSpeed-LLM 维持 Megatron-LM 兼容,但有 5 个常见差异要注意:""")
res("""差异点                  Megatron-LM           MindSpeed-LLM
  通信后端                 NCCL(CUDA)           HCCL(Ascend)
  并行后端实现             torch.distributed    torch.distributed + HCCL
  FlashAttn                flash_attn(CUDA)      自研 NPU 实现
  Fused 算子               apex(CUDA)            自研 Ascend C
  checkpoint 格式          torch               torch (兼容)
  tokenizer                HF / 原始            同
  数据格式                 jsonl/bin           jsonl/bin (兼容)
  Megatron 脚本可跑性      100%                  ~80% (差异主要在 fused)""")
mea("""实操:把 Megatron 训练脚本迁到 MindSpeed,通常改 <50 行:
  - 加 import torch_npu
  - 通信从 nccl 改成 hccl
  - 打开 fused 开关
模型 checkpoint 可直接用,HuggingFace 转换也兼容。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:MindSpeed-LLM = Ascend 上的 Megatron-LM,API/模型定义兼容,
  切到昇腾只要改通信后端 + 开 fused 加速开关。
- 熟手:70B 训练要 TP×PP×DP 组合,长序列加 CP,大 MoE 加 EP;自研 fused
  算子 + flash-attn 是性能来源;msprof 调优看 Cube/Vector 占比。
【进阶】在 8 卡 Atlas 800 上跑一个 LLaMA-7B 训练,开全 fused 对比加速。
EOF
echo "############################################################"
