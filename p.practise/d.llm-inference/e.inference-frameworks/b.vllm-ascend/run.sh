#!/bin/bash
# ============================================================
# 实验: b.vllm-ascend
# 说明: vLLM-Ascend 架构:昇腾适配层、NPU 算子/通信映射、与 vLLM 主线的关系与演进
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# vLLM-Ascend = vLLM 官方对 Ascend NPU 的适配:
#   - vLLM 主线: 基于 CUDA + NCCL
#   - vLLM-Ascend: 替换为 CANN + HCCL
# 设计哲学: 尽量复用 vLLM 上层逻辑,只改\"硬件相关层\"
#   - 算子: torch_npu 替换 torch.cuda
#   - 通信: HCCL 替换 NCCL
#   - attention: AscendC FlashAttn 替换 CUDA FlashAttn
#   - 模型加载: 权重转换 torch → torch_npu
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: b.vllm-ascend | 昇腾适配层: torch_npu + HCCL + AscendC"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 与 vLLM 主线的关系 ---
hdr(1,TOTAL,"vLLM-Ascend 与主线关系")
why("""vLLM 主线不断更新,vLLM-Ascend 怎么跟?两种策略:""")
res("""策略                  优                劣
  fork-and-patch        简单              难跟上主线
  upstream + 适配层      与主线同步        适配工作量大
  vLLM-Ascend 选择:     upstream 模式, 写\"插件层\"
  
  ┌────────────────────────────────────┐
  │ vLLM 上层(API/调度/PagedAttn)     │  ← 直接复用
  ├────────────────────────────────────┤
  │ vLLM-Ascend 适配层                  │  ← 主要工作
  │  - device: torch_npu / torch.cuda  │
  │  - comm: HCCL / NCCL               │
  │  - flash_attn: AscendC / cuda      │
  │  - paged_attn: 替换底层 kernel      │
  └────────────────────────────────────┘""")
mea("vLLM-Ascend 是 vLLM 官方社区项目,目标是\"上游化\"——\n  减少 fork,大部分代码提 PR 到 vLLM 主线。")

# --- 2. 算子映射 ---
hdr(2,TOTAL,"算子映射:torch.cuda → torch_npu")
why("""torch_npu 是昇腾版 PyTorch,大部分算子一一对应:""")
res("""算子类型                CUDA 实现                NPU 实现
  torch.add               CUDA kernel              torch_npu.add
  nn.Linear               cuBLAS GEMM              Ascend BLAS / Cube
  F.silu                  cuda silu kernel         Ascend silu
  F.softmax               cuDNN softmax            Ascend softmax
  LayerNorm               cuDNN LN                 Ascend LN
  FlashAttn               flash_attn (CUDA)         AscendC FA
  AllReduce               NCCL                     HCCL
  paged_attention         vLLM CUDA kernel          vLLM AscendC kernel""")
mea("vLLM-Ascend 重点适配 3 个算子:\n  1. PagedAttention kernel (CANN 加速)\n  2. FlashAttention (AscendC)\n  3. Quantization (W8A16/W4A16/INT8 KV)")

# --- 3. 性能对比 ---
hdr(3,TOTAL,"性能:Ascend vs CUDA 推理")
why("""vLLM-Ascend 与 vLLM-CUDA 的性能对比(同模型同 batch):""")
res("""LLaMA-3-8B, batch=32, 1024 prompt + 256 generate:
  硬件              吞吐 (tok/s)    TTFT    TPOT
  A100 80G          4500            60ms    25ms
  H100 80G          6800            45ms    18ms
  Atlas 800T A2     4200            70ms    28ms    (Ascend)
  Atlas 900I A3     6500            50ms    20ms    (Ascend)
  
  结论: Ascend 与同期 CUDA 卡性能相当""")
mea("""华为 Atlas 900I A3 (2024) 性能接近 H100;
  Atlas 800T A2 (2023) 性能接近 A100。
  国产替代可行,需熟悉 CANN/HCCL 工具链。""")

# --- 4. 部署差异 ---
hdr(4,TOTAL,"部署差异:昇腾 vs CUDA")
why("""昇腾 vs CUDA 部署区别:""")
res("""步骤              CUDA 生态               Ascend 生态
  驱动              nvidia driver           Ascend driver
  工具链            CUDA Toolkit            CANN Toolkit
  容器              nvidia/cuda:*           ascendhub.huawei.com
  Python 包          torch + xformers       torch + torch_npu
  通信              NCCL                    HCCL
  Profiling         nsys / Nsight           msprof / ascend-toolkit
  显存监控          nvidia-smi              npu-smi""")
res("""vllm-ascend 部署命令:
  pip install vllm-ascend torch_npu
  vllm serve /path/to/LLaMA --device npu --tensor-parallel-size 4
  msprof --output=./prof python3 -m vllm.entrypoints.openai.api_server ...""")
mea("""主要差异:
  1. 镜像: 替换 nvidia/cuda → ascendhub
  2. 包名:  torch → torch_npu
  3. 通信:  NCCL → HCCL (大部分代码无需改)
  4. profiling: nsys → msprof
  5. 错误码/异常: 不同,需查 CANN 文档""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:vLLM-Ascend = vLLM 在昇腾 NPU 上的官方适配;主要工作是把
  torch.cuda / NCCL 换成 torch_npu / HCCL,大部分上层逻辑复用。
- 熟手:upstream 模式,尽量提 PR;关键适配 3 类算子(PagedAttn / FlashAttn
  / 量化);Atlas 900 A3 性能接近 H100;msprof 替代 nsys。
【进阶】在 Atlas 800T A2 上跑 vllm-ascend LLaMA-3-8B,对比 A100 性能。
EOF
echo "############################################################"
