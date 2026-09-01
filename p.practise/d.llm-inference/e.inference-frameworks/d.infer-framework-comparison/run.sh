#!/bin/bash
# ============================================================
# 实验: d.infer-framework-comparison
# 说明: 推理框架对比:功能/性能/硬件支持/部署差异、选型决策
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 推理框架选 4 个维度:
#   1. 功能: 文本/多模态/代码/长上下文
#   2. 性能: 吞吐/延迟/显存
#   3. 硬件: NVIDIA/Ascend/AMD/CPU
#   4. 部署: OpenAI 兼容/自研/RPC
# 主流框架 (2024+):
#   - vLLM: 主流,生产首选
#   - SGLang: 学术+工程,RadixAttention
#   - TensorRT-LLM: NVIDIA 极致性能
#   - TGI: HF 官方,易用
#   - llama.cpp: CPU 推理首选
#   - LMDeploy: 国产,功能多
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: d.infer-framework-comparison | 推理框架选型"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 框架横评 ---
hdr(1,TOTAL,"主流推理框架横评")
why("""2024+ 主流框架,从易用到极致性能:""")
res("""框架           易用  性能  功能  硬件适配       维护方
  vLLM           高    高    全    CUDA/Ascend     社区+UCB
  SGLang         中    极高  全    CUDA            社区+Stanford
  TensorRT-LLM   低    极高  较全  NVIDIA          NVIDIA 官方
  TGI            高    中    中    CUDA            HF 官方
  llama.cpp      极高  中    中    CPU/GPU/Apple   社区
  LMDeploy       中    高    全    CUDA/Ascend     国产商汤""")
mea("""选型建议:
  - 通用生产: vLLM (首选)
  - 极致性能/学术: SGLang
  - NVIDIA 极限优化: TensorRT-LLM
  - 简单部署: TGI
  - CPU/Apple Silicon: llama.cpp
  - 国产化 + 多功能: LMDeploy""")

# --- 2. 性能对比 ---
hdr(2,TOTAL,"同模型同硬件性能对比(经验值)")
why("""LLaMA-3-8B + A100 80G + 64 并发 + ShareGPT 数据:""")
res("""框架              吞吐 tok/s    TTFT P99    TPOT P99
  HF transformers    800          800ms       80ms
  TGI                2400         300ms       45ms
  vLLM               3500         250ms       35ms
  SGLang             3800         230ms       32ms
  TensorRT-LLM       4500         200ms       28ms
  vLLM+speculative   5500         250ms       22ms
  SGLang+speculative 6000         230ms       20ms""")
mea("""vLLM/SGLang 是当前最优工程框架,差距 5-10% 视具体配置;
  TensorRT-LLM 性能最强但部署最复杂;
  HF transformers 几乎是 baseline,简单但慢。""")

# --- 3. 决策树 ---
hdr(3,TOTAL,"选型决策树")
why("""从需求倒推:""")
res("""Q1: 用什么硬件?
  ├── NVIDIA  → Q2
  ├── Ascend  → vLLM-Ascend / LMDeploy
  └── CPU/Apple → llama.cpp

Q2: 极致性能还是易用?
  ├── 极致 → TensorRT-LLM (但要写 config + 编译)
  └── 易用 → vLLM / SGLang

Q3: 主要场景?
  ├── 通用文本    → vLLM (社区, 文档全)
  ├── 复杂 agent  → SGLang (RadixAttention 强)
  ├── 多模态      → vLLM-Omni / LMDeploy
  ├── 长上下文    → vLLM (YaRN/ALiBi 支持好)
  └── 代码补全    → vLLM + FIM 特殊 token

Q4: 部署环境?
  ├── K8s + 微服务  → vLLM (REST API)
  ├── 边缘设备      → llama.cpp / TFLite
  └── 嵌入式        → llama.cpp + 量化""")
mea("""实战经验:
  - 90% 选 vLLM,稳
  - 长上下文 + agent 选 SGLang
  - NVIDIA 内部 + 极致性能 选 TensorRT-LLM
  - 国产化 选 vLLM-Ascend / LMDeploy
  - CPU / 边缘 选 llama.cpp""")

# --- 4. 国产化适配 ---
hdr(4,TOTAL,"国产硬件 + 框架对应")
why("""国产硬件推理框架适配情况:""")
res("""硬件              推荐框架        状态
  Ascend 800T/900I  vLLM-Ascend     成熟 (华为官方)
  昇腾              MindIE          华为官方, 集成度高
  昇腾              LMDeploy        商汤, 性能好
  寒武纪            Cambricon      寒武纪自研, 文档少
  海光 DCU          vLLM (有限)     实验性
  摩尔线程 MTT     vLLM (适配中)   实验性""")
mea("""国产化路径:
  - 主流选择: 昇腾 + MindIE / vLLM-Ascend
  - 性能优先: LMDeploy (商汤在 Ascend 上优化好)
  - 兼容老: 寒武纪 (自家框架, 难上手)
  - 现状: 昇腾生态最完整, 其他硬件建议先验可行性
  - 趋势: 2024+ 国产推理框架快速补齐,与 NVIDIA 差距缩小""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:90% 场景选 vLLM (生产首选);SGLang 偏学术+工程;TensorRT-LLM 极致
  性能但复杂;llama.cpp 跑 CPU/Apple;国产选 vLLM-Ascend 或 LMDeploy。
- 熟手:用决策树快速选;性能差距 5-10% 选框架 5 分钟集成更重要;
  国产硬件昇腾生态最完整,其他硬件先验可行性;OpenAI 兼容 API 是标配。
【进阶】同一模型 vLLM / SGLang / TensorRT-LLM 三家部署,用 guidellm 压测对比;
  再在 Atlas 900 上跑 vllm-ascend 看国产性能。
EOF
echo "############################################################"
