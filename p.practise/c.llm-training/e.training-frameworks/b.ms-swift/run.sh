#!/bin/bash
# ============================================================
# 实验: b.ms-swift
# 说明: ms-swift 架构:魔搭 SWIFT 训练-推理-部署一体化、插件化模型适配、多模态扩展点
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# ms-swift (Scalable lightWeight Infrastructure for Fine-Tuning) 是
# 魔搭社区出品的一体化框架,核心是「插件化」:
#   - Model 插件: 每个模型一个 Model 类(继承 MultimodelMixin)
#   - Template 插件: 每个模型/对话格式一个 Template
#   - Dataset 插件: 注册机制自定义数据
#   - Trainer 插件: 支持 transformers / accelerate / DeepSpeed / Megatron
# 优势对比 LF:
#   - 多模态扩展点更全(图像/音频/视频)
#   - 与 ModelScope 生态深度绑定(模型/数据集 registry)
#   - 推理/部署/评测全链路(llm-utils, sglang-deploy, vllm-deploy)
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: b.ms-swift | 插件化微调框架,训练-推理-部署一体化"
echo "############################################################"

python3 <<'PY'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 整体架构 ---
hdr(1,TOTAL,"ms-swift 整体架构")
why("""3 大块:
  1. 训练(ms-swift sft/rlhf):基于 transformers/DeepSpeed/Megatron
  2. 推理(swift infer / sglang-deploy / vllm-deploy):本地/服务化
  3. 评测(swift eval):MMLU/CEval/MT-Bench""")
res("""模块                功能                         对应命令
  ms_swift           训练核心                     swift sft
  ms_swift.infer     推理(CLI/SDK)               swift infer
  ms_swift.deploy    部署(gradio/openai兼容)     swift deploy
  ms_swift.eval      评测                         swift eval
  ms_swift.ui        WebUI(类似 LLaMA Board)      swift web-ui
  ModelScope         模型/数据集 registry         modelscope download""")
mea("""\"训练-推理-部署\"一体化是 ms-swift 最大卖点:同一个 yaml 既能训也能
直接 deploy 到 sglang/vllm。LF 也能 deploy 但没那么顺。""")

# --- 2. 插件化模型接入 ---
hdr(2,TOTAL,"插件化:一个模型 = 一个 Python 文件")
why("""ms-swift 接入一个新模型只需 2 步:
  1. swift/model/  下新建 <model_name>.py
  2. 继承 Model 类,实现 register_model / patch_model
这样不污染 transformers 源码,纯插件化。""")
res("""ms-swift/swift/model/
  ├── llm/           # 纯文本 LLM
  │   ├── qwen.py
  │   ├── llama.py
  │   ├── internlm.py
  │   └── ...
  ├── mllm/          # 多模态 LLM
  │   ├── qwen_vl.py
  │   ├── internvl.py
  │   └── ...
  └── utils.py       # 基类 + 工具""")
mea("""这种\"每个模型一个文件\"的设计优点:
  - 新模型接入只要 200~500 行
  - 升级 transformers 时只影响基类
  - 自定义模型可放自己的 plugin 目录
对比 LF 的\"模型注册表 dict\": 更易扩展,但新模型代码量大点。""")

# --- 3. 多模态扩展点 ---
hdr(3,TOTAL,"多模态:从纯文本扩展到 vision/audio/video")
why("""ms-swift 的多模态抽象:
  - Tokenizer:文本 + 图像/音频 tokenizer
  - Template:把多模态消息渲染成 token 序列(图像 patch token)
  - MultimodelMixin:实现图像/音频 encoder → projector → LLM
  - Dataset:支持图文对、视频帧、音频+文本""")
res("""ms-swift 支持的模态组合:
  模态                  代表模型               任务
  文本                  LLaMA/Qwen            SFT/DPO
  图文                  Qwen-VL / InternVL    多模态 SFT
  视频                  VideoLLaMA            视频理解
  音频                  Qwen-Audio            语音 SFT
  图像生成              Stable Diffusion       LoRA
  语音合成              CosyVoice             微调""")
mea("""多模态微调的关键:图像 encoder 冻结,只训 projector + LLM 部分;
QLoRA 还能把 LLM 部分降到 4bit。ms-swift 一行 --freezen_vit 即可。""")

# --- 4. 训练-部署闭环 ---
hdr(4,TOTAL,"训练-部署闭环:一个 yaml 走完全程")
why("""ms-swift 的杀手锏:同一个训练配置可以直接 deploy。
sft 完之后:
  swift deploy --model_checkpoint xxx --deploy_model True
  → 起 vLLM / SGLang 服务
  → OpenAI 兼容 API
  → 立即在业务上接""")
res("""流程:
  1. swift sft --model qwen-7b --dataset alpaca_zh --output_dir ./out
  2. swift export --ckpt_dir ./out --to awq     (可选量化导出)
  3. swift deploy --ckpt_dir ./out --infer_backend vllm
  4. 业务方:curl http://host:8000/v1/chat/completions""")
mea("""端到端约 1-2 小时(7B 微调 + 部署),比 LF 多模型转换那步省心。
生产环境:用 ms-swift 训 → ms-swift 部署 → 业务接 OpenAI API。
  国产硬件支持:swift 支持 Ascend NPU(部分算子走 torch_npu)。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:ms-swift = 魔搭出品的\"训练+推理+部署\"一体化;插件化模型接入,
  多模态扩展点比 LF 完整;一个 yaml 训完直接 deploy。
- 熟手:Trainer 后端可选 transformers/DeepSpeed/Megatron;多模态 LoRA
  冻结 vision encoder 是常用技巧;推理后端 vLLM/SGLang 切换无侵入。
【进阶】跑一个 qwen-vl-chat 的 LoRA 微调,导出 vllm 部署。
EOF
echo "############################################################"
