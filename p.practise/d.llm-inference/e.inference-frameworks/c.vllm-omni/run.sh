#!/bin/bash
# ============================================================
# 实验: c.vllm-omni
# 说明: vLLM-Omni 架构:多模态推理扩展、模态接入点、与 vLLM 的复用关系
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# vLLM-Omni = vLLM 的多模态扩展,处理图像/音频/视频输入。
# 核心思想: 在 vLLM 调度框架上,加\"模态预处理\"和\"模态编码器\"
#   - 文本 token: 走原 vLLM 路径
#   - 图像 token: vision encoder → projector → patch token 插入文本
#   - 音频 token: audio encoder → token 插入
#   - 视频 token: 视频分帧 → 多图像 → 多 token
# 关键设计:
#   1. 模态注册机制 (类似 LLaVA/Qwen-VL)
#   2. KV cache 多模态共享
#   3. 多模态 prefill 单独调度
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: c.vllm-omni | 多模态推理:图像/音频/视频接入"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 整体架构 ---
hdr(1,TOTAL,"vLLM-Omni 整体架构")
why("""vLLM-Omni = vLLM + 多模态适配层""")
res("""┌────────────────────────────────────┐
  │ HTTP/RPC (OpenAI 兼容 + 多模态字段) │
  │ Modality Router (判断输入类型)       │
  │ Tokenizer (text + image)            │
  │ Scheduler (continuous batching)     │
  │ Engine                                │
  │   ├── Text Decoder (复用 vLLM)      │
  │   ├── Vision Encoder (CLIP/InternVL) │
  │   ├── Audio Encoder (Whisper)        │
  │   ├── Projector (MLP/Q-Former)       │
  │   └── KV cache (PagedAttn 共享)      │
  └────────────────────────────────────┘""")
mea("vLLM-Omni 与主 vLLM 共用调度、KV cache、推理循环。\n  模态相关:encoder + projector 是新加的,vLLM 主流程不变。")

# --- 2. 模态接入点 ---
hdr(2,TOTAL,"4 种模态的接入方式")
why("""不同模态走不同路径:""")
res("""模态     编码器              数据流
  文本    -                   token 直接进 LLM
  图像    ViT/CLIP/SAM         图像 → patch token (256-1024 个) → 插入文本
  音频    Whisper/Wav2Vec     音频 → mel → 编码 → token → 插入
  视频    视频 → 帧 → 图像 → 图像 pipeline
  视频    VideoMAE/TimeSformer 视频 → 时空 patch → token (更高效)""")
mea("""vLLM-Omni 支持的模型 (2024):
  - LLaVA / LLaVA-NeXT (图像)
  - Qwen-VL / Qwen2-VL (图像)
  - InternVL (图像)
  - Qwen-Audio (音频)
  - Video-LLaVA (视频)
  - Llama 3.2 Vision (图像)""")

# --- 3. 性能与显存 ---
hdr(3,TOTAL,"多模态推理:显存与性能")
why("""多模态推理 3 个额外开销:
  1. vision encoder 一次前向
  2. projector 一次
  3. patch token 增多 → 文本 token + 256-1024 image token""")
res("""LLaVA-1.5-7B, batch=4, 单图 336×336:
  组件               显存       时间
  Vision Encoder     2 GB      20ms
  Projector          0.5 GB    5ms
  LLM 文本部分       8 GB      80ms   (prompt = 50 文本 + 576 图像)
  KV cache (4 seq)   6 GB      -
  ──────────────────────────────────
  合计               16.5 GB   ~110ms  (vs 纯文本 7B ~ 50ms)""")
mea("""多模态推理 2-3× 慢于纯文本,主要在 vision encoder + 长 prompt。
  优化:
  - ViT 量化 (INT8 → -50% 显存)
  - 图像分块 + 仅编码相关区域
  - 多图共享 ViT 前向 (batch processing)""")

# --- 4. 与 vLLM 主线的关系 ---
hdr(4,TOTAL,"与 vLLM 主线的关系")
why("""vLLM-Omni = vLLM 的多模态分支,与主线的整合进度:""")
res("""阶段                状态
  2024-Q1           独立项目 (vllm-omni)
  2024-Q3           整合进 vLLM-0.5 (实验)
  2024-Q4           vLLM-0.6 官方支持图像 (LLaVA/Qwen-VL)
  2025+             全模态 (图像/音频/视频) 整合进 vLLM 主线
  趋势              多模态 = vLLM 默认能力""")
mea("""vLLM-Omni 正在合并进 vLLM 主线,2025 后用户无需单独安装。
  早期: vllm-omni 独立仓库
  现在: vllm[multimodal] extras
  未来: vllm 主仓库内置""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:vLLM-Omni = vLLM 加多模态(图像/音频/视频);走 vision encoder →
  projector → patch token 插入文本;性能比纯文本慢 2-3×,2025 整合进 vLLM。
- 熟手:与 vLLM 共用调度和 PagedAttention,模态部分独立;多模态 KV cache
  按 token 算,长 prompt 优化更关键;ViT 量化、批处理多图能省显存。
【进阶】用 vllm-omni 跑 LLaVA-1.5,看多图推理的吞吐和显存;测 vllm[multimodal]
  extras 的 API 兼容性。
EOF
echo "############################################################"
