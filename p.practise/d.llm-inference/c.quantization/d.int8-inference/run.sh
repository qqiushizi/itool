#!/bin/bash
# ============================================================
# 实验: d.int8-inference
# 说明: INT8 推理精度/性能权衡
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# INT8 推理 = 权重量化 INT8 + 激活 INT8 动态量化。
# 优势:
#   - 显存: 7B FP16=14G → INT8=7G (-50%)
#   - 吞吐: GEMM 用 INT8 算 (Tensor Core) 加速 1.5-2×
#   - 功耗: ~1/4 (INT8 算力密度高)
# 代价:
#   - 精度: 1-2% 任务掉点 (ppl 涨 0.1-0.3)
#   - 复杂算子 (LayerNorm, Softmax) 难量化, 仍 FP16
# 工程: bitsandbytes / FasterTransformer / TensorRT-LLM
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: d.int8-inference | INT8 推理:精度 + 性能 + 显存"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 显存与吞吐对比 ---
hdr(1,TOTAL,"显存 + 吞吐:FP16 vs INT8")
why("""7B 模型在 A100 上的典型数据:""")
P = 7e9
out = ["  模式      权重     激活     加速   显存  吞吐"]
out.append(f"  FP16      {P*2/1e9:.0f} G    FP16     1.0×   {P*2/1e9+1:.0f}G   1.0×")
out.append(f"  INT8      {P*1/1e9:.0f} G    INT8     1.5×   {P*1/1e9+0.5:.0f}G   1.5×")
out.append(f"  W4A16     {P*0.5/1e9:.1f} G    FP16     1.2×   {P*0.5/1e9+1:.0f}G   1.2×")
out.append(f"  W4A16+INT8KV 3.5 G  INT8     1.6×   {3.5+0.5:.0f}G   1.6×")
res("\n".join(out))
mea("INT8 推理几乎\"白送\" 1.5× 吞吐 + 50% 显存,代价很小。\n  LLM 推理标准配置: W4A16 + INT8 KV cache + 动态 batch。")

# --- 2. 精度影响(模拟) ---
hdr(2,TOTAL,"精度影响:perplexity 与任务准确率")
why("""7B 模型 INT8 量化典型精度损失:""")
res("""指标              FP16      INT8     差值
  WikiText ppl    5.68      5.72     +0.04  (几乎无)
  C-Eval acc      0.45      0.44     -0.01  (几乎无)
  GSM8K           0.31      0.30     -0.01  (几乎无)
  HumanEval       0.30      0.29     -0.01  (几乎无)""")
mea("INT8 精度损失 < 1%,几乎所有任务不掉点。\n  真正损失大的场景: 1) 极小数 (< 7B) 2) 极长输出 (重复 token 放大误差) 3) 数值敏感任务 (math)。")

# --- 3. INT8 GEMM 加速原理 ---
hdr(3,TOTAL,"INT8 GEMM 加速:Tensor Core")
why("""A100 Tensor Core INT8 算力 624 TOPS,FP16 算力 312 TFLOPS。
  INT8 是 FP16 的 2×。但要\"对齐\":
  1. 量化: weight/activation  → INT8
  2. 矩阵乘 (INT8): 累加用 INT32 (避免溢出)
  3. 反量化: INT32 结果 → FP16 输出""")
res("""INT8 GEMM 流程:
  FP16 W,A (8bit 量化) → INT8 → Tensor Core INT8 GEMM → INT32 → FP16 输出
  
  算力对比 (A100):
  FP16:  312 TFLOPS
  INT8:  624 TOPS  (2×)
  FP8:   ~1.2 POPS (4×)""")
mea("INT8 GEMM 加速 2×,但有些算子难 INT8(softmax/layernorm),仍 FP16。\n  实测整模型加速 1.3-1.6× (受慢算子拖累)。")

# --- 4. 推理框架 INT8 支持 ---
hdr(4,TOTAL,"主流推理框架 INT8 支持")
why("""每个框架的 INT8 启用方式:""")
res("""框架              启用方式                       备注
  vLLM              --quantization gptq/awq         加载预量化模型
  TGI               --quantize int8                 bitsandbytes
  TensorRT-LLM      --int8_kv_cache + 量化插件       极致性能
  FasterTransformer int8 模式 + 校准脚本             NVIDIA 官方
  llama.cpp         -t 8 (CPU INT8)                 跨平台
  bitsandbytes      load_in_8bit=True               HF 一行启用
  transformers      load_in_8bit=True               HF 集成 bnb""")
mea("""生产部署:
  - 单卡 GPU 推理: vLLM + GPTQ/AWQ INT4 + INT8 KV
  - CPU 推理: llama.cpp Q4_K_M (混合 INT4/INT6/FP16)
  - 极限吞吐: TensorRT-LLM INT8 + FlashAttn""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:INT8 推理 = 权重 + 激活都量化到 INT8,显存省 50%,吞吐 1.5×,
  精度损失 < 1%,几乎不掉点;vLLM/bnb 1 行启用。
- 熟手:LayerNorm/Softmax 难量化仍 FP16;Tensor Core INT8 算力 2× FP16;
  配合 INT8 KV cache + FlashAttn 是单卡推理黄金配置;生产用 GPTQ/AWQ。
【进阶】用 vLLM 加载 GPTQ INT4 模型,对比 FP16 测 throughput 和延迟。
EOF
echo "############################################################"
