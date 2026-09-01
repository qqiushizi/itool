#!/bin/bash
# ============================================================
# 实验: b.fusion-classification
# 说明: 融合分类:计算密集/访存密集、elementwise/规约/线性融合
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 融合按算子类型分 3 类:
#   1. 线性 + elementwise: Linear + ReLU / Bias / Dropout
#   2. 归约 + 归一化: Mean + Var + Normalize + Scale + Shift
#   3. 多 elementwise: Add + ReLU + Cast + ...
# 按 Roofline 分:
#   - 计算密集: Linear+GEMM 类 (Cube)
#   - 访存密集: elementwise/norm 类 (Vector)
# 融合原则:
#   - 同 Pipe 内的算子易融
#   - 跨 Pipe 难 (Cube + Vector 融合要 AscendC 写)
#   - 跨 Block 难 (归约类)
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: b.fusion-classification | 融合分类:按算子/按瓶颈"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 按算子类型分 ---
hdr(1,TOTAL,"按算子类型分 3 类")
why("融合按被融合算子类型分 3 类:")
out = ["  类型              例子                       难度   收益"]
out.append("  线性+elementwise  Linear+Bias+ReLU+Dropout  易     1.3-1.5x")
out.append("  归约+normalize    Mean+Var+Norm+Scale+Shift 中     1.5-2x (RMSNorm)")
out.append("  多 elementwise    Add+ReLU+Cast+Mul          易     1.3-1.5x")
out.append("  多 GEMM 串        Q@K^T+Softmax@V (Attn)     难     2-5x (FlashAttn)")
out.append("  规约类            Softmax+LayerNorm           中     1.5x")
res("\n".join(out))
mea("最容易的是 elementwise 类, 最难但收益大的是 Attention 类。")

# --- 2. 按 Roofline 分 ---
hdr(2,TOTAL,"按 Roofline 分 2 类")
why("按算子强度分:")
out = ["  类别           I (FLOPs/Byte)   例子                融合重点"]
out.append("  访存密集        < 156             elementwise/norm    减少 IO, 留中间结果")
out.append("  算力密集        > 156             GEMM                减少 launch, 数据复用")
res("\n".join(out))
mea("访存密集的算子融合收益最大 (省 IO)。\n  算力密集的算子融合收益较小, 但 launch 节省仍可观。")

# --- 3. 融合的 4 层级 ---
hdr(3,TOTAL,"融合的 4 个层级")
why("融合按「粒度」分 4 层:")
out = ["  层级                例子                        收益"]
out.append("  1. 算子内            Linear epilogue (bias+relu)  1.2x")
out.append("  2. 算子间            Linear+ReLU+Add (residual)   1.3-1.5x")
out.append("  3. 子图              QKV+Attn+Out 合成 1 kernel   1.5-2x")
out.append("  4. 整图              整 layer 1 kernel            2-5x (激进)")
res("\n".join(out))
mea("层级越高, 收益越大, 但通用性越差。\n  实战: 层级 1-2 几乎必做 (编译器自动), 3-4 需手写。")

# --- 4. 实战选择 ---
hdr(4,TOTAL,"实战:按场景选融合")
why("不同场景适合的融合:")
out = ["  场景              推荐融合                       实现方式"]
out.append("  通用 LLM 推理     框架默认 fused 算子            vLLM/SGLang 内置")
out.append("  极致推理性能      FlashAttn + Fused Linear       手写 AscendC")
out.append("  模型训练          torch.compile + FlashAttn     torch.compile 自动")
out.append("  多模态推理        vision encoder + projector     框架内建融合")
out.append("  旧模型优化        Graph fusion                   TVM / TensorRT")
res("\n".join(out))
mea("""80% 场景用框架默认融合就够:
  - vLLM: RMSNorm/RoPE/SwiGLU 全部 fused
  - SGLang: 整层 fused
  - torch.compile: 通用 fusion
只有性能极限场景才手写 AscendC""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:融合分 3 类(线性+elementwise, 归约+norm, 多 elementwise);
  按 Roofline 分 2 类(访存密集收益大, 算力密集收益小);
  4 个层级: 算子内 / 算子间 / 子图 / 整图, 层级越高收益越大但越难;
  80% 场景用框架默认融合就够。
- 熟手:框架默认融合已覆盖 80%;极致场景手写 AscendC;
  访存密集算子融合重点省 IO,算力密集重点减 launch;
  训练 shape 变化大,融合不如推理稳定。
【进阶】用 torch.compile 模式 fuser/inductor,看 LLM 训练时 fused 了多少
  kernel;对比 fused vs unfused 的 msprof 时间轴。
EOF
echo "############################################################"
