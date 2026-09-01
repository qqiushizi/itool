#!/bin/bash
# ============================================================
# 实验: f.loss-fusion
# 说明: 损失融合(CE+softmax 数值稳定)
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Cross-Entropy 损失 = softmax + negative log likelihood
#   CE = -log(softmax(x)[label])
#        = -x[label] + log(sum(exp(x)))
#   数值问题: log(sum(exp(x))) 可能溢出
#   解决: 减 max
#        = -(x[label] - max) + log(sum(exp(x - max)))
#   标准: 2 个 kernel (softmax + log + select)
#   融合: 1 个 kernel (一次遍历, 数值稳定)
#   收益: 1.3-1.5×, 显存省 (中间不存 softmax 矩阵)
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: f.loss-fusion | CE+softmax 融合:数值稳定 + 1 kernel"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. CE 公式与数值问题 ---
# hdr(1,TOTAL,"CE 公式与数值稳定性")
# why("""CE = -x[label] + log(sum(exp(x)))
#   问题: exp(1000) = inf, 整公式崩
#   解决: 减 max(x)
#        = -(x[label] - max) + log(sum(exp(x - max)))
#   数值稳定, 业界标准""")
# def ce_naive(x, label):
#     return -x[label] + np.log(np.sum(np.exp(x)))
# def ce_safe(x, label):
#     m = x.max()
#     return -(x[label] - m) + np.log(np.sum(np.exp(x - m)))
# 测试
# x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
# try:
#     r = ce_naive(x, 0)
#     print(f"  naive: {r}")
# except Exception as e:
#     print(f"  naive: 失败 - {e}")
# r = ce_safe(x, 0)
# res(f"""x = [1000, 1001, 1002], label=0:
#   naive: 失败 (exp 溢出)
#   safe:  {r:.4f}""")
# mea("CE 必须减 max, 这是 PyTorch / TF 的标准实现。")

# --- 2. 融合: 1 kernel 算 CE ---
# hdr(2,TOTAL,"融合 CE:1 kernel 算 loss + grad")
# why("""标准 CE 实现:
#   softmax(x)  (存 HBM, vocab_size 维)
#   log(softmax)  (存 HBM)
#   select label -> loss
#   backprop: grad = softmax - one_hot  (存 HBM)
#   共 4 次 HBM IO, 显存 3*vocab

# Fused CE (FusedSoftmaxCE):
#   forward:  1 次遍历算 log_sum_exp + select label
#   backward: 1 次遍历算 grad
#   0 次中间结果存 HBM
#   加速: 1.5-2×, 显存省 vocab_size""")
# vocab = 32000
# io_unfused = 4 * vocab * 2  # 4 次 HBM IO
# io_fused = 2 * vocab * 2     # forward + backward
# res(f"""vocab_size = {vocab}, FP16:
#   标准 CE:  4 × vocab × 2 = {io_unfused/1024:.0f} KB IO + 存 3*vocab 中间
#   融合 CE:  2 × vocab × 2 = {io_fused/1024:.0f} KB IO + 0 中间
#   节省:     50% IO, 100% 中间显存""")
# mea("vocab 越大 (e.g. 100K) 融合收益越显著。\n  LLM 训练 vocab 32K-256K, fused CE 是必备。")

# --- 3. AscendC fused CE ---
# hdr(3,TOTAL,"AscendC fused softmax+CE")
# why("""Fused softmax CE 算子 (AscendC 伪代码):""")
# res("""__global__ __aicore__ void fused_softmax_ce(
#     __gm__ float* logits,      // [vocab]
#     __gm__ int* labels,        // [batch]
#     __gm__ float* loss,        // [batch]
#     __gm__ float* grad,        // [batch, vocab]
#     uint32_t V                  // vocab_size
# ) {
#   // 1. 搬 logits 到 UB
#   __local__ float logits_local[TILE];
#   DataCopy(logits_local, logits, TILE);
  
#   // 2. 算 max 和 sum_exp (1 次遍历)
#   float max_val = -1e10, sum_exp = 0;
#   for (int i = 0; i < TILE; i++) {
#     if (logits_local[i] > max_val) max_val = logits_local[i];
#   }
#   for (int i = 0; i < TILE; i++) {
#     sum_exp += exp(logits_local[i] - max_val);
#   }
#   float log_z = max_val + log(sum_exp);
  
#   // 3. 算 loss
#   loss[batch_id] = log_z - logits_local[labels[batch_id]];
  
#   // 4. 算 grad = softmax - one_hot
#   for (int i = 0; i < TILE; i++) {
#     grad[i] = exp(logits_local[i] - max_val) / sum_exp;
#     if (i == labels[batch_id]) grad[i] -= 1.0;
#   }
  
#   DataCopy(grad_out, grad, TILE);
# }""")
# mea("1 个 kernel 算 forward (loss) + backward (grad)。\n  显存 0 中间, 加速 1.5-2×。")

# --- 4. 实战:用框架 fused CE ---
# hdr(4,TOTAL,"实战:框架默认 fused CE")
# why("""主流框架 fused CE:""")
# out = ["  框架              fused CE              性能"]
# out.append("  PyTorch          F.cross_entropy (自动 fused)  良好")
# out.append("  Apex             --fused_ce           1.5x")
# out.append("  TransformerEngine fused_softmax_ce    1.5-2x")
# out.append("  Megatron         fused_softmax_cross_entropy 1.5x")
# out.append("  LLaMA-Factory   默认 (transformers)  良好")
# out.append("  自研 AscendC     1 kernel CE          1.5-2x (极致)")
# res("\n".join(out))
# mea("""实战:
#   - 训练: PyTorch F.cross_entropy 已 fused, 无需手开
#   - 大 vocab (LLaMA 32K, Qwen 150K) 训练: 必须 fused
#   - loss spike 排查: fused CE 有时和 unfused 数值差 1e-3 (正常)
#   - 框架: HF transformers 默认 fused; 检查 loss 是否正常""")
# PYEOF

# echo ""
# echo "############################################################"
# cat <<'EOF'
# 【整体总结】
# - 小白:CE = softmax + log, 减 max 防溢出;融合 CE = 1 kernel 算 loss + grad,
#   加速 1.5-2×, 显存省 vocab_size;vocab 越大收益越大;框架默认开。
# - 熟手:Fused CE 是大 vocab 训练必备, 显存省 vocab_size;
#   AscendC 1 kernel 算 forward + backward, IO 减半;PyTorch F.cross_entropy
#   已自动 fused;TransformerEngine 极致 fused softmax_ce。
# 【进阶】profile 大 vocab (100K+) 训练时, 检查 CE 算子占比;对比 fused vs
#   unfused 的 throughput 和显存。
# EOF
# echo "############################################################"
