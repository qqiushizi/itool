#!/bin/bash
# ============================================================
# 实验: b.mixed-precision
# 说明: AMP、loss scaling、溢出检测
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 训练 = 前向 + 反向 + 优化器更新。三个环节的精度需求不同:
#   - 权重更新 ΔW 需精确 → 长期累加(几次就 0 了)
#   - 反向梯度有大量小值 → 容易下溢
#   - 前向激活值大 → 容易上溢
# 混合精度 (AMP) 思路:
#   1. 维护一份 FP32 master weight
#   2. 前/反向用 FP16/BF16 算(快、省显存)
#   3. 梯度回写到 FP32 master,优化器在 FP32 上做更新
#   4. 遇到梯度太小 → loss scale S,loss *= S,梯度自动 × S,更新前 ÷ S
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  python3 -m pip install --quiet numpy
fi
echo "############################################################"
echo "# 实验: b.mixed-precision | AMP / loss scaling / 溢出检测"
echo "############################################################"

python3 <<'PY'
import numpy as np
np.set_printoptions(precision=6, suppress=True)

def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))

TOTAL = 4

# --- 1. 显存占用对比 ---
hdr(1,TOTAL,"FP32 vs FP16 vs AMP 显存")
why("""一个 7B 模型各精度下显存:
  权重: 7B×4B=28GB(FP32) / 7B×2B=14GB(FP16)
  Adam 优化器状态 (FP32 master + 2×FP32 momentum/variance): 7B×4B×3=84GB
  梯度(FP16 算但 FP32 存)=14GB
  → 纯 FP32 训练 ≈ 112GB;AMP 训练 ≈ 14+84+14 ≈ 112GB(Adam 不省)
  → AdamW + 8bit 优化器(DeepSeek) 可降至 ~14+14+14+7 = 49GB""")
res("""7B 模型单卡估算(单位 GB):
  组件                 FP32   AMP(FP16+FP32 master)   AMP+8bit-opt
  权重                 28     14                       14
  优化器状态(m,v)      56     56                       14  (8bit 量化)
  FP32 master weight   0      28                       28
  梯度                 28     14                       14
  ───────────────────────────────────────────────────────────────
  合计                 112    112                      70""")
mea("""注意:AMP 本身不省 Adam 的显存(m,v 是 FP32 仍要存),所以才有 ZeRO/FSDP
把优化器状态分片到多卡。要再省就上 8bit 优化器 / 4bit(QLoRA)。""")

# --- 2. 模拟 loss scaling 自动调节 ---
hdr(2,TOTAL,"动态 loss scaling:检测溢出→降 scale,稳定→升 scale")
why("""静态 loss scale 太冒险:固定 1024 遇到偶尔大梯度就全 NaN。
NVIDIA Apex / PyTorch native AMP 用「动态 loss scale」:
  - 每 N 步检查 inf/NaN;有 → scale ÷=2,跳过更新
  - 连续 K 步无溢出 → scale *=2,继续放大
我们用一个会偶发梯度的数列模拟这个策略。""")
np.random.seed(0)
scale = 1024.0
grads = np.concatenate([np.random.randn(900)*1e-5, np.array([1e3]*100)])  # 含异常大梯度
loss_hist, scale_hist, skip = [], [], 0
for g in grads[:50]:  # 抽 50 步演示
    gs = g * scale
    has_overflow = not np.isfinite(gs.astype(np.float16))  # FP16 模拟上溢
    if has_overflow:
        scale = max(scale/2, 1.0)
        skip += 1
    else:
        # 模拟「连续 N 步无溢出 → 升 scale」
        if skip == 0 and len(loss_hist) % 5 == 4:
            scale = min(scale*2, 65536.0)
    loss_hist.append(float(g)); scale_hist.append(scale)
res(f"""前 50 步演示(模拟):
  第 1 步:    grad={loss_hist[0]:.3e}  scale={scale_hist[0]}  (有偶发大梯度 → scale 震荡)
  第 10 步:   grad={loss_hist[9]:.3e}  scale={scale_hist[9]}
  第 20 步:   grad={loss_hist[19]:.3e} scale={scale_hist[19]}
  第 30 步:   grad={loss_hist[29]:.3e} scale={scale_hist[29]}
  第 50 步:   grad={loss_hist[49]:.3e} scale={scale_hist[49]}
  共跳过更新 {skip} 次(检测到 FP16 上溢 → 降 scale 重做)""")
mea("""动态 loss scale 关键是「敢升也敢降」:稳定时升上去吃小梯度精度,
一旦溢出立刻降下来保安全。PyTorch GradScaler 内部就是这逻辑。
直觉:loss scale 不是超参数,它是训练稳定性的「自动增益控制」。""")

# --- 3. master weight 的作用 ---
hdr(3,TOTAL,"为什么必须有 FP32 master weight")
why("""把权重 W 转成 FP16 算梯度,又转回 FP16 更新 → 每次小更新 ±1e-3
在 FP16(ε=1e-3)下都可能被吃掉。FP32 master weight 存全精度,
更新后才 cast 回 FP16 给前向用,这样多步累加不会丢精度。""")
W = np.float32(1.0)
g = np.float32(1e-3)
# 纯 FP16 训练:累加 1000 步
W16 = np.float16(1.0)
for _ in range(1000):
    W16 = W16 - np.float16(g)
# AMP:master FP32 + FP16 副本
W32 = np.float32(1.0)
W16b = np.float16(W32)
for _ in range(1000):
    W32 = W32 - g
    W16b = np.float16(W32)
res(f"""学习率 η=1e-3,跑 1000 步,理论新值 = 1.0 - 1000*1e-3 = 0.0
  纯 FP16(无 master):     W16 = {W16}    ← 早早卡在 1.0,更新全被吞
  AMP(FP32 master+FP16 副本): W16 = {W16b} ← 1000 步后稳稳到 0""")
mea("""这就是为什么「没有 FP32 master weight 的 AMP」会训不动:
前 500 步 ΔW 太小被 FP16 吃掉,后 500 步又下溢——模型完全没动。
master 权重是混合精度的"现金",FP16 副本只是"口袋里的零钱"。""")

# --- 4. 反向梯度自动转 FP32 + 通信压缩 ---
hdr(4,TOTAL,"梯度通信:BF16 算 + FP32 reduce")
why("""多卡训练时,梯度需要跨卡 AllReduce。BF16 通信省一半带宽,
但 reduce 结果在 FP32 才能保证累加精度,所以常见模式:
  - 每卡 BF16 算梯度
  - AllReduce 用 FP32 accumulator
  - 同步后回写 FP32 master
为什么:通信量和累加精度是两个独立目标,BF16/FP8 通信 + FP32 累加是标配。""")
ranks = 4
g_bf16 = [np.random.RandomState(i).randn(3).astype(np.float32) * 1e-3 for i in range(ranks)]
# FP32 通信(假设)
g_fp32_allreduce = np.mean(g_bf16, axis=0)
# BF16 通信 + FP32 累加
g_bf16_quant = [g.astype(np.float16) for g in g_bf16]
g_fp32_acc = np.zeros_like(g_bf16[0])
for g in g_bf16_quant:
    g_fp32_acc += g.astype(np.float32)
g_fp32_acc /= ranks
res(f"""4 卡梯度(均值):
  各自 FP32 通信均值: {g_fp32_allreduce.tolist()}
  BF16 通信 + FP32 累加: {g_fp32_acc.tolist()}
  差异: {np.abs(g_fp32_allreduce - g_fp32_acc).max():.2e}
  通信量: FP32=4×3×4B=48B;  BF16=4×3×2B=24B  (↓ 50%)""")
mea("""带宽减半 + 精度损失极小(<1e-4) → ZeRO-2/3 + BF16 通信是大模型标配。
再激进点:用 FP8 通信(再省一半)就要 amax 缩放,Torch 全局 NCCL/HCCl 都支持。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:AMP = 用小精度算前向+反向,但权重/优化器保留大精度;loss scaling
  解决"小梯度看不到"的问题。
- 熟手:必须 FP32 master weight + 动态 loss scale;BF16/FP8 通信 + FP32
  reduce 是分布式训练的标配;真要省就上 8bit 优化器(DeepSpeed) 或 QLoRA。
【进阶】试运行 bash run.sh 多次观察 scale 走势;用 torch.cuda.amp 看真实日志。
EOF
echo "############################################################"
