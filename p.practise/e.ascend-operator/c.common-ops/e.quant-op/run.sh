#!/bin/bash
# ============================================================
# 实验: e.quant-op (★新增)
# 说明: Quant / DeQuant 算子: Vector 单元 + scale/cast + per-channel + tiling
# 母目录: e.ascend-operator/c.common-ops
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 量化的"概念与策略"(对称/非对称、scale/zero-point、PTQ)在
#   d.llm-inference/c.quantization 讲;本实验讲"算子怎么落地到硬件"。
# Quant  算子: q = clip(round(x / scale) + zp, qmin, qmax) -> int8
#   拆成 Vector 指令: Div(scale) -> Round -> Add(zp) -> Clip -> Cast(FP->INT)
# DeQuant算子: x = (q - zp) * scale -> fp16
#   拆成 Vector 指令: Cast(INT->FP) -> Sub(zp) -> Mul(scale)
# 两者都是"逐元素、访存密集",走 Vector 单元(4096-bit 宽):
#   INT8 512 元素/cycle, FP16 256 元素/cycle —— 位数越低吞吐越高。
# 在 W8A8 推理里,Quant/DeQuant 紧贴 GEMM:
#   y = DeQuant( Quant(x) @ Quant(w) ) = (xq @ wq) * scale_x * scale_w
# 大权重放不下 UB(256KB),必须 tiling: 搬一块(MTE2)->量化(Vector)->写回(MTE3)。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: e.quant-op | Quant/DeQuant 算子: Vector + scale + tiling"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 5

# --- 1. 算子本质:Cast+scale,走 Vector ---
hdr(1,TOTAL,"Quant/DeQuant = 一串 Vector 指令")
why("""把两个算子拆成 Vector 单元能直接执行的指令序列,
  看清它们和 GEMM(Cube)的本质区别:""")
out = ["  算子      指令序列(Vector)                          单元  位数吞吐"]
out.append("  Quant     Div(scale)->Round->Add(zp)->Clip->Cast   Vector FP16->INT8 256->512/cyc")
out.append("  DeQuant   Cast(INT->FP)->Sub(zp)->Mul(scale)       Vector INT8->FP16 512->256/cyc")
out.append("  MatMul    Cube 点乘                                 Cube  --")
out.append("")
out.append("  共同点: 逐元素、访存密集(I≈1 FLOP/Byte), 瓶颈是搬数据不是算")
out.append("  关键: Quant 让数据变窄(INT8 8bit), 搬运/存储都省一半以上")
res("\n".join(out))
mea("Quant/DeQuant 不是「计算密集」算子, 而是「格式转换+缩放」算子。\n  优化重心在减少 IO、与 GEMM 融合, 而非榨 Cube 算力。")

# --- 2. per-tensor vs per-channel 精度 ---
hdr(2,TOTAL,"per-tensor vs per-channel:离群通道的代价")
why("""权重的不同输出通道(out dim)幅值常差很多(有离群通道)。
  per-tensor : 全局一个 scale = max(|W|)/127, 离群通道把 scale 撑大, 其余通道精度被压缩。
  per-channel: 每个输出通道独立 scale_c, 各通道用满 INT8 动态范围, 误差小很多。
  代价: per-channel 要存 [out] 个 scale(小), 反量化时按通道广播 Mul。""")
np.random.seed(1)
oc, ic = 64, 128
W = np.random.randn(oc, ic).astype(np.float32) * 0.1
W[3] *= 8.0   # 制造一个离群通道
def quant_per_tensor(W):
    s = np.abs(W).max() / 127
    q = np.round(W / s).clip(-128, 127).astype(np.int8)
    return q, s
def quant_per_channel(W):
    s = np.abs(W).max(axis=1, keepdims=True) / 127   # (oc,1)
    q = np.round(W / s).clip(-128, 127).astype(np.int8)
    return q, s
qt, st = quant_per_tensor(W)
qc, sc = quant_per_channel(W)
err_t_max = np.abs(W - qt.astype(np.float32) * st).max()
err_c_max = np.abs(W - qc.astype(np.float32) * sc).max()
err_t_mean = np.abs(W - qt.astype(np.float32) * st).mean()
err_c_mean = np.abs(W - qc.astype(np.float32) * sc).mean()
res(f"""权重 [oc={oc}, ic={ic}], 第 3 通道幅值放大 8x:
  per-tensor  : scale={st:.4f}, 最大误差={err_t_max:.4f}, 平均误差={err_t_mean:.5f}
  per-channel : scale[3]={sc[3,0]:.4f}(大), 其余≈{sc[0,0]:.4f}(小)
                最大误差={err_c_max:.4f}, 平均误差={err_c_mean:.5f}
  最大误差比 = {err_t_max/err_c_max:.1f}x (由离群通道决定, 看不出优势)
  平均误差比 = {err_t_mean/err_c_mean:.1f}x (per-channel 在非离群通道精度高得多)""")
mea("权重量化几乎都用 per-channel(AWQ/GPTQ/W8A8 都是);\n  激活量化因动态变化才用 per-tensor/per-token。")

# --- 3. tiling:UB 放不下大权重 ---
hdr(3,TOTAL,"tiling:分块搬运 + 量化")
why("""UB 只有 256KB。一个 [512,1024] FP16 权重 = 1MB, 放不下。
  做法: 按 tile 切(如 [64,1024]=128KB), 循环:
    MTE2: 从 HBM 搬 INT8/FP16 块进 UB
    Vector: 在 UB 内做 Quant/DeQuant(不落 HBM)
    MTE3: 写回 HBM
  tile 越大, 复用越好, launch 越少; 但受 UB 上限约束。""")
def dequant_tiled(q, scale, tile_oc=64):
    oc, ic = q.shape
    out = np.empty((oc, ic), dtype=np.float32)
    n_tiles = 0
    for s in range(0, oc, tile_oc):
        e = min(s + tile_oc, oc)
        block = q[s:e].astype(np.float32)            # Cast (在"UB"内)
        out[s:e] = block * scale[s:e] if scale.ndim > 1 else block * scale  # Mul
        n_tiles += 1
    return out, n_tiles
Wdq, nt = dequant_tiled(qc, sc, tile_oc=16)
err_tiled = np.abs(W - Wdq).max()
ub_bytes = 16 * ic * 2   # FP16
res(f"""[oc={oc},ic={ic}] INT8 权重, tile_oc=16:
  分块数:        {nt} 块
  每块 UB 占用:  {ub_bytes/1024:.0f} KB (FP16, 装得下 256KB UB)
  分块反量化误差: {err_tiled:.4f} (与不分块一致, 数值无损)
  若整权重一次性: {oc*ic*2/1024:.0f} KB > 256KB, 装不下""")
mea("tiling 不改变数值, 只解决「装不下」和「流水线 overlap」。\n  选 tile 原则: 装得进 UB + 尽量大(减 launch) + 对齐 Cube 向量宽。")

# --- 4. W8A8 推理数据流模拟 ---
hdr(4,TOTAL,"W8A8 数据流:Quant(x)·Quant(w)->DeQuant")
why("""把 Quant/DeQuant 套到一次线性层 y=x@W^T:
  1) xq = Quant(x, sx);  wq = Quant(W, sw_per_ch)
  2) zq = xq @ wq^T            (INT8 GEMM, 走 Cube)
  3) y  = zq * (sx * sw_per_ch)  (DeQuant, 一次 Mul)
  数学等价: y ≈ x @ W^T。看误差与"省在哪"。""")
np.random.seed(2)
m, k, n = 8, 128, 64
x = np.random.randn(m, k).astype(np.float32) * 0.3
Wm = np.random.randn(n, k).astype(np.float32) * 0.1
Wm[5] *= 6.0
y_fp = x @ Wm.T
sx = np.abs(x).max() / 127
xq = np.round(x / sx).clip(-127, 127).astype(np.int8)
sw = np.abs(Wm).max(axis=1, keepdims=True) / 127
wq = np.round(Wm / sw).clip(-127, 127).astype(np.int8)
zq = (xq.astype(np.int32) @ wq.astype(np.int32).T)   # INT8 GEMM 用 INT32 累加
y_q = zq.astype(np.float32) * (sx * sw).ravel()
rel = np.abs(y_fp - y_q).max() / (np.abs(y_fp).max() + 1e-9)
res(f"""[m={m},k={k},n={n}] 线性层:
  FP16 基线 y 范围:    [{y_fp.min():.3f}, {y_fp.max():.3f}]
  INT8 GEMM 累加:      INT32 (防溢出)
  反量化后 y 范围:      [{y_q.min():.3f}, {y_q.max():.3f}]
  最大相对误差:        {rel*100:.2f}%""")
mea("""W8A8 把 GEMM 从 FP16 换成 INT8:
  - 算力: Cube INT8 ≈ FP16 的 2x (位数减半吞吐翻倍)
  - 显存/带宽: 权重+激活省 50%+ (访存密集推理的关键收益)
  - 误差: 通常 <1%, 可接受; 大模型权重对 INT8 不敏感
  反量化的 scale 合并(sx*sw)是省一次 Mul 的小技巧。""")

# --- 5. Ascend 实现 + 分工 ---
hdr(5,TOTAL,"Ascend 实现 & 与概念节的分工")
why("""昇腾侧的落地方式, 以及和概念节(量化策略)的边界:""")
out = ["  层次          接口/实现                           说明"]
out.append("  aclnn 算子    aclnnQuant / aclnnDequant          官方封装, Cast+scale")
out.append("  AscendC 手写  kernel: MTE2->Vector(Quant)->MTE3 tiling + 双缓冲可手动调")
out.append("  框架自动      torch_npu / MindIE 自动插 Quant/DeQuant 围绕 int8 GEMM")
out.append("  融合形态      DeQuant+MatMul+Quant 夹心          见 g.quant-fusion (本节不展开)")
out.append("")
out.append("  分工:")
out.append("    d.llm-inference/c.quantization -> 选哪种量化(symmetric/PTQ/AWQ)、误差度量")
out.append("    本节 e.quant-op               -> 算子在 Vector 上怎么跑、tiling、per-channel")
out.append("    g.quant-fusion                -> 把 DeQuant 塞进 GEMM, 省中间 FP16 落 HBM")
res("\n".join(out))
mea("""实战:
  - 推理框架默认已插好 Quant/DeQuant, 一般不用手写
  - 手写 AscendC 时: per-channel scale 用 [oc] 个 float, Vector 广播 Mul
  - msprof 看 Cast/Mul 占比高 = 量化算子在拖后腿, 应转融合形态""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Quant/DeQuant 是"格式转换+缩放"算子, 走 Vector 单元(不是 Cube);
  在 W8A8 里围着 GEMM:先把 x/W 压成 INT8, GEMM 完再乘回 scale。
- 熟手:权重用 per-channel 量化(离群通道不连累别人);大权重要 tiling 进 UB;
  W8A8 收益在显存/带宽减半 + Cube INT8 吞吐翻倍, 误差通常 <1%;
  scale 合并(sx*sw)省一次 Mul。
【进阶】用 AscendC 写一个 per-channel DeQuant 算子: MTE2 搬 INT8 块 -> Vector
  Cast+广播 Mul(scale) -> MTE3 写回; 用 msprof 看 Vector 利用率, 再与融合形态对比。
EOF
echo "############################################################"
