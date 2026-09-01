#!/bin/bash
# ============================================================
# 实验: g.quant-fusion (★新增)
# 说明: DeQuant+MatMul(+Quant) 反量化夹心融合 (W8A8/W4A8)
# 母目录: e.ascend-operator/e.fusion-operator/b.common-fusion-ops
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# W8A8 推理一次线性层, "不融合"的数据流:
#   HBM 读 INT8 W -> DeQuant -> 写 FP16 W 到 HBM -> 读 FP16 W -> MatMul
#   -> 写 FP16 Z 到 HBM -> Quant -> 写 INT8 Z 到 HBM
#   痛点: 反量化出来的 FP16 权重/激活只是"过路客", 却落了两次 HBM(最贵的访存)。
# "反量化夹心"融合(quant-fusion)的思路:
#   权重以 INT8 常驻 HBM; 每个 tile:
#     MTE2 搬 INT8 块进 UB -> Vector 在 UB 内反量化成 FP16 -> Cube 直接拿它做 GEMM
#   FP16 中间量全程待在 UB, 从不落 HBM。
#   数学上等价: (wq * scale_w) @ x, 只是把 *scale_w 折进 GEMM 的输出 scale 或在 UB 内做。
# 收益: 省掉 FP16 权重/激活 的 HBM 往返; 访存密集推理里这是主要加速来源。
# 进阶: W4A8(W4 权重反量化成 FP16 再做 FP16 GEMM)、per-channel scale、把 DeQuant
#   折成 GEMM 的 epilogue scale(A * scale_w)进一步省指令。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: g.quant-fusion | DeQuant+MatMul 夹心: 省中间 FP16 落 HBM"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 不融合 vs 融合:IO 对比 ---
hdr(1,TOTAL,"不融合 vs 融合: HBM 往返次数")
why("""一次 y=x@W^T (W8A8), 两种实现搬运的数据量:""")
m, k, n = 1024, 4096, 4096
w_int8 = k * n * 1          # INT8 权重
w_fp16 = k * n * 2          # 反量化后 FP16 权重
x_int8 = m * k * 1
z_fp16 = m * n * 2
z_int8 = m * n * 1
io_unfused = w_int8 + w_fp16 + w_fp16 + x_int8 + z_fp16 + z_fp16 + z_int8
io_fused   = w_int8 + x_int8 + z_int8
out = [f"  形状 [m={m},k={k},n={n}] (字节):"]
out.append(f"  不融合(7 次 HBM): 读INT8 W + 写FP16 W + 读FP16 W + 读INT8 X + 写FP16 Z + 读FP16 Z + 写INT8 Z")
out.append(f"    总 IO = {io_unfused/1e6:.0f} MB  (FP16 权重 {w_fp16/1e6:.0f}MB 来回, FP16 Z {z_fp16/1e6:.0f}MB 来回)")
out.append(f"  融合  (3 次 HBM): 读INT8 W + 读INT8 X + 写INT8 Z; FP16 全在 UB")
out.append(f"    总 IO = {io_fused/1e6:.0f} MB")
out.append(f"  节省: {io_unfused/io_fused:.1f}x  (主要来自不搬运 FP16 中间量)")
res("\n".join(out))
mea("访存密集推理(Decode 阶段 m 小)里, GEMM 本身算不满, 瓶颈是搬权重。\n  融合让搬运的权重从 FP16 缩成 INT8, 又省了中间落盘, 是 W8A8 加速的核心。")

# --- 2. 反量化夹心:数学等价 + tile 流水 ---
hdr(2,TOTAL,"夹心 kernel: 反量化在 UB 内, 直接喂 Cube")
why("""融合 kernel 内部按 tile 推进, FP16 中间量只在 UB:
  for tile in W (按 n 切):
    MTE2: 搬 INT8 W_tile 进 UB
    Vector: W_fp16 = W_tile.cast(FP16) * scale_w   (UB 内, 不落 HBM)
    Cube:  Z_tile = X @ W_fp16^T                    (直接用 UB 里的 FP16)
    (可选) Vector: Quant(Z_tile) -> INT8
    MTE3: 写 INT8 Z_tile 回 HBM
  数学: Z = X @ (Wq*scale_w)^T = (X @ Wq^T) * scale_w  ← scale 可后置到输出""")
np.random.seed(0)
kt, nt = 128, 64
Wq = np.random.randint(-127, 127, size=(nt, kt)).astype(np.int8)
X  = np.random.randn(8, kt).astype(np.float32) * 0.3
scale_w = (np.abs(Wq.astype(np.float32)).max(axis=1)/127 + 1e-9).reshape(-1,1)
W_fp = Wq.astype(np.float32) * scale_w
Z_ref = X @ W_fp.T
Z_post = (X @ Wq.astype(np.int32).T).astype(np.float32) * scale_w.ravel()
res(f"""数值等价性验证 (tile={kt}x{nt}):
  先反量化再GEMM:  ||Z||={np.linalg.norm(Z_ref):.4f}
  先INT8 GEMM再乘scale(后置): ||Z||={np.linalg.norm(Z_post):.4f}
  最大差异: {np.max(np.abs(Z_ref-Z_post)):.2e}  (完全等价)""")
mea("scale 后置(先 INT8 GEMM, 再乘 scale_w)让 Cube 跑纯 INT8, 反量化缩放挪到 Vector epilogue。\n  这是夹心融合最省的形态: Cube 不碰 FP16, Vector 只在输出做一次 Mul。")

# --- 3. CPU 模拟:分块反量化+GEMM, IO 计数 ---
hdr(3,TOTAL,"CPU 模拟: 分块融合 vs 不融合, 统计搬运")
why("""模拟两种实现的"UB<->HBM 搬运字节数", 验证融合省在哪。""")
def unfused(x, wq, sw):
    # 不融合: 先整体反量化(写FP16) 再GEMM
    w_fp = wq.astype(np.float32) * sw          # 落 HBM
    io = wq.nbytes + w_fp.nbytes + w_fp.nbytes # 读INT8 + 写FP16 + 读FP16
    z_fp = x @ w_fp.T                          # 落 HBM
    io += x.nbytes + z_fp.nbytes + z_fp.nbytes
    zq = np.round(z_fp).clip(-127,127).astype(np.int8)  # 假装再 Quant
    io += zq.nbytes
    return zq, io
def fused(x, wq, sw, tile_n=64):
    # 融合: 按tile反量化在"UB"内, 直接GEMM, 只读写INT8到HBM
    m, k = x.shape; n = wq.shape[0]
    z = np.zeros((m, n), dtype=np.float32)
    io = x.nbytes + wq.nbytes                  # 读INT8 X + 读INT8 W(只一次)
    for s in range(0, n, tile_n):
        e = min(s+tile_n, n)
        w_fp_tile = wq[s:e].astype(np.float32) * sw[s:e]   # UB 内, 不计 HBM
        z[:, s:e] = x @ w_fp_tile.T
    zq = np.round(z).clip(-127,127).astype(np.int8)
    io += zq.nbytes                            # 写INT8 Z
    return zq, io
m, k, n = 16, 1024, 512
Wq = np.random.randint(-127,127,size=(n,k)).astype(np.int8)
X  = np.random.randn(m,k).astype(np.float32)*0.3
sw = (np.abs(Wq.astype(np.float32)).max(axis=1)/127+1e-9).reshape(-1,1)
zu, iou = unfused(X, Wq, sw)
zf, iof = fused(X, Wq, sw, tile_n=64)
res(f"""[m={m},k={k},n={n}], tile_n=64:
  不融合 HBM 搬运: {iou/1e6:.2f} MB
  融合   HBM 搬运: {iof/1e6:.2f} MB
  节省: {iou/iof:.1f}x
  数值一致: {np.array_equal(zu, zf)}""")
mea("融合省下的全是「FP16 中间量的往返」。\n  Decode 阶段 m 小、GEMM 访存密集, 这部分 IO 占比极高, 融合收益最大。")

# --- 4. W4A8 / 进阶 + Ascend 实现 ---
hdr(4,TOTAL,"W4A8、per-channel scale 与 Ascend 落地")
why("""夹心融合的几种变体与昇腾侧实现:""")
out = ["  形态        权重   激活   GEMM       反量化方式"]
out.append("  W8A8        INT8   INT8   INT8 Cube   scale 后置到输出(Vector epilogue)")
out.append("  W4A8        INT4   INT8   INT8 Cube   权重 INT4->FP16(UB) 再做 INT8? 实为 W4->FP16 + FP16 GEMM")
out.append("  W4A16       INT4   FP16   FP16 Cube   weight-only: 权重反量化, 激活不变(AWQ/GPTQ 常用)")
out.append("  per-channel scale  [oc]个scale  按通道广播 Mul, 比per-tensor精度高(见 e.quant-op)")
out.append("")
out.append("  Ascend 实现:")
out.append("    aclnnFusedDequantMatmul / WeightQuantBatchMatmul  官方融合算子")
out.append("    AscendC: MTE2搬INT8/INT4 -> Vector反量化(UB) -> Cube GEMM -> (Vector Quant) -> MTE3")
out.append("    torch_npu / MindIE: int8 W8A8 / W4A16 matmul 已内置融合 kernel")
res("\n".join(out))
mea("""实战要点:
  - W8A8 全量化: 权重+激活都 INT8, 收益最大, 误差略高(适合吞吐优先)
  - W4A16 weight-only: 只量化权重, 激活 FP16, 精度好(AWQ/GPTQ 主流, LLM 首选)
  - 融合 kernel 把"反量化"从一次独立 HBM 往返, 降成 UB 内的"过路操作"
  - msprof 若看到独立的 Cast/Mul(DeQuant) kernel 排在 MatMul 前后, 说明没融合, 可优化""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:量化推理里"反量化出来的 FP16"只是过路客;融合 = 让它在 UB 里待着、直接喂
  给 GEMM, 不落 HBM。省的是最贵的访存, 不是算力。
- 熟手:scale 后置(先 INT8 GEMM 再乘 scale_w)让 Cube 跑纯 INT8、反量化挪到输出
  epilogue, 是最省形态;Decode 访存密集阶段收益最大;W4A16(weight-only)是 LLM
  精度与速度的甜点。
【进阶】用 AscendC 写一个 fused DeQuant+MatMul: MTE2 搬 INT8 权重块 -> Vector
  反量化(UB) -> Cube GEMM, 双缓冲让搬运与计算 overlap;msprof 对比独立 DeQuant
  +MatMul 的 HBM 带宽, 验证 FP16 中间量消失。
EOF
echo "############################################################"
