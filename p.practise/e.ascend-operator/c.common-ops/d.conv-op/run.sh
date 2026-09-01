#!/bin/bash
# ============================================================
# 实验: d.conv-op
# 说明: Im2col/直接卷积、Cube 加速
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 卷积 = 大量小矩阵乘。
# 直接卷积: 7x7 卷积 = 49 次乘加 / 输出元素 → 慢
# Im2col: 把输入\"展开\"成大矩阵,然后 1 次 GEMM
#   输入 (N, C, H, W) → (N, C*kH*kW, H*W)  (col 形式)
#   权重 (Cout, C*kH*kW) → 不变
#   输出 (N, Cout, H, W) = 权重 @ col
# 优势: 用 Cube (GEMM) 代替手写卷积, 加速比大
# 劣势: Im2col 本身有内存开销 (临时矩阵)
# 进阶: Winograd (更省乘法), FFT-based (大 kernel)
# LLM 几乎不用卷积, 但 ViT / 图像模型用
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: d.conv-op | Im2col + Cube GEMM 加速 + Winograd"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np, time
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 直接卷积 vs Im2col ---
hdr(1,TOTAL,"直接卷积 vs Im2col+GEMM")
why("""3x3 卷积, 输入 8x8:
  直接:  对每个输出位置 (6x6=36 个) 做 3x3=9 次乘加 = 36*9 = 324 次
  Im2col + GEMM: 展开成 (3*3, 6*6) = (9, 36), 与 (1, 9) 权重 GEMM = 1*9*36 = 324 次
  数量一样, 但 GEMM 走 Cube, 直接卷积走 Vector""")
N, C, H, W = 1, 3, 8, 8
kH, kW = 3, 3
Cout = 1
x = np.random.randn(N, C, H, W)
w = np.random.randn(Cout, C, kH, kW)
# 直接卷积 (naive)
def conv_direct(x, w):
    N, C, H, W = x.shape
    _, _, kH, kW = w.shape
    out = np.zeros((N, C, H-kH+1, W-kW+1))
    for n in range(N):
        for c in range(Cout):
            for i in range(out.shape[2]):
                for j in range(out.shape[3]):
                    s = 0
                    for ci in range(C):
                        for ki in range(kH):
                            for kj in range(kW):
                                s += x[n, ci, i+ki, j+kj] * w[c, ci, ki, kj]
                    out[n, c, i, j] = s
    return out
# Im2col + GEMM
def im2col(x, kH, kW):
    N, C, H, W = x.shape
    cols = np.zeros((N, C*kH*kW, (H-kH+1)*(W-kW+1)))
    for n in range(N):
        for c in range(C):
            for ki in range(kH):
                for kj in range(kW):
                    cols[n, c*kH*kW + ki*kW + kj] = x[n, c, ki:ki+H-kH+1, kj:kj+W-kW+1].flatten()
    return cols
# 实测太慢, 用小尺寸演示
x = np.random.randn(1, 3, 6, 6).astype(np.float32)
w = np.random.randn(1, 3, 3, 3).astype(np.float32)
t = time.perf_counter()
for _ in range(10): r1 = conv_direct(x, w)
t1 = (time.perf_counter()-t)/10
cols = im2col(x, 3, 3)
w_2d = w.reshape(1, -1)
t = time.perf_counter()
for _ in range(1000): r2 = (w_2d @ cols).reshape(1, 1, 4, 4)
t2 = (time.perf_counter()-t)/1000
res(f"""CPU 1x3x6x6, 3x3 卷积 (估):
  直接卷积: {t1*1000:.1f} ms / 次
  Im2col+GEMM: {t2*1000:.3f} ms / 次
  加速: {t1/t2:.0f}×""")
mea("Im2col + GEMM 在大尺寸下用 BLAS 加速, 实际加速 10-100×。\n  缺点: Im2col 占用额外内存 (C*kH*kW 倍)。")

# --- 2. Im2col 内存开销 ---
hdr(2,TOTAL,"Im2col 内存开销")
why("""Im2col 把 HxWxC 输入展开成 (C*kH*kW, H*W), 内存膨胀 kH*kW 倍。
  ResNet 第一层 7x7 卷积:
    输入 (224, 224, 3) = 600 KB
    Im2col (3*7*7, 218*218) = (147, 47524) = 28 MB
    膨胀: 47×
  巨大! 优化: Winograd, FFT, 直接卷积""")
in_size = 224*224*3*4
col_size = 3*7*7*218*218*4
res(f"""7x7 卷积, 输入 224x224x3 (FP32):
  输入:     {in_size/1024/1024:.1f} MB
  Im2col:   {col_size/1024/1024:.1f} MB
  膨胀:     {col_size/in_size:.0f}×""")
mea("""Im2col 内存开销是 1 个核心问题。优化方法:
  1. Winograd: 把 3x3 卷积变成 4x4 element-wise 矩阵乘, 乘法数少 36%
  2. FFT: 大 kernel (>5x5) 优势明显
  3. 直接卷积: 大 kernel, 大 stride 用直接更快
  4. 隐式 GEMM: 边算边展开, 不真存 Im2col 矩阵""")

# --- 3. Winograd F(2x2, 3x3) ---
hdr(3,TOTAL,"Winograd:F(2x2,3x3) 卷积")
why("""Winograd 把 3x3 卷积变 4 个 4x4 矩阵乘 (替代 9 个 3x3 矩阵乘):
  - 输入变换: 4x4 element-wise (G)
  - 输出变换: 2x2 element-wise (A^T)
  - 中间: 4x4 element-wise 矩阵乘 (逐点乘)
  乘法数: 36 (直接) → 16 (Winograd) = 2.25× 减少
  实际加速: 1.5-2×""")
res("""Winograd 加速 (经验):
  算法         乘法数       实际加速
  直接卷积     9*M*N       1.0×
  Im2col+GEMM  9*M*N       0.9-1.0×  (有 Im2col 开销)
  Winograd     16/4 = 4*输出  1.5-2×
  FFT          N*log(N)     1.2-1.5× (大 kernel)""")
mea("Winograd 是 NCNN / TVM / TensorRT 推理框架卷积默认算法之一。\n  限制: 数值范围小, 需 FP32 累加器, FP16 需特别处理。")

# --- 4. Cube 卷积实践 ---
hdr(4,TOTAL,"Cube 卷积:AscendC 实战")
why("""昇腾 Cube 单元 + Im2col = 极快卷积:
  1. Im2col (Vector 单元): 展开成 (C*kH*kW, H*W)
  2. Cube 算 Matmul: 权重 @ Im2col
  3. Reshape 回 (N, Cout, H', W')
  
  AscendC 提供 conv2d API, 内部已做 Im2col。
  自研: 也可以手写 Im2col + Matmul, 更灵活。""")
out = ["  方式                    算力利用率    灵活度    适用"]
out.append("  conv2d (内置)           80%          低        标准卷积")
out.append("  Im2col + Matmul (手写)  75%          高        自定义 (空洞, 分组)")
out.append("  Winograd (手写)         70%          高        3x3 卷积")
out.append("  FFT (手写)              50%          高        大 kernel")
out.append("  直接卷积 (手写)         40%          极高      极小或特殊")
res("\n".join(out))
mea("""实战选型:
  - 标准 CNN 训练: 用 conv2d 内置
  - 自定义算子 (ViT patch embed, dilate): 手写 Im2col + Matmul
  - 极致性能: Winograd, 但 FP16 需精度 hack
  - LLM 几乎不用卷积, 但 ViT 必备""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:卷积可以用 Im2col 转成 GEMM,用 Cube 加速;Winograd 再省 36% 乘法;
  LLM 几乎不用卷积,但 ViT / 图像模型必备;Cube 卷积实战 80% 算力利用率。
- 熟手:Im2col 内存膨胀 kH*kW 倍,大输入需特殊处理;Winograd 是 3x3 卷积最优
  但需 FP32 累加器;自研卷积常见于 ViT patch_embed / 空洞卷积;msprof 看 Cube
  利用率 70%+ 算优秀。
【进阶】用 AscendC 写一个 Im2col + Matmul 卷积算子 (1x1 conv), 对比内置
  conv2d 的性能,理解 Im2col 的开销。
EOF
echo "############################################################"
