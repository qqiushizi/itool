#!/bin/bash
# ============================================================
# 实验: e.dimensionality-reduction
# 说明: PCA 推导与实现、与 SVD 的关系
# 模块: p.practise/a.ai-fundamentals  AI 基础
# ============================================================
# 【第一性原理】
# PCA(主成分分析):高维数据找"信息最密集的少数方向",投影过去实现降维。
# - 核心思路:数据去中心化后,协方差矩阵的特征向量就是"方差最大的方向"。
#   最大特征值对应的方向 = 第一主成分(信息最多),依次类推。
# - 与 SVD 的关系:对中心化矩阵 X 做 SVD 得 X=UΣV^T,则 V 的列就是主成分方向,
#   Σ 的奇异值平方正比于各方向的方差。PCA 和 SVD 在数学上是同一件事的两种表达。
# - 降维:只保留前 k 个主成分,把 n 维压到 k 维,丢失的是小方差方向(噪声/次要信息)。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: PCA 推导 / 特征值 / 与 SVD 的关系"
echo "============================================================"

python3 <<'PY'
import numpy as np
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(9)

# ---------- 造高维数据:真实低维结构 + 噪声 ----------
n, d = 100, 5
true_dir = np.array([3, 0, 0, 0, 0])                        # 信息只集中在第1维
X = rng.standard_normal((n,1)) * true_dir + rng.standard_normal((n,d)) * 0.3
print(f"数据 shape={X.shape}, 真实信息集中在前几维\n")

# ---------- 1. PCA 法一:协方差矩阵特征分解 ----------
print("【1】PCA(特征分解法):协方差矩阵的特征向量=主成分方向")
Xc = X - X.mean(axis=0)                                      # 中心化(去均值)
Cov = (Xc.T @ Xc) / (n - 1)                                  # 协方差矩阵
eigvals, eigvecs = np.linalg.eigh(Cov)                       # 特征值/向量(eigh 适合对称矩阵)
idx = np.argsort(eigvals)[::-1]                              # 按特征值从大到小排序
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
print(f"  特征值(方差) = {eigvals.round(3)}")
print(f"  方差解释比   = {(eigvals/eigvals.sum()*100).round(1)}%")
# 直觉:第1主成分占绝大部分方差,说明数据本质是低维的,其余维是噪声。

# ---------- 2. PCA 法二:SVD(两者等价) ----------
print("\n【2】PCA(SVD 法):对中心化矩阵做 SVD,奇异值²/总 = 方差占比")
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
var_ratio = (S**2) / (S**2).sum()
print(f"  奇异值      = {S.round(3)}")
print(f"  方差解释比 = {(var_ratio*100).round(1)}%  (应与特征分解法一致)")
# 验证:SVD 的 Vt 行 vs 特征向量列,方向应一致(可能差一个符号)
match = np.allclose(np.abs(Vt[0]), np.abs(eigvecs[:,0]))
print(f"  SVD 主成分方向与特征向量是否一致(绝对值)={match}")
# 直觉:SVD 不需要显式算协方差矩阵,数值更稳,是 sklearn PCA 的底层实现。

# ---------- 3. 降维:5维 → 2维 ----------
print("\n【3】降维:保留前 2 个主成分,投影到 2 维")
k = 2
W = Vt[:k].T                                                 # 主成分方向 [d, k]
X_reduced = Xc @ W                                           # 投影 [n, k]
print(f"  原维度 {d} -> 降维后 {X_reduced.shape[1]}")
print(f"  保留信息 = {var_ratio[:k].sum()*100:.1f}%")
# 重构:用 2 维反推回 5 维,看丢多少
X_recon = X_reduced @ W.T + X.mean(axis=0)
recon_err = np.mean((X - X_recon)**2)
print(f"  重构均方误差 = {recon_err:.4f}  (丢弃维的方差)")
# 直觉:保留前2维已抓住几乎全部信息,重构误差≈被丢弃的噪声方差。

# ---------- 4. 不同 k 的信息保留 ----------
print("\n【4】保留不同主成分数 k 的累计信息:")
cum = np.cumsum(var_ratio)
for k in range(1, d+1):
    print(f"  k={k}: 累计方差 = {cum[k-1]*100:.1f}%")
# 直觉:k=1 就够 90%+,说明数据可大幅压缩。实际中常选累计方差达 95% 的最小 k。
PY

echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:PCA 找方差最大的方向当主成分,投影过去就能降维;特征分解和 SVD 是同一件事的两种算法;
  保留大特征值方向=留信息,丢小方向=去噪声。
- 熟手:SVD 法数值更稳且能处理 d>n 的瘦高数据,是工程首选;PCA 是线性降维,非线性结构用 t-SNE/UMAP;
  PCA 也用于特征去相关、白化、可视化。
- 延伸:把噪声调到 0 看"完美低维";对比 PCA 降维前后做分类的准确率变化。
EOF
echo "============================================================"
