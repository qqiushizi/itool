#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
体验代码: SVD 低秩近似 —— 改参数看压缩效果
==========================================
用途: 自己调参数,直观体会"保留多少奇异值"如何影响压缩精度与信息保留率。
      这是理解 LoRA / PCA / 模型压缩"为什么能省"的关键直觉。

用法:
  python3 demo.py                    # 默认:随机矩阵,保留全部奇异值
  python3 demo.py --k 1              # 只保留 1 个奇异值(激进压缩)
  python3 demo.py --k 2              # 保留 2 个
  python3 demo.py --seed 7           # 换一个随机矩阵
  python3 demo.py --rows 4 --cols 6  # 自定义矩阵尺寸
  python3 demo.py --k 1 --k 2 --k 3  # 对比多个 k(看精度如何随 k 变化)

无需 numpy 也能理解原理;若已安装 numpy 则用真实分解,否则用纯 Python 近似演示。
"""
import argparse, sys

def have_numpy():
    try:
        import numpy; return True
    except ImportError:
        return False

def run_with_numpy(args):
    import numpy as np
    rng = np.random.default_rng(args.seed)
    # 构造一个"有低秩结构"的矩阵:真值低秩 + 少量噪声
    true_rank = min(args.rows, args.cols) // 2
    L = rng.standard_normal((args.rows, true_rank))
    R = rng.standard_normal((true_rank, args.cols))
    A = L @ R + 0.1 * rng.standard_normal((args.rows, args.cols))

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    np.set_printoptions(precision=3, suppress=True)

    print("=" * 60)
    print(f"原始矩阵 A: shape={A.shape}, 真实低秩结构 rank≈{true_rank}")
    print("=" * 60)
    print(A)
    print(f"\n奇异值 S = {S.round(3).tolist()}")
    total_energy = np.sum(S ** 2)
    print(f"总能量(奇异值平方和) = {total_energy:.3f}\n")

    ks = args.k if args.k else list(range(1, len(S) + 1))
    print("=" * 60)
    print("对比不同保留数 k 的压缩效果")
    print("=" * 60)
    print(f"{'k':>3} | {'能量保留率':>10} | {'重构误差':>10} | {'压缩比':>8}")
    print("-" * 60)
    for k in ks:
        k = min(k, len(S))
        A_low = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        keep = np.sum(S[:k] ** 2) / total_energy
        err = np.linalg.norm(A - A_low) / np.linalg.norm(A)   # 相对误差
        # 压缩比:原始存 rows*cols 个数,低秩存 k*(rows+cols) 个数
        orig = args.rows * args.cols
        comp = k * (args.rows + args.cols)
        ratio = comp / orig if orig > 0 else 0
        print(f"{k:>3} | {keep*100:>9.1f}% | {err*100:>9.2f}% | {ratio:>7.2f}x")

    print("\n--- 解读 ---")
    print("  能量保留率:保留了多少主要信息(越高越好)")
    print("  重构误差:压缩后和原矩阵差多少(越低越好)")
    print("  压缩比:存储代价占原始的比例(越低越省)")
    print("  → k 越大越精确但越不省;k 小省空间但丢信息。这就是压缩的核心权衡。")

    if len(ks) == 1:
        k = min(ks[0], len(S))
        A_low = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        print(f"\n保留 k={k} 时,重建矩阵 =")
        print(A_low.round(3))

def run_pure(args):
    # 无 numpy 时的简化演示:用小固定矩阵手动展示概念
    A = [[3, 2, 2], [2, 3, -2]]
    print("=" * 60)
    print("纯 Python 模式(建议安装 numpy 获得完整体验: pip install numpy)")
    print("=" * 60)
    print("原矩阵 A:")
    for row in A: print("  ", row)
    print("\n这个 2x3 矩阵的奇异值约为 [5, 3](由 numpy 计算得出)")
    print("保留 k=1: 只用最大的奇异值,能量保留约 73.5%,已抓住主要结构")
    print("保留 k=2: 用全部奇异值,能量保留 100%,完美还原")
    print("\n→ 安装 numpy 后运行: python3 demo.py --k 1 --k 2  看真实对比")

def main():
    p = argparse.ArgumentParser(
        description="SVD 低秩近似体验:改 --k 看压缩效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python3 demo.py --k 1 --k 2 --k 3")
    p.add_argument("--k", type=int, action="append", default=None,
                   help="保留的奇异值个数(可重复指定对比,如 --k 1 --k 2)")
    p.add_argument("--seed", type=int, default=42, help="随机种子(换矩阵)")
    p.add_argument("--rows", type=int, default=4, help="矩阵行数")
    p.add_argument("--cols", type=int, default=4, help="矩阵列数")
    args = p.parse_args()

    if have_numpy():
        run_with_numpy(args)
    else:
        run_pure(args)

if __name__ == "__main__":
    main()
