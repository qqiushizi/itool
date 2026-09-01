#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""体验代码: 决策树与 GBDT —— 改树数/学习率/噪声看效果
用法:
  python3 demo.py --teach          # 完整讲解(run.sh 默认调用)
  python3 demo.py --trees 20       # 增加树数,看 MSE 下降
  python3 demo.py --lr 0.3         # 改学习率
  python3 demo.py --noise 0.5      # 加噪声,看过拟合
"""
import argparse
try:
    import numpy as np
    HAVE = True
except ImportError:
    HAVE = False


def entropy(y):
    if len(y) == 0:
        return 0.0
    p = np.array([np.mean(y == 0), np.mean(y == 1)])
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def info_gain(y, left, right):
    n = len(y)
    h_parent = entropy(y)
    h_left = entropy(y[left])
    h_right = entropy(y[right])
    return h_parent - (len(left) / n * h_left + len(right) / n * h_right)


def best_split(x, y):
    best_g, best_t = -1, None
    for t in np.unique(x):
        left, right = x <= t, x > t
        if left.all() or right.all():
            continue
        g = info_gain(y, left, right)
        if g > best_g:
            best_g, best_t = g, t
    return best_g, best_t


class Stump:
    def fit(self, x, r):
        best, self.t, self.bl, self.br = 1e18, 0, 0, 0
        for t in np.unique(x):
            m = x <= t
            if m.all() or not m.any():
                continue
            cost = ((r[m] - r[m].mean()) ** 2).sum() + ((r[~m] - r[~m].mean()) ** 2).sum()
            if cost < best:
                best, self.t = cost, t
                self.bl, self.br = r[m].mean(), r[~m].mean()

    def predict(self, x):
        return np.where(x <= self.t, self.bl, self.br)


def gbdt(x, y, n_trees, lr):
    F = np.zeros_like(y)
    mses = [np.mean((y - F) ** 2)]
    for _ in range(n_trees):
        residual = y - F
        tree = Stump()
        tree.fit(x, residual)
        F += lr * tree.predict(x)
        mses.append(np.mean((y - F) ** 2))
    return F, mses


def teach():
    print("\n" + "=" * 60)
    print("【步骤 1/3】决策树:选信息增益最大的切分点")
    print("=" * 60)
    print("\n--- 做什么&为什么 ---")
    print("  造分类数据,遍历所有切分点,选信息增益最大的。")
    print("  信息增益 = 分裂后不确定性下降多少。增益越大,切分越优。")
    print("\n--- 运行结果 ---")
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 10, 60)
    y = (x > 5).astype(int)
    flip = rng.choice(60, 6, replace=False)
    y[flip] = 1 - y[flip]
    g, t = best_split(x, y)
    print(f"  最优切分点 x<={t:.3f}, 信息增益={g:.4f} bits")
    print(f"  左侧(x<={t:.2f}) 正例比例={y[x <= t].mean():.2f}")
    print(f"  右侧(x>{t:.2f})  正例比例={y[x > t].mean():.2f}")
    print("\n--- 结果解读 ---")
    print("  最优切分让左右两侧尽量'纯'。树=递归对子集继续找最优切分。")

    print("\n" + "=" * 60)
    print("【步骤 2/3】GBDT:后一棵树拟合前一棵的残差")
    print("=" * 60)
    print("\n--- 做什么&为什么 ---")
    print("  用一维非线性目标(sin),8棵深度1的弱树接力拟合残差。")
    print("  每棵树只学一点点(残差),累加起来能逼近非线性曲线=集弱成强。")
    print("\n--- 运行结果 ---")
    rng = np.random.default_rng(5)
    x = np.linspace(0, 6, 40)
    y = np.sin(x) + rng.standard_normal(40) * 0.2
    F, mses = gbdt(x, y, 8, 0.5)
    for k in range(8):
        print(f"  第{k + 1}棵树: 累计 MSE={mses[k + 1]:.4f}")
    print("\n--- 结果解读 ---")
    print("  MSE 从 0.5 左右逐棵下降到约 0.06。残差被逐步消化,预测逼近 sin。")

    print("\n" + "=" * 60)
    print("【步骤 3/3】噪声如何影响信息增益")
    print("=" * 60)
    print("\n--- 做什么&为什么 ---")
    print("  对比纯数据 vs 含噪数据在切点 x<=5 的信息增益。")
    print("  噪声降低增益,树会变深变杂=过拟合来源,故需剪枝/限深度。")
    print("\n--- 运行结果 ---")
    xs = np.linspace(0, 10, 60)
    y_pure = (xs > 5).astype(int)
    y_noisy = y_pure.copy()
    rng = np.random.default_rng(9)
    y_noisy[rng.choice(60, 9, replace=False)] ^= 1
    for t in [3.0, 5.0]:
        lp, rp = xs <= t, xs > t
        gp = info_gain(y_pure, lp, rp)
        gn = info_gain(y_noisy, lp, rp)
        print(f"  切点 x<={t}: 纯数据增益={gp:.4f}, 含噪增益={gn:.4f}")
    print("\n--- 结果解读 ---")
    print("  纯数据增益高(切分有效);含噪增益低甚至为负(噪声让子节点更乱)。")

    print("\n" + "#" * 50)
    print("【整体总结】")
    print("  小白:决策树用'是/否'把数据分纯,选信息增益最大的切分;")
    print("        GBDT 让多棵弱树接力拟合残差,累加成强模型。")
    print("  熟手:GBDT 每轮拟合负梯度(平方损失即残差);XGBoost/LightGBM 是其工程优化。")


def single(args):
    rng = np.random.default_rng(args.seed)
    x = np.linspace(0, 6, 40)
    y = np.sin(x) + rng.standard_normal(40) * args.noise
    F, mses = gbdt(x, y, args.trees, args.lr)
    print("=" * 60)
    print(f"GBDT: 树数={args.trees}, 学习率={args.lr}, 噪声={args.noise}")
    print("=" * 60)
    print(f"  初始 MSE={mses[0]:.4f}  ->  最终 MSE={mses[-1]:.4f}")
    print("\n--- 解读 ---")
    if mses[-1] < mses[0] * 0.3:
        print("  MSE 大幅下降,GBDT 有效拟合了非线性目标。")
    elif mses[-1] < mses[0]:
        print("  MSE 下降有限:可增加树数或调学习率。")
    else:
        print("  MSE 几乎不降:参数不合适或噪声过大。")


def main():
    p = argparse.ArgumentParser(description="决策树与 GBDT 体验")
    p.add_argument("--teach", action="store_true")
    p.add_argument("--trees", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--noise", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()
    if not HAVE:
        print("建议安装 numpy: pip install numpy")
        return
    if args.teach:
        teach()
    else:
        single(args)


if __name__ == "__main__":
    main()
