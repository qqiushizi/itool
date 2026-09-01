#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""体验代码: 逻辑回归 —— 改学习率/噪声看收敛与决策边界
用法:
  python3 demo.py --teach        # 完整三段式讲解(run.sh 默认调用)
  python3 demo.py --lr 0.5       # 改学习率
  python3 demo.py --noise 0.3    # 加标签噪声
  python3 demo.py --boundary     # 看决策边界随训练移动
"""
import argparse
try:
    import numpy as np
    HAVE = True
except ImportError:
    HAVE = False


def make_data(n, noise, seed=3):
    rng = np.random.default_rng(seed)
    sep = 2.0
    x = np.concatenate([rng.standard_normal(n // 2) - sep, rng.standard_normal(n // 2) + sep]).reshape(-1, 1)
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    flip = rng.choice(n, int(n * noise), replace=False)
    y[flip] = 1 - y[flip]
    X = np.column_stack([x[:, 0], np.ones(n)])
    return x, y, X


def train(X, y, lr, steps=300):
    w = np.zeros(X.shape[1])
    loss = []
    for i in range(steps):
        pred = 1 / (1 + np.exp(-(X @ w)))
        eps = 1e-12
        l = -np.mean(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps))
        w -= lr * X.T @ (pred - y) / len(y)
        loss.append(l)
    return w, loss


def teach():
    print("\n" + "=" * 60)
    print("【步骤 1/4】sigmoid:把分数压成概率")
    print("=" * 60)
    print("\n--- 做什么&为什么 ---")
    print("  对不同 z 算 sigma(z)=1/(1+e^-z)。分类要概率(0~1),sigmoid 光滑压缩。")
    print("  z=0 恰好 0.5 = 决策边界。softmax 是其多类推广。")
    print("\n--- 运行结果 ---")
    for z in [-3, -1, 0, 1, 3]:
        print(f"  z={z:>2} -> sigma(z)={1 / (1 + np.exp(-z)):.3f}")
    print("\n--- 结果解读 ---")
    print("  z 越大越接近1,越小越接近0,z=0 是五五开。")

    print("\n" + "=" * 60)
    print("【步骤 2/4】训练逻辑回归 + 决策边界")
    print("=" * 60)
    print("\n--- 做什么&为什么 ---")
    print("  造二分类数据,梯度下降训练,算决策边界 z=0 对应的 x。")
    print("  决策边界是两类分界线,逻辑回归本质就是找这条线。")
    print("\n--- 运行结果 ---")
    x, y, X = make_data(100, 0.0)
    w, loss = train(X, y, 0.1)
    print(f"  训练完成: w={w[0]:.4f}, b={w[1]:.4f}")
    print(f"  决策边界: x = {-w[1] / w[0]:.3f}  (sigma=0.5 处)")
    for s in [0, 99, 299]:
        print(f"  步{s + 1:>3}: 对数损失={loss[s]:.4f}")
    print("\n--- 结果解读 ---")
    print("  loss 从 0.69(随机)降到约 0.07。决策边界约 0,正是两簇中点。")

    print("\n" + "=" * 60)
    print("【步骤 3/4】为什么用对数损失而非 MSE")
    print("=" * 60)
    print("\n--- 做什么&为什么 ---")
    print("  真实类=1,对比不同预测的对数损失 vs MSE。")
    print("  MSE 在 sigmoid 下非凸难优化,对'自信地错'惩罚不够。")
    print("\n--- 运行结果 ---")
    print(f"  {'预测':<7}{'对数损失':>10}{'MSE':>10}")
    for pred in [0.9, 0.5, 0.1, 0.01]:
        print(f"  {pred:<7}{-np.log(pred + 1e-12):>10.3f}{(pred - 1) ** 2:>10.4f}")
    print("\n--- 结果解读 ---")
    print("  预测0.01却真实为1:对数损失4.6(指数重罚),MSE仅0.98。")

    print("\n" + "=" * 60)
    print("【步骤 4/4】概率校准:预测0.8真是80%吗?")
    print("=" * 60)
    print("\n--- 做什么&为什么 ---")
    print("  把预测概率分箱,看每箱实际正例比例是否匹配。")
    print("  校准好=置信度可信。风控/医疗场景极重要。")
    print("\n--- 运行结果 ---")
    pred = 1 / (1 + np.exp(-(X @ w)))
    bins = np.linspace(0, 1, 6)
    print(f"  {'预测区间':<14}{'样本数':<8}{'实际正例比例'}")
    for i in range(len(bins) - 1):
        m = (pred >= bins[i]) & (pred < bins[i + 1])
        if m.sum() > 0:
            print(f"  [{bins[i]:.1f},{bins[i + 1]:.1f})    {m.sum():<8}{y[m].mean():.3f}")
    print("\n--- 结果解读 ---")
    print("  高概率箱实际正例比例也高=校准好。偏差大需温度缩放校准。")

    print("\n" + "#" * 50)
    print("【整体总结】")
    print("  小白:sigmoid把分数压成概率,决策边界是z=0直线,用对数损失而非MSE。")
    print("  熟手:逻辑回归是单层神经网络特例;对数损失梯度=(pred-y)x;校准决定置信度。")


def single(args):
    x, y, X = make_data(100, args.noise)
    w, loss = train(X, y, args.lr)
    print("=" * 60)
    print(f"逻辑回归: lr={args.lr}, noise={args.noise}")
    print("=" * 60)
    print(f"  最终 loss={loss[-1]:.4f}  w={w[0]:.4f} b={w[1]:.4f}")
    print(f"  决策边界 x={-w[1] / w[0]:.3f}")
    print("\n--- 解读 ---")
    if loss[-1] < 0.1:
        print("  收敛好,loss低,决策边界合理。")
    elif loss[-1] < 0.4:
        print("  有噪声干扰,loss偏高。")
    else:
        print("  loss高:噪声大或学习率不合适。")


def boundary(args):
    print("=" * 60)
    print("决策边界随训练移动(每50步)")
    print("=" * 60)
    x, y, X = make_data(100, args.noise)
    w = np.zeros(2)
    for i in range(301):
        pred = 1 / (1 + np.exp(-(X @ w)))
        w -= args.lr * X.T @ (pred - y) / 100
        if i % 50 == 0:
            print(f"  步{i:>3}: 决策边界 x={-w[1] / w[0]:>+8.3f}")


def main():
    p = argparse.ArgumentParser(description="逻辑回归体验")
    p.add_argument("--teach", action="store_true")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--boundary", action="store_true")
    args = p.parse_args()
    if not HAVE:
        print("建议 numpy 可获完整体验: pip install numpy")
        return
    if args.teach:
        teach()
    elif args.boundary:
        boundary(args)
    else:
        single(args)


if __name__ == "__main__":
    main()
