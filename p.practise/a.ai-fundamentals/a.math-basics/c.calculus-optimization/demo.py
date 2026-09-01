#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
体验代码: 梯度下降 —— 改学习率/步数看收敛与发散
================================================
用途: 自己调参数,直观体会"学习率太大发散、太小太慢"。
      这是理解所有优化器(SGD/Adam)和训练调参的关键直觉。

用法:
  python3 demo.py                       # 默认:lr=0.1, 30步
  python3 demo.py --lr 0.3              # 大学习率,观察发散
  python3 demo.py --lr 0.01             # 小学习率,观察收敛慢
  python3 demo.py --lr 0.1 --steps 100  # 更多步数
  python3 demo.py --start 8 8           # 换起点
  python3 demo.py --compare             # 同时对比多个学习率

纯 Python,无需依赖。
"""
import argparse

def loss(x, y): return x*x + 3*y*y
def grad(x, y): return (2*x, 6*y)

def descend(lr, steps, start):
    x, y = start
    traj = [(x, y, loss(x, y))]
    diverged = False
    for _ in range(steps):
        gx, gy = grad(x, y)
        x -= lr * gx; y -= lr * gy
        l = loss(x, y)
        traj.append((x, y, l))
        if abs(l) > 1e8 or abs(x) > 1e4 or abs(y) > 1e4:
            diverged = True; break
    return traj, diverged

def fmt(traj, diverged, every=5):
    out = []
    for i, (x, y, l) in enumerate(traj):
        if i % every == 0 or i == len(traj)-1:
            bar = "▁▂▃▄▅▆▇█"[min(7, int(l / max(t[2] for t in traj) * 7))] if l > 0 else "█"
            out.append(f"  步{i:>3}: x={x:>+8.4f} y={y:>+8.4f} loss={l:>10.4f}")
    return "\n".join(out), diverged

def single(args):
    traj, div = descend(args.lr, args.steps, (args.start[0], args.start[1]))
    print("=" * 60)
    print(f"梯度下降: f(x,y)=x²+3y², 起点={args.start}, lr={args.lr}, 步数={args.steps}")
    print("=" * 60)
    txt, _ = fmt(traj, div)
    print(txt)
    print(f"\n--- 解读 ---")
    fx, fy, fl = traj[-1]
    if div:
        print(f"  ⚠ 发散!loss 越来越大,学习率 {args.lr} 太大了,一步跨过谷底反弹。")
    elif fl < 0.01:
        print(f"  ✓ 收敛成功!最终 loss={fl:.6f},已逼近最小值 (0,0)。")
    else:
        print(f"  还在收敛中,最终 loss={fl:.4f}。学习率小=稳但慢,可加步数或调大 lr。")
    print(f"  终点 = ({fx:.4f}, {fy:.4f})")

def compare(args):
    print("=" * 60)
    print("学习率对比:同一问题、不同 lr 的 30 步结果")
    print("=" * 60)
    print(f"  {'lr':>6} | {'最终loss':>12} | {'状态':<10} | {'loss下降轨迹(每5步)'}")
    print("  " + "-" * 56)
    for lr in [0.01, 0.1, 0.3, 0.5]:
        traj, div = descend(lr, 30, (args.start[0], args.start[1]))
        fl = traj[-1][2]
        status = "发散" if div else ("收敛" if fl < 0.01 else "进行中")
        spark = " ".join(f"{t[2]:.1f}" for t in traj[::5])
        print(f"  {lr:>6} | {fl:>12.4f} | {status:<10} | {spark}")

def main():
    p = argparse.ArgumentParser(description="梯度下降体验:改学习率看收敛/发散",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lr", type=float, default=0.1, help="学习率(默认0.1)")
    p.add_argument("--steps", type=int, default=30, help="迭代步数(默认30)")
    p.add_argument("--start", type=float, nargs=2, default=[4.0, 4.0], help="起点x y")
    p.add_argument("--compare", action="store_true", help="对比多个学习率")
    args = p.parse_args()
    if args.compare: compare(args)
    else: single(args)

if __name__ == "__main__":
    main()
