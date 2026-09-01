#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
体验代码: 交叉熵 —— 改预测概率看损失大小
==========================================
用途: 自己调参数,体会"预测越准交叉熵越小,自信地错损失爆炸"。
      理解为什么分类任务用交叉熵而不是 MSE。

用法:
  python3 demo.py                       # 默认:对比几个预测概率
  python3 demo.py --true 1 --pred 0.99  # 几乎全对,看交叉熵多小
  python3 demo.py --true 1 --pred 0.01  # 自信地错,看交叉熵爆炸
  python3 demo.py --true 1 --pred 0.5   # 摇摆不定
  python3 demo.py --mse                 # 同时算 MSE 对比,理解为何不用 MSE

纯 Python,无需依赖。
"""
import math, argparse

def cross_entropy(true, pred):
    pred = max(min(pred, 1-1e-12), 1e-12)   # 裁剪避免 log(0)
    if true == 1: return -math.log2(pred)
    return -math.log2(1 - pred)

def mse(true, pred): return (pred - true) ** 2

def single(args):
    ce = cross_entropy(args.true, args.pred)
    print("=" * 56)
    print(f"二分类交叉熵:真实标签={args.true}, 预测概率={args.pred}")
    print("=" * 56)
    print(f"  交叉熵 = -log2({args.pred}) = {ce:.4f} 比特")
    if args.mse:
        m = mse(args.true, args.pred)
        print(f"  MSE    = ({args.pred}-{args.true})² = {m:.4f}")
    print(f"\n--- 解读 ---")
    if args.true == 1:
        if args.pred > 0.9:
            print(f"  预测 {args.pred} 很接近真实 1,交叉熵 {ce:.3f} 很小 ✓")
        elif args.pred > 0.4:
            print(f"  预测 {args.pred} 摇摆,交叉熵 {ce:.3f} 中等")
        else:
            print(f"  ⚠ 预测 {args.pred} 却真实为 1,交叉熵 {ce:.3f} 爆炸!模型「自信地错」被重罚")
    if args.mse:
        print(f"\n  对比:此例交叉熵={ce:.3f} 而 MSE={mse(args.true,args.pred):.4f}")
        print(f"  交叉熵对「自信地错」惩罚指数级重,MSE 只是平方级——这是分类用交叉熵的原因。")

def compare(args):
    print("=" * 56)
    print("预测概率 vs 交叉熵 (真实标签=1)")
    print("=" * 56)
    print(f"  {'预测概率':>8} | {'交叉熵':>10} | {'MSE':>8} | {'惩罚曲线'}")
    print("  " + "-" * 50)
    for pred in [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001]:
        ce = cross_entropy(1, pred); m = mse(1, pred)
        bar = "█" * min(40, int(ce * 3))
        print(f"  {pred:>8.3f} | {ce:>10.3f} | {m:>8.4f} | {bar}")
    print("\n  → 预测从 0.99→0.001,交叉熵从 0.014→9.97(指数爆炸),MSE 只从 0→0.998(平缓)")
    print("  → 这就是交叉熵的价值:逼模型「别自信地犯错」")

def main():
    p = argparse.ArgumentParser(description="交叉熵体验:改预测概率看损失",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--true", type=int, default=1, choices=[0,1], help="真实标签(0或1)")
    p.add_argument("--pred", type=float, default=0.7, help="预测概率(0~1)")
    p.add_argument("--mse", action="store_true", help="同时显示 MSE 对比")
    p.add_argument("--compare", action="store_true", help="对比一系列预测概率")
    args = p.parse_args()
    if args.compare: compare(args)
    else: single(args)

if __name__ == "__main__":
    main()
