#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
体验代码: 线性回归 —— 改噪声/特征数看三种解法表现
==================================================
用法:
  python3 demo.py                       # 默认
  python3 demo.py --noise 5             # 加大噪声,看估计偏差
  python3 demo.py --features 20         # 高维,对比正规方程与梯度下降速度
  python3 demo.py --noise 0             # 无噪声,看能否精确还原
  python3 demo.py --steps 100           # 减少梯度下降步数,看是否收敛
"""
import argparse, time
try:
    import numpy as np
    HAVE_NP = True
except ImportError:
    HAVE_NP = False

def run(args):
    if not HAVE_NP:
        print("建议安装 numpy: pip install numpy"); return
    rng = np.random.default_rng(args.seed)
    n, d = args.samples, args.features
    w_true = rng.standard_normal(d)                 # 真实权重
    X = rng.standard_normal((n, d))
    X = np.column_stack([X, np.ones(n)])            # 加偏置列
    w_full = np.append(w_true, 1.0)
    y = X @ w_full + rng.standard_normal(n) * args.noise

    print("="*60)
    print(f"线性回归: 样本{n}, 特征{d}, 噪声σ={args.noise}, 梯度步数{args.steps}")
    print("="*60)

    # 正规方程
    t0=time.time(); w_norm=np.linalg.inv(X.T@X)@X.T@y; t_norm=time.time()-t0
    err_norm=np.max(np.abs(w_norm-w_full))
    # 梯度下降
    w=np.zeros(d+1); lr=0.01; t0=time.time()
    for _ in range(args.steps):
        w-=lr*X.T@(X@w-y)/n
    t_gd=time.time()-t0; err_gd=np.max(np.abs(w-w_full))
    # 最小二乘
    t0=time.time(); w_lsq,*_=np.linalg.lstsq(X,y,rcond=None); t_lsq=time.time()-t0
    err_lsq=np.max(np.abs(w_lsq-w_full))

    print(f"  {'方法':<12} | {'最大参数误差':>12} | {'耗时(秒)':>10}")
    print("  "+"-"*44)
    print(f"  {'正规方程':<12} | {err_norm:>12.4f} | {t_norm:>10.5f}")
    print(f"  {'梯度下降':<12} | {err_gd:>12.4f} | {t_gd:>10.5f}")
    print(f"  {'最小二乘':<12} | {err_lsq:>12.4f} | {t_lsq:>10.5f}")
    print("\n--- 解读 ---")
    if args.noise == 0:
        print("  无噪声:三种方法误差都应≈0(能精确还原真值)")
    else:
        print(f"  噪声σ={args.noise}:误差主要来自噪声,非方法缺陷。噪声越大估计越偏。")
    if d >= 20: print(f"  高维(d={d}):注意正规方程耗时——特征多时求逆变慢。")
    if args.steps < 500: print(f"  梯度下降仅{args.steps}步,可能未收敛,加大 --steps 看误差下降。")

def main():
    p=argparse.ArgumentParser(description="线性回归三法对比体验")
    p.add_argument("--noise",type=float,default=2.0,help="噪声标准差")
    p.add_argument("--features",type=int,default=2,help="特征维度")
    p.add_argument("--samples",type=int,default=50,help="样本数")
    p.add_argument("--steps",type=int,default=2000,help="梯度下降步数")
    p.add_argument("--seed",type=int,default=7,help="随机种子")
    run(p.parse_args())

if __name__=="__main__": main()
