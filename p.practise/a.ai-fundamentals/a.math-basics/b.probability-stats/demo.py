#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
体验代码: 贝叶斯定理 —— 改先验/检测参数看后验怎么变
==================================================
用途: 自己调参数,体会"先验"和"检测准确率"如何影响"阳性=真患病"的概率。
      这是理解正则化(先验)、贝叶斯优化的直觉入口。

用法:
  python3 demo.py                          # 默认:患病率1%,敏感度99%,特异度95%
  python3 demo.py --prior 0.1              # 提高患病率到10%,看后验如何飙升
  python3 demo.py --prior 0.01 --spec 0.99 # 罕见病但检测更准,看是否还是虚惊
  python3 demo.py --prior 0.5 --sens 0.8   # 患病率高但敏感度低
  python3 demo.py --sweep prior            # 扫描先验,看后验随患病率的变化曲线
  python3 demo.py --sweep spec             # 扫描特异度,看检测精度的影响

纯 Python,无需任何依赖。
"""
import argparse

def bayes(prior, sens, spec):
    p_pos = sens * prior + (1 - spec) * (1 - prior)
    if p_pos == 0: return 0.0, 0.0
    post = sens * prior / p_pos
    return post, p_pos

def single(args):
    post, p_pos = bayes(args.prior, args.sens, args.spec)
    print("=" * 56)
    print("贝叶斯:阳性检测结果,到底有多大概率真患病?")
    print("=" * 56)
    print(f"  先验 P(患病)        = {args.prior}")
    print(f"  敏感度 P(阳性|患病) = {args.sens}  (真患者被检出的概率)")
    print(f"  特异度 P(阴性|健康) = {args.spec}  (健康人被正确排除的概率)")
    print(f"  假阳率 P(阳性|健康) = {1-args.spec:.4f}  (= 1 - 特异度)")
    print(f"  全概率 P(阳性)      = {p_pos:.4f}")
    print(f"  后验 P(患病|阳性)   = {post:.4f}  → {post*100:.1f}%\n")
    print("--- 解读 ---")
    if post > 0.8:
        verdict = "很高!说明检测可信(通常因为患病率高或检测极准)"
    elif post > 0.3:
        verdict = "中等,需要复检确认"
    else:
        verdict = "很低!阳性多半是虚惊(罕见病+假阳性占多数)"
    print(f"  阳性后真患病概率 {post*100:.1f}% —— {verdict}")
    print(f"  关键:假阳率({1-args.spec:.2%})×健康人({1-args.prior:.0%}) "
          f"vs 敏感度({args.sens:.0%})×患者({args.prior:.0%})")

def sweep(args):
    print("=" * 56)
    print(f"扫描 {args.sweep}:后验 P(患病|阳性) 如何随该参数变化")
    print("=" * 56)
    vals = {
        "prior": [0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5],
        "spec":  [0.90,0.95,0.97,0.99,0.995,0.999,0.9999],
        "sens":  [0.50,0.70,0.80,0.90,0.95,0.99,1.0],
    }[args.sweep]
    print(f"  {'参数值':>10} | {'后验概率':>10} | {'直方图':<30}")
    print("  " + "-" * 54)
    for v in vals:
        kw = {"prior":v,"spec":v,"sens":v}[args.sweep]
        post,_ = bayes(
            args.prior if args.sweep!="prior" else v,
            args.sens  if args.sweep!="sens"  else v,
            args.spec  if args.sweep!="spec"  else v)
        bar = "█" * int(post * 30)
        print(f"  {v:>10.4f} | {post*100:>9.1f}% | {bar}")

def main():
    p = argparse.ArgumentParser(
        description="贝叶斯定理体验:改参数看后验变化",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prior", type=float, default=0.01, help="先验患病率(默认0.01)")
    p.add_argument("--sens", type=float, default=0.99, help="敏感度 P(阳性|患病)(默认0.99)")
    p.add_argument("--spec", type=float, default=0.95, help="特异度 P(阴性|健康)(默认0.95)")
    p.add_argument("--sweep", choices=["prior","spec","sens"], default=None,
                   help="扫描某参数看后验变化曲线")
    args = p.parse_args()
    if args.sweep: sweep(args)
    else: single(args)

if __name__ == "__main__":
    main()
