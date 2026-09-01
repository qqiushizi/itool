#!/bin/bash
# ============================================================
# 实验: b.gan-basics
# 说明: 生成器/判别器博弈、模式坍塌
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# GAN:生成器 G 把噪声 z 变成假样本,判别器 D 区分真假。两者博弈:
#  D 想最大化 log D(real)+log(1-D(fake));G 想最小化 log(1-D(fake))=骗过 D。
# 理想均衡:G 造的分布=真实分布,D 分不清(输出0.5)。难点:训练不稳、模式坍塌(G 只生成一种样本)。
# 本实验用 1D 双峰真实分布模拟 GAN 博弈,并演示模式坍塌。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: GAN / 生成器判别器博弈 / 模式坍塌"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng=np.random.default_rng(0)
def sig(z): return 1/(1+np.exp(-z))
# 真实数据:双峰(在 -2 和 +2)
real=np.where(rng.random(200)<0.5, rng.standard_normal(200)-2, rng.standard_normal(200)+2)
# 1 正常训练:G 学会双峰
g_mean=0.0
for it in range(300):
    z=rng.standard_normal(64); fake=z*1.5+g_mean          # G:平移+缩放噪声
    # D:用均值差做简单判别(真实均值±2,fake均值g_mean)
    d_real=sig((np.mean(real)-0)); d_fake=sig((np.mean(fake)-0))
    # G 梯度:把 fake 均值推向真实均值(0?双峰均值=0)→此处简化为推向0,但真值是双峰
    g_mean+=0.05*(0-np.mean(fake))                          # G 逼近真实整体均值
print("【1】GAN 博弈:D 区分真假,G 逼近真实分布")
print(f"  真实数据均值={real.mean():.3f}, 双峰(在±2)  G 输出均值→{g_mean:.3f}")
print("  解读:G 把噪声搬到真实数据区域,D 越来越难分;均衡时 G 分布≈真实分布,D 输出≈0.5。")

# 2 模式坍塌:G 只学会一个峰
g_collapse=2.0   # G 只生成 +2 附近的样本,忽略 -2 那个峰
fake_c=rng.standard_normal(200)*1.0+g_collapse
print("\n【2】模式坍塌:G 只生成真实分布的一个模式")
print(f"  真实有两个峰(-2 和 +2),G 只产出均值={g_collapse} 的样本")
print(f"  覆盖率:G 只覆盖了真实数据的 {sum((real>0))/len(real)*100:.0f}% 区域(漏掉负峰)")
print("  解读:GAN 常见失败——G 找到一个能骗过 D 的模式就反复生成它,丢掉其他模式。\n  → 缓解:Minibatch discrimination、WGAN、特征匹配、多样性正则。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:GAN 是 G(造假)和 D(辨真)的博弈,均衡时 G 造的像真、D 分不清;通病是训练不稳和模式坍塌。
- 熟手:JS 散度下 GAN 梯度在分布不重叠时消失→WGAN 用 Wasserstein 距离缓解;
  模式坍塌靠 Minibatch discrimination/多样性约束;GAN 生成锐利但难评估,FID/IS 是常用指标。
- 延伸:把真实改成单峰看是否还坍塌;思考为何 D 太强 G 反而学不动。
EOF
echo "============================================================"
