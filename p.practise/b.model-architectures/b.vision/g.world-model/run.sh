#!/bin/bash
# ============================================================
# 实验: g.world-model
# 说明: 世界模型 / 潜空间动力学 / 动作条件 / 未来帧预测
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# 世界模型(World Model,代表:Ha & Schmidhuber 2018 / Dreamer / GAIA-1 / Sora)解决的核心问题:
#   "不真正去试错,而是在脑里'想象'下一步会怎样,从而高效决策。"
# 关键组件:
#   1) 视觉编码器:把高维观测(O_t,例如 64x64 图)压缩成低维潜变量 z_t(例如 32 维);
#   2) 动力学 / RNN:用过去若干帧的潜状态 + 当前动作 a_t,预测下一帧的潜状态 z_{t+1};
#   3) 解码器:把想象出的 z_{t+1} 还原回像素,以可视化"脑内画面"。
# 通常在策略学习(RL/自动驾驶)里,模型会从 z_t 抽出 h_t,再经策略网络选动作;本实验只演示"想象"。
# 本实验用 numpy 模拟: 编码 → 动作条件 RNN → 动力学预测 → 解码 → 多步想象。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 世界模型 / 潜空间动力学 / 动作条件 RNN"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(0)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# 0 数据生成:一帧"会向左飘的圆"作为演示(主作为 z 语义)
# 真值潜状态: 2 维 z = [x_center, y_center]
# 动作 a: 1 维 dx(向左 = -1, 右 = +1, 不动 = 0)
T = 5
true_z = np.zeros((T, 2))
true_z[0] = [3.0, 4.0]
actions = np.array([0.0, -1.0, -1.0, +0.0, +1.0])
for t in range(1, T):
    true_z[t] = true_z[t-1] + np.array([actions[t], 0.0])
print(f"【0】真实潜状态轨迹(我们想预测的目标):")
for t in range(T):
    print(f"     t={t}: z={true_z[t].round(2).tolist()}, a={actions[t]:+.0f}  → 下一 z 应为 x += a")

# 1 视觉编码器:把 6x6 二值图(圆的位置)压成 2 维潜变量
def encode(img):
    # 极简:用两列不同位置模板的"匹配度"作为潜变量
    template_x = np.zeros((6, 6)); template_x[:, [1, 4]] = 0.5  # 在 x 方向有两个识别区
    template_y = np.zeros((6, 6)); template_y[[2, 3], :] = 0.5  # 在 y 方向有两个识别区
    z = np.array([(img * template_x).sum()*2.0,
                  (img * template_y).sum()*2.0])
    return z
def decode(z, R=6):
    img = np.zeros((R, R))
    cx = int(np.clip(round(z[0]), 0, R-1)); cy = int(np.clip(round(z[1]), 0, R-1))
    for di in range(-1, 2):
        for dj in range(-1, 2):
            i, j = cx+di, cy+dj
            if 0<=i<R and 0<=j<R:
                img[i, j] = 1.0
    return img

frames = np.zeros((T, 6, 6))
z_obs = np.zeros((T, 2))
for t in range(T):
    f = decode(true_z[t])
    frames[t] = f
    z_obs[t] = encode(f)
print(f"\n【1】视觉编码器:6x6 图 → 2 维潜变量 z(两条'模板匹配度')")
print(f"     观测到的 z_obs = {z_obs.round(2).tolist()}")
print(f"     解读:真实 encoder 通常是非线性、有损的——z_obs 是连续实数(可微),而 true_z 是用以绘制像素的位置坐标。")

# 2 动力学 (动作条件 RNN,极简版:h = z + 线性(a), 预测下一个 z)
# 动力学真实形式: z_{t+1} = z_t + W_a @ a_t
W_a = np.array([[1.0, 0.0],
                [0.0, 0.0]])      # 只影响 x 维度
print(f"\n【2】动力学模型:z_{{t+1}} = z_t + W_a·a_t (动作条件 RNN,极简)")
print(f"     动作依赖权重 W_a =\n{W_a}")

# 3 自回归 / 多步"想象"
z_imag = np.zeros((T, 2))
z_imag[0] = z_obs[0]
print(f"\n【3】在世界模型里'想象'未来 {T-1} 步:")
for t in range(1, T):
    z_imag[t] = z_imag[t-1] + W_a @ np.array([actions[t], 0.0])
    err = float(np.linalg.norm(z_imag[t] - true_z[t]))
    print(f"     t={t}: 想象 z={z_imag[t].round(2).tolist()}, 真值 z={true_z[t].round(2).tolist()}, ||err||={err:.3f}")

# 4 解码器可视化:"脑内画面" vs 真实画面
print("\n【4】把想象的 z 解码成像素,与真实观测对比:")
for t in range(T):
    a = decode(true_z[t]); b = decode(z_imag[t])
    same = int((np.abs(a-b).sum() == 0))
    print(f"     t={t}: 想象画面==真实?  {bool(same)}  (像素差={int(np.abs(a-b).sum())})")

# 5 演示"决策":假设拿到一个奖励 = -|想象 z_x - 2|(希望 x 接近 2)
print("\n【5】决策示意:在想象空间搜索'让 x 接近 2'的连续 3 步动作序列")
best = (1e9, None)
for a1 in [-1.0, 0.0, 1.0]:
    for a2 in [-1.0, 0.0, 1.0]:
        for a3 in [-1.0, 0.0, 1.0]:
            z = z_imag[0].copy()
            z = z + W_a @ np.array([a1, 0])
            z = z + W_a @ np.array([a2, 0])
            z = z + W_a @ np.array([a3, 0])
            cost = abs(z[0] - 2)
            if cost < best[0]:
                best = (cost, (a1, a2, a3))
print(f"     最优动作序列 = {best[1]}  最终想象 z={z.round(2).tolist()}  距离目标=2 → {best[0]:.3f}")
print(f"     解读:这就是世界模型的核心价值——不在真实环境里试错,而是在 z 空间里并行搜索。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:世界模型 = "在脑子里想象下一步会怎样":把图压成潜变量 z,在 z 空间里模拟动力学,
  再用想象结果帮助决策。Dreamer 系列、GAIA-1(自动驾驶)、Sora(视频生成)都是同一思路的不同体量。
- 熟手:动力学可以用 RNN/SSM/Transformer;损失由"想象 z 与实际下一帧 z 的差距"驱动;
  当动力学误差累积时会"幻觉化",需要加 KL 正则(对应 Dreamer 的 RSSM)、或者用扩散模型作动力学头(Sora);
  在线 RL 时还会用"世界模型生成的经验"再训策略,大幅降低对真实交互的需求。
- 延伸:把动作改成离散 4 方向;把潜维度加到 16 体会"可学到的细节";换成多帧输入(过去 K 步)看为何需要 RNN。
EOF
echo "============================================================"
