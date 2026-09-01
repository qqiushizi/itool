#!/bin/bash
# ============================================================
# 实验: e.dit
# 说明: Diffusion Transformer / adaLN-Zero / patchify 去噪
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# 扩散模型(Diffusion):反复加噪→学会"去噪"。DiT 把 U-Net 换成 Transformer:
# 1) 把噪声图切成 patch(和 ViT 一样),得到一串 token;
# 2) 把"当前时间步 t"和"类别条件 c"拼成条件向量,通过 adaLN-Zero 注入每一层——
#    即从条件向量预测每层的 LayerNorm 的 γ、β,以及最后的缩放因子 α,
#    让条件信息"调制"注意力与前馈网络;
# 3) 最后一层把 token 再 reshape 回空间,得到去噪后的图。
# 本实验用 numpy 模拟:patchify → 加时间步+类别条件 → adaLN-Zero 调制 → 还原图像。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: DiT / adaLN-Zero / patchify 去噪"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(0)

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=-1, keepdims=True)

# 0 准备一张 8x8 噪声图
img = rng.standard_normal((8, 8))
H = W = 8; P = 4
n_patches = (H // P) * (W // P)
print(f"【0】输入:一张加噪后的 8x8 图(扩散过程中某一步的 x_t)")

# 1 patchify:把图切成 4x4 patch,展平 -> 线性投影到 d 维
d = 8
patches = []
for i in range(H // P):
    for j in range(W // P):
        patches.append(img[i*P:(i+1)*P, j*P:(j+1)*P].flatten())
patches = np.array(patches)
E = rng.standard_normal((P*P, d))
tokens = patches @ E   # [n_patches, d]
print(f"【1】patchify:{n_patches} 个 {P}x{P} patch -> 线性投影成 {d} 维 token,序列 shape={tokens.shape}")
print("  解读:DiT 把噪声图当 ViT 处理 -- 同样的 patch 嵌入,只是后面接的去噪头不一样。")

# 2 条件:时间步 t(标量) + 类别 c(整数) -> emb 联合向量
t_emb = rng.standard_normal(d)        # 时间步正弦位置编码的简化版
c_emb = rng.standard_normal(d)        # 类别嵌入
cond = t_emb + c_emb                   # 融合的条件向量
print(f"\n【2】条件注入:时间步 t={0.7} (图中加了几步噪) + 类别='猫'")
print(f"  条件向量 cond 由 t 嵌入 + 类别嵌入相加得到,shape={cond.shape}")
print("  解读:DiT 必须知道当前去到第几步、要生成什么类,才能有方向地降噪。")

# 3 adaLN-Zero:由 cond 预测每层的 gamma, beta, alpha
def adaLN(x, cond):
    # 简化版:cond 通过线性层预测 gamma, beta, alpha(初始 alpha=0,所以叫 'Zero')
    gamma = cond @ rng.standard_normal((d, d))
    beta  = cond @ rng.standard_normal((d, d)) * 0.1
    alpha = float(cond @ rng.standard_normal(d))   # 残差缩放,初值近似 0
    mu, sigma = x.mean(-1, keepdims=True), x.std(-1, keepdims=True)+1e-6
    x_norm = (x - mu) / sigma
    return x_norm * (1 + gamma) + beta, alpha

# 4 Transformer Block x L (L=2 演示)
L = 2
x = tokens
for layer in range(L):
    # 4a Self-Attention (示意,固定参数)
    Q = x @ rng.standard_normal((d, d))
    K = x @ rng.standard_normal((d, d))
    V = x @ rng.standard_normal((d, d))
    A = softmax(Q @ K.T / np.sqrt(d))
    sa = A @ V
    # 4b adaLN-Zero 把 cond 注入到 attention 输出前
    sa_mod, alpha_sa = adaLN(sa, cond)
    x = x + alpha_sa * sa_mod
    # 4c FFN
    ff = np.maximum(0, x @ rng.standard_normal((d, 4*d))) @ rng.standard_normal((4*d, d))
    ff_mod, alpha_ff = adaLN(ff, cond)
    x = x + alpha_ff * ff_mod
    print(f"  第{layer}层:alpha_sa={alpha_sa:+.2f}, alpha_ff={alpha_ff:+.2f}")

print("\n【3】adaLN-Zero:每层由 cond 预测 gamma、beta (调制 LayerNorm) 和 alpha (残差缩放)")
print("  解读:alpha 初值为 0 -> 一开始整层几乎不变,跳过梯度噪声;训练几步后才'解封',这是 DiT 的关键技巧。")
print("        条件信息通过每层重新调制,比简单把 cond 加到 token 上更有效。")

# 5 把 token 解 patchify 回图像
unproj = x @ E.T                    # [n_patches, P*P]
out = np.zeros((H, W))
for i in range(H // P):
    for j in range(W // P):
        idx = i*(W//P) + j
        out[i*P:(i+1)*P, j*P:(j+1)*P] = unproj[idx].reshape(P, P)

# 6 简单"加噪-去噪"演示:拿一张干净图,加噪,再用一个非常粗糙的"去噪"模型输出
clean = np.sin(np.linspace(0, 2*np.pi, 8*8)).reshape(8, 8)
noise = rng.standard_normal((8, 8)) * 1.5
noisy = clean + noise
print("\n【4】加噪->去噪示意(同张图不同 t):")
for t in [0.1, 0.5, 0.9]:
    # 简化:模型输出 = noisy * (1-t) 近似逐渐恢复干净图
    est = noisy * (1 - t)
    err = np.abs(est - clean).mean()
    print(f"  t={t:.1f} -> 估计图与原图平均误差={err:.3f}")
print("  解读:真实 DiT 输出的不是 (1-t)*noisy,而是从噪声直接预测 x0 或噪声 epsilon,这里只是直觉示意。")

print(f"\n【5】输出:把 token 线性反投影回 {H}x{W} 的去噪图 patch 网格(本例 shape={out.shape})")
print("  解读:DiT 去噪头本质就是 patchify 的逆运算,把 token 序列再 reshape 回图像。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:DiT = ViT + 扩散 = 把"加噪去噪"这件事交给 Transformer 而不是 U-Net;条件(时间步+类别)通过
  adaLN-Zero 注入每层,一开始几乎不做事(alpha=0),训练稳定后再发挥作用。
- 熟手:adaLN 比 cross-attention 更轻量、比 in-context conditioning 更强;DiT-XL/2 是 SOTA 图像生成骨干;
  DiT 同样适合视频(逐帧 patchify)、latent 扩散(在 VAE 潜空间里 patchify)即 Stable Diffusion 3。
- 延伸:把 cond 改成"无类别"看 unconditional 生成;把 patch 从 4x4 调到 2x2 体会序列长度变化。
EOF
echo "============================================================"
