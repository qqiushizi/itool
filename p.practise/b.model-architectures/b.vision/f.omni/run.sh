#!/bin/bash
# ============================================================
# 实验: f.omni
# 说明: 全模态(Omni) / 文本+图像+音频 token 统一 / 跨模态注意力
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# 全模态(Omni)模型(GPT-4o、Gemini、Qwen2.5-Omni 等)的核心思想:
#   - 把不同模态都"翻译"成同一种 Token:文本 → 子词 BPE、图像 → ViT patch、音频 → 频谱帧
#   - 在向量空间里它们长度不同,但维度相同(d_model),于是可以拼成一条长序列
#   - 用一个统一的 Transformer Decoder 做"下一个 token 预测"(Next-Token Prediction)
#   - 不同模态之间靠 cross-attention(或全注意力 self-attention)互相访问;
#     同时存在 <|text|> <|image|> <|audio|> 之类的"模态标签"告诉模型当前 token 来自哪种输入
# 本实验用 numpy 模拟:三种模态各自的 tokenizer → 统一 d 维 → 拼成一条序列 →
# 自注意力 → 读出文本/图像/音频各自下一步 token 的 logits。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 全模态 Omni / 统一 tokenizer / 跨模态注意力"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(0)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

# 1 三个 "tokenizer":文本、图像、音频 -> 都投影到 d_model 维
d_model = 16

# 文本 tokenizer:6 个 token,词表大小 9(含特殊标记)
text_vocab = 9
text_ids = [0, 3, 4, 5, 6, 8]   # <|text|> 你好 猫 在 叫 <|eos|>
text_emb = rng.standard_normal((text_vocab, d_model))
text_tok = text_emb[text_ids]            # [n_text, d]
print(f"【1】文本 tokenizer:把 6 个 id 投影到 {d_model} 维 → shape={text_tok.shape}")
print(f"     序列 = {text_ids} ([<|text|>, '你好', '猫', '在', '叫', <|eos|>])")

# 图像 tokenizer:把 8x8 图切成 4 个 4x4 patch
img = rng.standard_normal((8, 8))
P = 4
n_p = (8 // P) * (8 // P)
img_tok_list = []
for i in range(8 // P):
    for j in range(8 // P):
        img_tok_list.append(img[i*P:(i+1)*P, j*P:(j+1)*P].flatten())
img_tok_list = np.array(img_tok_list)
img_emb = rng.standard_normal((P*P, d_model))
img_tok = img_tok_list @ img_emb   # [n_img, d]
print(f"\n【2】图像 tokenizer:8x8 图 → {n_p} 个 patch → shape={img_tok.shape}")

# 音频 tokenizer:把 32 帧 8-bin 频谱展成 token 序列
spec = rng.standard_normal((32, 8))
aud_emb = rng.standard_normal((8, d_model))
aud_tok = spec @ aud_emb            # [32, d]
print(f"【3】音频 tokenizer:32 帧 x 8 bin 频谱 → 32 个 token → shape={aud_tok.shape}")

# 2 拼成一条长序列,前面加模态标签 + 位置编码
mod_tag_T = text_emb[0]   # <|text|> 标记
mod_tag_I = text_emb[1]   # <|image|> 标记(复用词表的特殊位)
mod_tag_A = text_emb[2]   # <|audio|> 标记
img_tok_tagged  = img_tok + mod_tag_I
text_tok_tagged = text_tok + mod_tag_T
aud_tok_tagged  = aud_tok + mod_tag_A

seq = np.vstack([img_tok_tagged, text_tok_tagged, aud_tok_tagged])
pos = rng.standard_normal(seq.shape) * 0.1
seq = seq + pos
L = seq.shape[0]
print(f"\n【4】三模态拼成一条序列:总长 L={L}({n_p} 图 + {len(text_ids)} 文 + {32} 音)")
print(f"     每 token 用模态标签(id加和)告诉模型'我来自哪个模态';再加位置编码。")

# 3 统一 Transformer Decoder:带因果 mask 的自注意力(下一 token 预测)
Q = seq @ rng.standard_normal((d_model, d_model))
K = seq @ rng.standard_normal((d_model, d_model))
V = seq @ rng.standard_normal((d_model, d_model))
mask = np.triu(np.ones((L, L)) * -1e9, k=1)   # 上三角屏蔽(看未来)
attn_logits = (Q @ K.T) / np.sqrt(d_model) + mask
A = softmax(attn_logits, axis=-1)
out = A @ V   # 每个位置都"看见"自己和之前的所有模态

last_text_pos = n_p + len(text_ids) - 1
visible = np.array([0.0]*L)
visible[:n_p] = 1.0
visible[n_p:last_text_pos] = 1.0
weights = A[last_text_pos]
img_w  = weights[:n_p].sum()
text_w = weights[n_p:n_p+len(text_ids)].sum()
audio_w = weights[n_p+len(text_ids):].sum()
print(f"\n【5】统一 Transformer Decoder:注意力的全局统计")
print(f"     '呢'(文本最后位置)可见 token 数:图={int(visible[:n_p].sum())}, 文={int(visible[n_p:n_p+len(text_ids)].sum())}, 音={int(visible[n_p+len(text_ids):].sum())}")
print(f"     随机参数 softmax 后关注占比(示意):图={img_w:.2f}, 文={text_w:.2f}, 音={audio_w:.2f}")
print(f"     解读:因果断让'呢'能读到前面所有 token — 这是 Omni 跨模态推理的结构基础;训练后权重会把注意力聚焦到相关的图/音 token 上。")

# 4 模态专用输出头:从同一隐藏状态,分别预测下一图像 token / 文本 token / 音频 token
v_text, v_img, v_aud = text_vocab, n_p, 32
W_text = rng.standard_normal((d_model, v_text))
W_img  = rng.standard_normal((d_model, v_img))
W_aud  = rng.standard_normal((d_model, v_aud))

pos_img  = 0
pos_text = n_p + 2   # '猫' 那一格,让它预测下一个文本 token
pos_aud  = n_p + len(text_ids) + 5
logits_text = out[pos_text] @ W_text
logits_img  = out[pos_img]  @ W_img
logits_aud  = out[pos_aud]  @ W_aud
probs_text = softmax(logits_text); probs_img = softmax(logits_img); probs_aud = softmax(logits_aud)
print(f"\n【6】同一隐藏状态,三种输出头(解嵌矩阵不同):")
print(f"     图像位置预测下一图像 patch token:argmax={int(np.argmax(probs_img))}, maxP={probs_img.max():.2f}")
print(f"     文本位置'猫'预测下一文本 token :argmax={int(np.argmax(probs_text))}, maxP={probs_text.max():.2f}")
print(f"     音频位置预测下一音频帧 token    :argmax={int(np.argmax(probs_aud))}, maxP={probs_aud.max():.2f}")
print(f"     解读:同一个 Decoder 串起三种 head,共享表征 + 各自输出 —— Omni 的核心架构。")

params_attn = d_model*d_model*4
params_head = d_model*(v_text+v_img+v_aud)
params_total = params_attn + params_head + d_model*3
print(f"\n【7】共用参数(注意力 QKV+O):{params_attn}; 三类 head 共:{params_head}; 简略合计≈{params_total:,}")
print(f"     解读:模态之间共享主干,只在 token 化和 head 处分流 → 比每个模态单独训练更省参数、更容易迁移。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:全模态 = "把文字、图、声音都切成小片,变成等长向量,扔进同一个 Transformer"。
  模态标签 + 共享主干 + 三类输出头,模型就能同时理解多种输入并任意一种输出。
- 熟手:重点不是统一 token,而是统一"学习目标"——Next-Token Prediction。把图像 patch 序列、文本
  token 序列、音频帧序列交错训练可以产生"涌现"的多模态能力;GPT-4o / Gemini / Qwen-Omni 都用
  这条思路,但实现上常用 cross-attention 与流式语音 codec。
- 延伸:试着把三种序列顺序打乱(交织 vs 单段);思考为什么全自注意力对长序列代价 O(L^2)。
EOF
echo "============================================================"
