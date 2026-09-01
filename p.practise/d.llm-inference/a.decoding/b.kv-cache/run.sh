#!/bin/bash
# ============================================================
# 实验: b.kv-cache
# 说明: KV 缓存显存建模、增长与命中
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 自回归解码每步要算新 token 对所有历史 token 的 attention。
# 没 KV cache:每步重算 O(n²) FLOPs,无法接受。
# 有 KV cache:每步只算新 token 的 Q,缓存历史的 K, V
#   - 每步新增: 2 个 (n, d_k) 张量
#   - 总缓存: 2 * L * d_model (FP16 时 = 2*4 = 8B/参/层)
# 公式(单请求):
#   KV cache = 2 * batch * n_layers * seq_len * n_heads * head_dim * dtype_bytes
# 例:7B 模型(L=32, h=32, d_head=128), seq=4096, FP16:
#   = 2 * 1 * 32 * 4096 * 32 * 128 * 2 = 2.1 GB (单请求)
# 100 个并发请求: 210 GB → 装不下,这就是 PagedAttention 的动机。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: b.kv-cache | KV 缓存显存建模:增长、命中、复用"
echo "############################################################"

python3 <<'PY'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. KV cache 显存公式 ---
hdr(1,TOTAL,"KV cache 显存公式:为什么 batch×seq 涨得快")
why("""单请求 KV cache 大小:
  KV = 2 * n_layers * n_heads * head_dim * seq_len * dtype_bytes
  系数 2 = K 和 V
  系数 n_layers = 每层都要存一份
FP16 时 = 4B / token / 层 / head
所以:KV 增长正比于 n_layers × seq_len × batch""")
def kv_gb(n_layers, n_heads, head_dim, seq, batch, dtype_bytes=2):
    return 2 * n_layers * n_heads * head_dim * seq * batch * dtype_bytes / 1e9

# 7B 配置
L, h, d_h = 32, 32, 128
for seq in [512, 2048, 4096, 8192, 32768]:
    for batch in [1, 8, 32, 128]:
        gb = kv_gb(L, h, d_h, seq, batch)
        print()
res(f"""LLaMA-7B 配置: L={L}, h={h}, d_h={d_h}, FP16
  seq   \\ batch   1      8       32      128
  512           {kv_gb(L,h,d_h,512,1):.2f}    {kv_gb(L,h,d_h,512,8):.2f}    {kv_gb(L,h,d_h,512,32):.2f}    {kv_gb(L,h,d_h,512,128):.2f} GB
  2048          {kv_gb(L,h,d_h,2048,1):.2f}    {kv_gb(L,h,d_h,2048,8):.2f}    {kv_gb(L,h,d_h,2048,32):.2f}    {kv_gb(L,h,d_h,2048,128):.2f} GB
  4096          {kv_gb(L,h,d_h,4096,1):.2f}    {kv_gb(L,h,d_h,4096,8):.2f}    {kv_gb(L,h,d_h,4096,32):.2f}    {kv_gb(L,h,d_h,4096,128):.2f} GB
  8192          {kv_gb(L,h,d_h,8192,1):.2f}    {kv_gb(L,h,d_h,8192,8):.2f}    {kv_gb(L,h,d_h,8192,32):.2f}    {kv_gb(L,h,d_h,8192,128):.2f} GB
  32768         {kv_gb(L,h,d_h,32768,1):.2f}    {kv_gb(L,h,d_h,32768,8):.2f}    {kv_gb(L,h,d_h,32768,32):.2f}    {kv_gb(L,h,d_h,32768,128):.2f} GB""")
mea("""batch×seq 是显存杀手:7B 模型 128 并发 + 8K context → 168 GB。
这就是为什么 KV cache 优化(量化、PagedAttention)是推理框架核心命题。""")

# --- 2. 不同模型对比 ---
hdr(2,TOTAL,"不同模型的 KV cache 总量")
why("""不同架构 KV cache 大小差很多:
  - 7B (32 层):   baseline
  - 70B (80 层):  2.5× 7B
  - MoE:        比 dense 多,因为专家层 KV 不分摊
  - MQA/GQA:    head_dim 不变但 n_kv_head 减少 → KV 缩 4-8×""")
configs = {
    "LLaMA-7B  (h=32, d=128)": (32, 32, 128),
    "LLaMA-70B (h=64, d=128)": (80, 8, 128),    # GQA: 8 KV heads
    "LLaMA-70B-MHA": (80, 64, 128),
    "Mistral-7B (GQA, h=8)": (32, 8, 128),
    "Qwen-72B GQA": (80, 8, 128),
}
res("模型              seq=4096, batch=1 KV cache GB:")
for name, (L, h, d_h) in configs.items():
    gb = kv_gb(L, h, d_h, 4096, 1)
    print(f"  {name:30s} {gb:.2f} GB")
mea("""GQA(Grouped Query Attention)就是为省 KV:MHA 的 K,V head 数 = Q head 数;
GQA 让 K,V head 减到 Q 的 1/4~1/8,K/V 缓存同步减少 4-8×。
现代大模型(LLaMA-2/3、Mistral、Qwen)基本都用 GQA。""")

# --- 3. 命中与复用:prefix cache ---
hdr(3,TOTAL,"Prefix cache:system prompt 不重算")
why("""真实场景 system prompt 经常一样 (\"你是 helpful 助手...\"),不同请求共用。
前缀缓存:把已算过的 K,V 存住,新请求如果共享前缀,直接复用。
实现:prefix tree (RadixAttention) 或 hash table。
收益:省 prefill 时间和显存(共用那段不重存)。""")
sys_prompt = "你是 helpful 助手, 请回答用户问题。" * 50   # 约 500 token
sys_len = 500
res(f"""System prompt = {sys_len} tokens, 1 个 user 提问 = 100 tokens
  无 prefix cache: 每次 prefill = 600 tokens, 占 KV = 600×显存
  有 prefix cache: 第一次算 600,之后只用算 100, 复用前 500 tokens
  加速:  PREFILL FLOPs 减少 {600/100:.1f}×
  显存:  复用后系统 prompt 段只存一份""")
mea("""SGLang 的 RadixAttention / vLLM 的 prefix cache 都在做这件事:
  - 自动检测 system prompt 重叠
  - 算过的 token 不重算 K,V
  - 长 system prompt 场景(几万 token)收益极大""")

# --- 4. 量化与压缩 ---
hdr(4,TOTAL,"KV cache 量化:从 FP16 → INT4")
why("""FP16 KV cache 占 2B/token/层/head;量化到 INT8 = 1B,INT4 = 0.5B。
INT4 精度有损,但 cache 反复读(尤其 attention),误差能摊薄。
现代方案:KV cache 量化 (FP8/INT8/INT4)、分页(PagedAttention)、跨层共享。""")
def kv_size(seq, batch, bytes_per_elem, n_layers=32, n_heads=32, d_h=128):
    return 2 * n_layers * n_heads * d_h * seq * batch * bytes_per_elem / 1e9
res(f"""LLaMA-7B, seq=4096, batch=128:
  FP16 KV: {kv_size(4096,128,2):.1f} GB
  FP8  KV: {kv_size(4096,128,1):.1f} GB   (-50%)
  INT8 KV: {kv_size(4096,128,1):.1f} GB   (-50%)
  INT4 KV: {kv_size(4096,128,0.5):.1f} GB (-75%)""")
mea("""KV 量化是有损的(尤其 INT4),但实际 perplexity 损失 < 0.1。
  生产环境:INT8 KV + FP16 模型,既省显存又几乎不掉点。
  vLLM 已支持 --kv-cache-dtype=int8。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:KV cache = 把每步算过的 K,V 存下来,避免重算;显存随 batch×seq
  线性涨,所以 batch 大时是大头;GQA/prefix cache/量化都能省。
- 熟手:长 system prompt 必须开 prefix cache;batch 大必须开 PagedAttention;
  KV 量化 INT8 几乎不掉点;Multi-Query 注意力是现代 LLM 标配省 KV 手段。
【进阶】vLLM 实测开/关 --enable-prefix-caching 的吞吐差距;INT8 KV 跑 CEval 看掉点。
EOF
echo "############################################################"
