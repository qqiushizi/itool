#!/bin/bash
# ============================================================
# 实验: b.data-pipeline
# 说明: 数据打包、packing、mask 构造
# 模块: p.practise/c.llm-training  LLM 训练
# ============================================================
# 【第一性原理】
# 预训练要把海量文本变成定长序列喂模型。流程:分词→拼成长流→按 max_seq_len 切块(packing)→构造注意力 mask。
# packing:把多个短文档拼进一个固定长度块,提高利用率(否则 padding 浪费算力)。
# mask:因果 mask(每 token 只看前面)+ 文档边界 mask(packing 时跨文档不能互相看),防止"串文"。
# 本实验模拟把几段文本打包成定长块,并构造因果+边界的注意力 mask。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 数据管线 / packing / 因果+边界 mask"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=0, suppress=True, linewidth=120)
# 几段"文档"(用 token id 表示)
docs=[[1,2,3],[4,5,6,7],[8,9],[10,11,12,13,14]]
seq_len=8
print("【1】packing:把多个短文档拼进定长块(提高利用率,避免 padding 浪费)")
packed=[]; buf=[]
for d in docs:
    buf+=d
    while len(buf)>=seq_len:
        packed.append(buf[:seq_len]); buf=buf[seq_len:]
if buf: packed.append(buf+[0]*(seq_len-len(buf)))   # 末尾不足补 padding(0)
for i,blk in enumerate(packed):
    print(f"  块{i}: {blk}")
print(f"  利用率:有效token={sum(len(d) for d in docs)}/{len(packed)*seq_len} = {sum(len(d) for d in docs)/(len(packed)*seq_len)*100:.0f}%")
print("  解读:packing 把短文档拼满定长块,几乎不浪费;末尾不足才 padding。→ 大幅提升训练吞吐。")

# 2 因果 mask + 文档边界 mask
print("\n【2】注意力 mask:因果(只看过去)+ 文档边界(packing 时跨文档不可见)")
# 块0=[1,2,3 | 4,5,6,7 | 8] 来自文档0(pos0-2)、文档1(pos3-6)、文档2(pos7)
doc_ids=[0,0,0,1,1,1,1,2]   # 同文档同号,跨文档不同号
causal=np.tril(np.ones((seq_len,seq_len),int))   # 下三角=因果
boundary=np.zeros((seq_len,seq_len),int)
for i in range(seq_len):
    for j in range(seq_len):
        if doc_ids[i]>=0 and doc_ids[j]>=0 and doc_ids[i]!=doc_ids[j]:
            boundary[i,j]=1
        if doc_ids[i]<0 or doc_ids[j]<0: boundary[i,j]=1   # padding 全屏蔽
mask=causal*(1-boundary)
print("  因果mask(下三角):"); print("  ",np.array2string(causal,separator=' '))
print("  最终mask(因果×去边界×去padding):"); print("  ",np.array2string(mask,separator=' '))
print("  解读:文档0(pos0-2)只看自己过去,文档1(pos3-6)只看自己过去,文档2(pos7)只看自己→跨文档不互看,防止 packing 串文。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:预训练数据=分词→拼流→定长切块(packing)→构造 mask;packing 提利用率,mask 防止跨文档和未来泄漏。
- 熟手:packing 带来跨文档注意力,需文档边界 mask(或 position id 重置);padding token 不参与 loss 和 attention;
  高质量数据管线(去重、过滤、配比)对预训练效果影响极大,常比模型结构更关键。
- 延伸:把 seq_len 调大看 packing 块数减少;去掉边界 mask 思考为何会"串文"。
EOF
echo "============================================================"
