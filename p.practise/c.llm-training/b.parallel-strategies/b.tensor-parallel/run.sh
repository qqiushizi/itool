#!/bin/bash
# ============================================================
# 实验: b.tensor-parallel
# 说明: TP 列/行切分、通信量分析
# 模块: p.practise/c.llm-training  LLM 训练
# ============================================================
# 【第一性原理】
# 张量并行(TP):单卡放不下模型时,把单个层的权重矩阵切到多卡上协同计算。
#  列并行:权重 W 按"列"切,每卡算 X·W_i 得部分结果,再 AllGather 拼回;
#  行并行:权重 W 按"行"切,每卡用部分输入算,再 AllReduce 求和。
# Megatron 巧妙组合:第一个 Linear 列切(免通信接激活),第二个 Linear 行切(只需一次 AllReduce)。
# 本实验演示列/行切分,并算通信量。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 张量并行 / 列切 / 行切 / 通信量"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng=np.random.default_rng(0)
X=rng.standard_normal((2,4)); W=rng.standard_normal((4,6))
full=X@W   # 单卡参考结果
# 1 列并行:W 按列切到2卡,各算 X·W_i,AllGather 拼列
W0,W1=W[:,:3],W[:,3:]
Y0,Y1=X@W0,X@W1
col=np.hstack([Y0,Y1])
print("【1】列并行:W 按列切,各卡算部分列,AllGather 拼回")
print(f"  单卡结果 shape={full.shape}, 列并行拼接结果一致: {np.allclose(col,full)}")
print("  解读:列切每卡只算部分输出维度,最后 AllGather 拼回完整输出。")

# 2 行并行:W 按行切,各卡用部分输入算,AllReduce 求和
W0,W1=W[:2,:],W[2:,:]
X0,X1=X[:,:2],X[:,2:]
Z0,Z1=X0@W0,X1@W1
row=Z0+Z1
print(f"\n【2】行并行:W 按行切,各卡用部分输入算,AllReduce 求和")
print(f"  行并行求和结果一致: {np.allclose(row,full)}")
print("  解读:行切每卡用部分输入维度算部分结果,AllReduce 把各卡结果相加得完整输出。")

# 3 Megatron 组合 + 通信量
print(f"\n【3】Megatron 组合:列切(Linear1)→激活→行切(Linear2),中间免通信,末尾仅一次 AllReduce")
for dout in [4096,12288]:
    comm=dout*2*4   # AllReduce 输出(行切末尾),约 2×输出×字节
    print(f"  输出维度={dout}: 每层通信≈{comm/1e6:.2f} MB (仅一次 AllReduce)")
print("  解读:列切+行切组合让 FFN 内部不需通信,只在层边界 AllReduce 一次→通信量与隐藏维度相关,远小于全聚合。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:张量并行把单层权重切到多卡:列切 AllGather 拼输出,行切 AllReduce 求和;Megatron 组合让中间免通信。
- 熟手:TP 通信在层内高频,需高速互联(NVLink),一般 TP≤8(单节点);列切接激活免 AllReduce 是 Megatron 关键;
  TP 与 DP 正交,可叠加;TP 切分要兼顾负载与通信。
- 延伸:把切分数从2改到4看通信;对比纯 DP 与 TP 的通信频率。
EOF
echo "============================================================"
