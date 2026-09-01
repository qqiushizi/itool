#!/bin/bash
# ============================================================
# 实验: b.tree-allreduce
# 说明: Tree-AllReduce 算法:log P 步, 适合小消息
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Tree-AllReduce = 用二叉树 (k-ary) 组织 allreduce。
# 步数: log_k(P), 比 ring 少 (2(P-1))。
# 通信量: 大, 树内部带宽瓶颈 (根节点)。
# 2 阶段:
#   1. reduce: 数据从叶向根累加
#   2. broadcast: 结果从根向叶广播
# 二叉树示例 (P=8):
#   - 步数 = log2(8) = 3
#   - vs ring = 2*7 = 14
#   - 加速比 4.7x (延迟)
# 关键问题:
#   - 根节点带宽瓶颈 (双向流量)
#   - 适合小消息 + 大集群 (P >= 16)
# 变体:
#   - 二叉树: 步数 log2(P)
#   - k 叉树: 步数 log_k(P)
#   - 双二叉树: 同时向上 + 向下, 链路复用
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: b.tree-allreduce | Tree-AllReduce 算法"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. Tree 拓扑结构 ---
hdr(1,TOTAL,"Tree 拓扑:二叉树示例")
why("""P=8 节点, 二叉树组织:""")
out = ["  层    节点          子节点"]
out.append("  0     0 (root)     1, 2")
out.append("  1     1             3, 4")
out.append("  1     2             5, 6")
out.append("  2     3             7")
out.append("  2     4             -")
out.append("  2     5             -")
out.append("  2     6             -")
out.append("  2     7             -")
out.append("  总节点 8, 深度 3 (log2 8)")
res("\n".join(out))
mea("""二叉树特点:
  1. 步数 = log2(P) (vs ring 2(P-1))
  2. P=8: 3 步 (vs ring 14)
  3. P=64: 6 步 (vs ring 126)
  4. P=128: 7 步 (vs ring 254)
  5. 延迟敏感场景必选
  root 瓶颈: 流量 2x 子节点, 需高带宽""")

# --- 2. Reduce + Broadcast 2 阶段 ---
hdr(2,TOTAL,"Reduce + Broadcast 2 阶段")
why("""Tree-AllReduce = Reduce (叶到根) + Broadcast (根到叶)""")
out = ["  阶段        方向     流量 (root 视角)    步数   流量 (链路)"]
out.append("  Reduce       叶 -> 根   N (入) + N (出)   log P   N/2  (每子节点)")
out.append("  Broadcast    根 -> 叶   N (出) + N (入)   log P   N/2")
out.append("  全过程       双向       2N                 2log P  N")
out.append("  vs Ring      单向       N (但 2(P-1) 步)  2(P-1)  N/P")
res("\n".join(out))
mea("""2 阶段分析:
  1. Reduce: 数据从叶上浮, 每步是 2 节点间求和
  2. Broadcast: 结果从根下沉, 复制给每子节点
  3. root 在中间, 流量 = 2N (双向)
  4. 链路流量 = N/2 (每条边)
  5. 总通信量 = N*log2(P) (vs ring 2N(P-1)/P)
小消息 (1KB): tree 延迟 log P, 占优
大消息 (1GB): tree 通信量 N*log P, ring 通信量 2N, ring 优""")

# --- 3. 通信量 vs Ring ---
hdr(3,TOTAL,"通信量对比:Tree vs Ring")
why("""不同 P 下, 两种算法的通信量:""")
import math
out = ["  P      Tree (N*log2 P)   Ring (2N(P-1)/P)   哪个优"]
for P in [2, 4, 8, 16, 32, 64, 128, 256]:
    tree = math.log2(P)
    ring = 2 * (P - 1) / P
    winner = "Tree" if tree < ring else "Ring"
    out.append(f"  {P:3d}    {tree:.2f}*N            {ring:.2f}*N              {winner}")
res("\n".join(out))
mea("""通信量交叉点:
  1. P <= 4: Tree 略优 (2 vs 1.5)
  2. P = 8: Tree 3*N vs Ring 1.75*N, Ring 优
  3. P = 64: Tree 6*N vs Ring 2*N, Ring 大幅优
  4. P = 256: Tree 8*N vs Ring 2*N, Ring 优
所以 通信量 上 Ring 几乎总优。
但 Tree 在 延迟 上:
  - 步数: log P vs 2(P-1)
  - 小消息 (延迟敏感): Tree 优
  - 大消息 (带宽敏感): Ring 优
实战: NCCL 自动选, 小消息 tree, 大消息 ring""")

# --- 4. 变体与实战 ---
hdr(4,TOTAL,"变体:k 叉树、双二叉树")
why("""Tree 算法的变体:""")
out = ["  变体                步数              流量     适用"]
out.append("  二叉树 (k=2)         log2(P)          2N       P 中等")
out.append("  k 叉树 (k=4)         log4(P)          4N       P 大, 链路多")
out.append("  双二叉树 (double)    2*log2(P)        2N       链路复用")
out.append("  binomial tree       log2(P)          ~2N      故障容错")
out.append("  k-ary + pipeline    log_k(P/N_chunks) N      大消息")
res("\n".join(out))
mea("""实战选择:
  1. P <= 16: ring 都行, 看延迟
  2. P = 16-64: tree 适合小消息, ring 适合大消息
  3. P >= 64: ring 是默认 (带宽优)
  4. NCCL/HCCL 内部: 2 算法, 自动选
  5. 用户层: 一般不用选, torch.distributed 帮你
NCCL 决策 (NCCL_ALGO):
  - tree: 步数 = log2(P)
  - ring: 步数 = 2(P-1)
  - 选择标准: 数据大小 vs 临界点
PyTorch 测试:
  torch.distributed.all_reduce  # 自动选""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Tree-AllReduce = Reduce (叶->根) + Broadcast (根->叶), 2 阶段;
  步数 log2(P) 远少于 ring 的 2(P-1), 适合小消息 (延迟敏感);
  通信量 N*log P 比 ring 2N(P-1)/P 略大, 大消息不划算;
  root 节点是带宽瓶颈, 需高带宽链路。
- 熟手:二叉树 P=64 时 6 步 (vs ring 126), 延迟降低 21x;
  通信量交叉点: P>=8 ring 通信量优, 但小消息延迟 tree 优;
  变体: k 叉树 (P 大)、双二叉树 (链路复用)、binomial tree (容错);
  NCCL/HCCL 自动选算法, 用户一般无需手动指定。
【进阶】在 1 个 P=8 集群上手写 Tree-AllReduce (用 8 进程),
  对比 NCCL ring 的耗时; 测不同消息大小 (1KB, 100KB, 10MB),
  画 耗时 vs 消息大小 曲线, 找 tree 和 ring 的交叉点。
EOF
echo "############################################################"
