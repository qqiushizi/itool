#!/bin/bash
# ============================================================
# 实验: c.hierarchical
# 说明: 分层 (Hierarchical) AllReduce:节点内 + 节点间
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Hierarchical AllReduce = 2 层: 节点内 (NVLink) + 节点间 (IB)。
# 节点内通信极快 (600 GB/s), 节点间慢 (25 GB/s)。
# 思路: 节点内先 allreduce, 再节点间 allreduce, 再节点内广播。
# 3 阶段 (假设 2 节点, 每节点 4 GPU):
#   1. 节点内: 4 GPU 内部 allreduce (NVLink 600 GB/s)
#   2. 节点间: 每节点 1 个代表参与 (IB 25 GB/s)
#   3. 节点内: 把节点间结果广播给所有 GPU
# 优势: 节点间通信量 = 节点内 1/N (N = 节点内 GPU 数)。
# 8 GPU/节点 (P=64, 8 节点):
#   - 纯 ring-allreduce 通信量 = 2N * 63/64
#   - 分层 = 2N * 7/8 (节点内) + 2(N/8) * 7/8 (节点间)
#         = N * 1.75 + N/8 * 1.75 = N * 1.97
# 几乎一样, 但分层 实测更快 (NVLink 比 IB 快 24x)。
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: c.hierarchical | 分层 AllReduce (节点内 + 节点间)"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 分层 3 阶段 ---
hdr(1,TOTAL,"分层 AllReduce 3 阶段")
why("""2 节点 x 4 GPU 示例, 8 GPU 总""")
out = ["  阶段         范围          通信量     带宽       参与"]
out.append("  1. 节点内    4 GPU 内部    N          600 GB/s   node 内 allreduce")
out.append("  2. 节点间    2 节点间      N/4        25 GB/s    1 GPU/node 通信")
out.append("  3. 节点内    4 GPU 内部    N/4        600 GB/s   广播到所有 GPU")
out.append("  总时间       1 + 2 + 3                = T1 + T2 + T3")
out.append("  关键: 节点内快, 节点间只发 1 份")
res("\n".join(out))
mea("""3 阶段流程:
  阶段 1: node 内 4 GPU 用 NVLink allreduce, 极快
  阶段 2: 每 node 1 个 GPU 代表走 IB allreduce
  阶段 3: 把代表结果广播回 node 内
  关键: 阶段 2 通信量 = N / G (G = node 内 GPU 数)
  G 越大, 节点间通信量越小""")

# --- 2. 通信量分析 ---
hdr(2,TOTAL,"通信量 vs Ring")
why("""P = G * N_node (G = GPU/节点, N_node = 节点数)
纯 Ring 通信量: 2N(P-1)/P
分层 通信量:
  阶段 1: 2N(G-1)/G (节点内)
  阶段 2: 2(N/G)(N_node-1)/N_node (节点间)
  阶段 3: (N/G)(N_node-1)/N_node * G = 2N(N_node-1)/N_node (广播回)
近似: 2N(1 - 1/G) + 2N(1-1/N_node)/G ~ 2N(1 - 1/P)""")
out = ["  配置              Ring        分层         哪个"]
out.append("  P=8, G=4, N=2    1.75 N     1.5 + 0.25 = 1.75 N  接近")
out.append("  P=64, G=8, N=8   1.97 N     1.75 + 0.22 = 1.97 N  接近")
out.append("  P=128, G=8, N=16 1.98 N     1.75 + 0.23 = 1.98 N  接近")
out.append("  通信量上两者相当")
out.append("  但 分层节点内走 NVLink, 实测快 2-3x")
res("\n".join(out))
mea("""通信量结论:
  1. 通信量上, 分层与 Ring 几乎相同
  2. 但分层节点内 600 GB/s, Ring 节点间也 25 GB/s
  3. 节点内通信 = 总通信 87.5% (G=8)
  4. 实测: 分层比纯 Ring 快 1.5-2x
  5. 节点数越多, 优势越明显""")

# --- 3. 实战:Ascend 集群 ---
hdr(3,TOTAL,"Ascend 集群:8 卡 mesh + IB")
why("""Ascend 910B 集群 (8 卡/节点, 16 节点, 128 卡) 实战:""")
out = ["  参数                 值          备注"]
out.append("  节点数                16          物理机")
out.append("  每节点卡数            8           910B")
out.append("  总卡数                128         训练规模")
out.append("  节点内 (HCCS)         200 GB/s    全互连 mesh")
out.append("  节点间 (IB HDR)       25 GB/s     200 Gb/s)")
out.append("  纯 Ring 节点内通信    25 GB/s     走 IB, 慢")
out.append("  分层 节点内通信        200 GB/s   走 HCCS, 快 8x")
out.append("  节点间通信量减少      1/8         G=8")
out.append("  加速比 (实测)         1.5-2x     通信耗时")
res("\n".join(out))
mea("""Ascend 集群设计:
  1. 8 卡 mesh (HCCS) 全互连, 比 PCIe 快 6x
  2. 节点间 IB HDR 200G, RDMA 加速
  3. 分层 allreduce 是 Ascend 大模型训练标配
  4. 通信量减少 1/8, 节点内走 HCCS 8x 快
  5. 实测加速比 1.5-2x
配置: HCCL 配置 rank table, 自动选分层算法""")

# --- 4. NCCL/HCCL 实现 ---
hdr(4,TOTAL,"NCCL/HCCL 启用分层")
why("""框架启用分层:""")
out = ["  框架         启用方式                       备注"]
out.append("  NCCL         自动 (节点内 NVLink 优先)      默认开启")
out.append("  NCCL         NCCL_ALGO=Ring,Tree             强制算法")
out.append("  NCCL         NCCL_IB_HCA=mlx5                指定 IB 网卡")
out.append("  NCCL         NCCL_SOCKET_IFNAME=eth0         备选 TCP")
out.append("  HCCL         配 rank table, 8 节点 8 卡     必填")
out.append("  HCCL         HCCL_ALGO=auto                  默认分层")
out.append("  PyTorch      配 MASTER_ADDR + WORLD_SIZE    标准启动")
out.append("  Megatron     tp_comm_overlap + DP overlap   手动分层")
res("\n".join(out))
mea("""实战配置 (NCCL 例子):
  # 8 节点, 每节点 8 GPU
  export NCCL_DEBUG=INFO
  export NCCL_IB_HCA=mlx5_0,mlx5_1
  export NCCL_SOCKET_IFNAME=eth0
  export NCCL_ALGO=Tree,Ring  # 多算法
  # 启动
  torchrun --nproc_per_node=8 \\
           --nnodes=8 \\
           --node_rank=0 \\
           --master_addr=node0 \\
           train.py

  # Ascend HCCL 同理, 但用 rank table
  # HCCL 内部自动分层 (节点内 HCCS + 节点间 IB)
  # 一般无需手动配

  # 验证分层是否启用
  # torch.distributed.all_reduce 时打印 INFO 日志
  # 看到 'NET' (节点间) 和 'NVL/HCCS' (节点内) 字样
  # 表示分层在工作""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:分层 AllReduce = 节点内 (NVLink 600 GB/s) + 节点间 (IB 25 GB/s) 2 层;
  3 阶段: 节点内 allreduce -> 节点间 allreduce -> 节点内 broadcast;
  节点间通信量减少 G 倍 (G = 节点内 GPU), 实测加速比 1.5-2x;
  NCCL/HCCL 默认启用, 大模型训练标配。
- 熟手:通信量上分层与 Ring 几乎相同 (2N(1-1/P)), 但节点内走高速;
  G 越大优势越明显 (8 卡节点, 节点内通信占比 87.5%);
  Ascend 8 卡 HCCS mesh (200 GB/s) + IB HDR (25 GB/s), 分层效果明显;
  配置: NCCL 自动, HCCL 配 rank table; 日志看 NET vs NVL 验证。
【进阶】在 8 节点 8 GPU 集群上跑 100GB allreduce, 对比:
  1. 纯 Ring (强制); 2. 分层 (默认); 3. 节点内 ring + 节点间 tree (混合);
  测不同 G (2/4/8), 画 加速比 vs G 曲线, 找最优 G 配置。
EOF
echo "############################################################"
