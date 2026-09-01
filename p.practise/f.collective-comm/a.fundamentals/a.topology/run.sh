#!/bin/bash
# ============================================================
# 实验: a.topology
# 说明: 集合通信拓扑结构 (NVLink/IB/RoCE/Rail-optimized)
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 分布式训练的核心是 通信, 通信效率 = 带宽 / 跳数。
# 拓扑结构决定 节点内 (intra-node) 和 节点间 (inter-node) 带宽。
# GPU 节点内:
#   - NVLink/NVSwitch: 600-900 GB/s (per GPU), 8 卡全互连
#   - PCIe: 32 GB/s (gen5), 受限
# Ascend 节点内:
#   - HCCS (华为 Cache Coherent System): 200 GB/s per chip
#   - 全互连, 8 卡 mesh
# 节点间:
#   - InfiniBand (IB): 200-400 Gb/s per port, RDMA
#   - RoCE: IB over Ethernet, 100-200 Gb/s
#   - 普通 TCP/IP: 10-25 Gb/s, 极慢
# Rail-optimized 拓扑:
#   - 同号卡跨节点 (rank 0 节点 0 -> rank 0 节点 1) 用高速链路
#   - 异号卡跨节点用慢速链路
#   - 设计目标: 同号卡通信走 RDMA, 异号卡通信走 PXN
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: a.topology | 集合通信拓扑 (NVLink/HCCS/IB/RoCE)"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 节点内拓扑 ---
hdr(1,TOTAL,"节点内拓扑:NVLink / HCCS / PCIe")
why("""单机 8 卡 / 16 卡的内部互连。""")
out = ["  类型              带宽 (per GPU)   延迟    卡数   备注"]
out.append("  NVLink 3.0         600 GB/s        ~1us    8      A100 标配")
out.append("  NVLink 4.0         900 GB/s        ~1us    8      H100")
out.append("  NVSwitch           900 GB/s 全互连  ~1us    8-16   A100/H100")
out.append("  HCCS (Ascend)      200 GB/s        ~2us    8      910 标配")
out.append("  PCIe Gen4          32 GB/s         ~5us    8      备选")
out.append("  PCIe Gen5          64 GB/s         ~3us    8      H100/B100")
out.append("  QPI/UPI (CPU)      50-100 GB/s     ~10us   2      跨 socket")
res("\n".join(out))
mea("""节点内通信关键:
  1. NVLink/HCCS 是大模型训练标配, 比 PCIe 快 20x
  2. 8 卡 NVLink 互连 -> allreduce 接近线性加速
  3. CPU 通信 (QPI) 慢, 跨 socket 要避免
  4. 拓扑查询: nvidia-smi topo -m / npu-smi info -t topo""")

# --- 2. 节点间拓扑 ---
hdr(2,TOTAL,"节点间拓扑:IB / RoCE / TCP")
why("""跨节点通信, 决定多机扩展效率。""")
out = ["  类型            带宽 (per port)   延迟    协议    适用"]
out.append("  IB HDR          200 Gb/s          ~2us    RDMA   高端训练")
out.append("  IB NDR          400 Gb/s          ~1.5us  RDMA   H100/集群")
out.append("  IB XDR          800 Gb/s          ~1us    RDMA   未来")
out.append("  RoCE v2         100-200 Gb/s      ~5us    RDMA   替代 IB")
out.append("  100 GbE         100 Gb/s          ~10us   TCP/IP 普通")
out.append("  25 GbE          25 Gb/s           ~20us   TCP/IP 慢")
out.append("  10 GbE          10 Gb/s           ~50us   TCP/IP 太慢")
res("\n".join(out))
mea("""节点间通信选择:
  1. 大模型训练: 必 IB NDR (400G) 或 HDR (200G)
  2. 中等规模: RoCE v2 (200G) 性价比
  3. 小规模/测试: 100 GbE 勉强
  4. 10 GbE: 仅测试用, 训练会卡
  5. 阿里云/华为云: 都有 RDMA 主机型, 选 high-perf 网卡""")

# --- 3. Rail-optimized 拓扑 ---
hdr(3,TOTAL,"Rail-optimized 拓扑")
why("""大模型训练专用拓扑:
  - 同号卡 (rank 0 -> rank 0 跨节点) 走高速
  - 异号卡 (rank 0 -> rank 1 跨节点) 走次速
  - 目的: allreduce / allgather 用同号卡, P2P 用异号卡
Ascend 8 卡 mesh + 节点间 RDMA""")
out = ["  节点            节点内               节点间 (rank 0 <-> rank 0)"]
out.append("  节点 0          [0 1 2 3 4 5 6 7]    <-IB-> 节点 1 rank 0")
out.append("  节点 1          [0 1 2 3 4 5 6 7]    <-IB-> 节点 2 rank 0")
out.append("  节点 2          [0 1 2 3 4 5 6 7]    <-IB-> 节点 3 rank 0")
out.append("  ...             ...                  ...")
out.append("  节点 N          [0 1 2 3 4 5 6 7]    <-IB-> 节点 N+1 rank 0")
out.append("  全集群: 同号卡构成 1D ring, 异号卡走 PXN (近 0 跳)  ")
res("\n".join(out))
mea("""Rail-optimized 设计原因:
  1. allreduce 是同号卡通信 (假设数据并行)
  2. 异号卡 (TP 内) 通信概率小 (PP 通信除外)
  3. 让同号卡走 RDMA, 异号卡走 mesh, 整体最优化
  4. Megatron-LM 论文中量化: 通信量 -30%
工具: NCCL_TOPO_FILE / HCCL_TOPO_FILE 指定""")

# --- 4. 拓扑检测 ---
hdr(4,TOTAL,"拓扑检测与配置")
why("""实际部署时检测拓扑, 优化通信:""")
res("""# 1. GPU 节点内拓扑
nvidia-smi topo -m
# 输出 (示例):
#     GPU0  GPU1  GPU2  ...  CPU Affinity
# GPU0  X    NV2   NV2   ...  0-15
# GPU1  NV2  X     NV1   ...  16-31
# GPU2  NV2  NV1   X     ...  0-15
# 解读: NV2 = 2 个 NVLink 链路, 越宽越好

# 2. Ascend 节点内拓扑
npu-smi info -t topo -i 0
# 或
cat /usr/local/Ascend/driver/version.info
hccs_tool -i 0 -show

# 3. 节点间 RDMA 检测
ibstat
# 或
ibdev2netdev
# 输出: mlx5_0 port 1 ==> net eth0 (Up)

# 4. NCCL/HCCL 拓扑感知
export NCCL_TOPO_FILE=/path/to/topo.xml
export NCCL_DEBUG=INFO
# 训练时会打印每个 rank 的路径

# 5. 测实际带宽
python3 -c "
import torch
import torch.distributed as dist
dist.init_process_group('nccl')
# allreduce 1 GB
x = torch.ones(256*1024*1024, dtype=torch.float32, device='cuda')
torch.cuda.synchronize()
import time
t = time.time()
for _ in range(10):
    dist.all_reduce(x)
torch.cuda.synchronize()
print(f'allreduce 1GB: {(time.time()-t)*100:.1f} ms, BW = {1/(time.time()-t)/10*8:.1f} GB/s')
""")
mea("""拓扑优化的工程步骤:
  1. 部署前画拓扑图 (NVLink mesh, IB 路径)
  2. 配置 NCCL/HCCL env (IB 网卡, RDMA)
  3. 跑实测带宽 (allreduce / allgather)
  4. 对比理论带宽, 找瓶颈 (PCIe 限? 路由错?)
  5. 调: 选网卡, 调队列, 调 buffer pool
  6. 复杂集群: 用 rail-optimized + 拓扑感知算法""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:分布式训练通信分 节点内 (NVLink/HCCS 600+ GB/s) + 节点间 (IB 200-400 Gb/s);
  拓扑决定带宽, 选错拓扑 (10 GbE) 训练慢 10x;
  Rail-optimized 让同号卡走高速, 异号卡走 mesh;
  部署必测实际带宽, 排查瓶颈 (PCIe 限? 路由错?)。
- 熟手:NVLink 3.0/4.0 是大模型标配, 比 PCIe 快 20x; IB NDR (400G) 是 H100 时代标配;
  Rail-optimized 是大模型训练专用, allreduce 通信量 -30%;
  拓扑检测: nvidia-smi topo -m + ibstat + NCCL_DEBUG=INFO;
  实际带宽 < 理论 50% 必有瓶颈, 查 PCIe 限速 / 路由 / 队列。
【进阶】用 nvidia-smi topo -m 画 1 个 8 卡节点的拓扑图, 计算:
  1. allreduce 1GB 的理论耗时; 2. 实测耗时; 3. 加速比; 4. 瓶颈在哪。
EOF
echo "############################################################"
