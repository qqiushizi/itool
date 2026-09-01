#!/bin/bash
# ============================================================
# 实验: b.performance
# 说明: 集合通信性能分析与瓶颈定位
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 集合通信性能分析 = 找瓶颈在哪。
# 3 个维度:
#   1. 算法: ring vs tree 选错
#   2. 拓扑: NVLink vs IB 路径
#   3. 同步: 阻塞 vs 异步, 是否 overlap
# 性能指标:
#   - 带宽 (BW): 实际 / 理论
#   - 延迟 (latency): P-1 * alpha
#   - 利用率 (util): 实际 BW / 峰值
#   - 步数 (steps): 通信算法
# 工具:
#   - NCCL benchmark (nccl-tests)
#   - torch.profiler (timeline)
#   - msprof / npu profiler
#   - 自己写 benchmark
# 优化路径:
#   1. 测基线 (无优化)
#   2. 开 NCCL 调试看算法
#   3. 配拓扑 (IB 网卡, 队列)
#   4. 开 overlap
#   5. 调 bucket size
#   6. 极致: 换硬件 (IB NDR 400G)
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: b.performance | 集合通信性能分析与瓶颈定位"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 性能指标体系 ---
hdr(1,TOTAL,"性能指标体系")
why("""衡量集合通信的 6 个指标:""")
out = ["  指标          公式                含义              典型值"]
out.append("  BW (带宽)     N / T              实际带宽            25 GB/s (IB HDR)")
out.append("  BW% (利用率)  BW / 理论          利用率              70-90% 健康")
out.append("  Latency       T - N/BW            延迟               1-5us")
out.append("  Steps         算法步数            通信算法            ring 2(P-1)")
out.append("  Vol (量)      N * (P-1)/P        通信数据量         2N (DP)")
out.append("  Time          端到端             总耗时             < 算力是目标")
res("\n".join(out))
mea("""指标健康值:
  1. BW% > 70% 健康
  2. BW% < 50% 必有瓶颈
  3. Latency 应 < 5us (节点内), 10us (节点间)
  4. Steps = 2(P-1) (ring) 或 log P (tree)
  5. 总耗时 应 < 算力耗时 (否则通信主导)
诊断:
  - BW% 低, Latency 正常: 带宽瓶颈
  - Latency 高: 延迟瓶颈 (消息小, 步数多)
  - Steps 不合理: 算法选错
  - Time > 算力: 通信主导, 必优化""")

# --- 2. NCCL 性能调优 ---
hdr(2,TOTAL,"NCCL 性能调优配置")
why("""NCCL 提供 20+ 环境变量调优:""")
out = ["  变量                    作用                        推荐"]
out.append("  NCCL_IB_HCA             指定 IB 网卡                 mlx5_0,mlx5_1")
out.append("  NCCL_IB_DISABLE         关 IB                       节点内测试")
out.append("  NCCL_SOCKET_IFNAME      TCP 网卡                    eth0")
out.append("  NCCL_P2P_LEVEL          NVLink P2P 启用             NVL (节点内)")
out.append("  NCCL_BUFFSIZE           通信 buffer                 4MB (大消息)")
out.append("  NCCL_NTHREADS           NCCL 线程                   4-8")
out.append("  NCCL_MIN_NCHANNELS      最小通道数                  2-4")
out.append("  NCCL_MAX_NCHANNELS      最大通道数                  8-16 (大消息)")
out.append("  NCCL_NSOCKS_PERTHREAD   每线程 socket               4")
out.append("  NCCL_PROTO              协议 (LL/Simple)            Simple (大消息)")
out.append("  NCCL_ALGO               算法 (Ring/Tree)            auto")
out.append("  NCCL_DEBUG              日志级别                    INFO (调试)")
res("\n".join(out))
mea("""NCCL 调优实战:
  # 1. 大消息优化
  export NCCL_BUFFSIZE=8388608  # 8MB
  export NCCL_MAX_NCHANNELS=16
  export NCCL_NSOCKS_PERTHREAD=4
  # 2. 小消息优化
  export NCCL_PROTO=LL  # low latency
  export NCCL_P2P_LEVEL=NVL
  # 3. 节点内 (禁用 IB)
  export NCCL_IB_DISABLE=1
  # 4. 节点间 (启用 RDMA)
  export NCCL_IB_HCA=mlx5_0,mlx5_1
  # 5. 调试
  export NCCL_DEBUG=INFO
  # 6. 超时
  export NCCL_TIMEOUT=300
  # Ascend 同, 用 HCCL_ 前缀""")

# --- 3. 瓶颈分类与定位 ---
hdr(3,TOTAL,"瓶颈分类与定位")
why("""5 类瓶颈, 定位方法:""")
out = ["  瓶颈类型     表现                 定位工具             解决"]
out.append("  带宽瓶颈     大消息, BW 接近上限  nccl-tests, profiler  换高速网卡, 压缩")
out.append("  延迟瓶颈     小消息, BW 远低上限  nccl-tests          换 LL 协议, 拓扑优化")
out.append("  算法瓶颈     步数不优            NCCL_DEBUG=INFO     换算法 (ring/tree)")
out.append("  拓扑瓶颈     走慢链路             nvidia-smi topo -m  rail-opt, PXN")
out.append("  调度瓶颈     kernel 间隔大        msprof timeline    overlap")
out.append("  算力瓶颈     python 主控慢         py-spy             异步, C++ 主控")
out.append("  队列瓶颈     NCCL queue 满       NCCL_DEBUG=INFO     减并发或加 buffer")
out.append("  序列化瓶颈   python tensor -> C  torch profiler      用 pinned memory")
res("\n".join(out))
mea("""定位流程:
  1. 跑 nccl-tests, 看 BW 是否健康
     - allreduce 1GB: 理论 25 GB/s, 测 22 GB/s 健康
  2. 看 BW% < 50%, 必有瓶颈
  3. 分类:
     a) 消息小, 延迟高: 延迟瓶颈 -> 调协议 / 算法
     b) 消息大, BW 低: 带宽瓶颈 -> 换网卡 / 路径
     c) 步数多: 算法瓶颈 -> 选 tree / 分层
     d) 间隔大: 调度瓶颈 -> overlap
  4. 看 msprof timeline, 找 kernel 间隔
  5. 极致优化: 换硬件 (IB NDR 400G)
NCCl-tests 用法:
  # 下载
  git clone https://github.com/NVIDIA/nccl-tests
  cd nccl-tests
  make MPI=1 CUDA_HOME=/usr/local/cuda NCCL_HOME=...
  # 跑
  ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8
  # 输出: 算法, 大小, 耗时, BW, algBW""")

# --- 4. 优化路径与量化 ---
hdr(4,TOTAL,"优化路径与量化收益")
why("""7B 模型 + 8 GPU DP 训练, 优化路径:""")
out = ["  优化点            通信耗时    节省     累计   难度"]
out.append("  baseline          800 ms      -       800    -")
out.append("  + 拓扑优化        650 ms     150      650    低")
out.append("  + DDP overlap     500 ms     150      500    低")
out.append("  + AMP (FP16)      350 ms     150      350    中")
out.append("  + bucket=50MB     300 ms      50      300    低")
out.append("  + 手动多 stream   250 ms      50      250    中")
out.append("  + FSDP            200 ms      50      200    中")
out.append("  + 梯度压缩        150 ms      50      150    高")
out.append("  + 换 IB NDR       100 ms      50      100    硬件")
res("\n".join(out))
mea("""优化路径选择 (按收益/难度比):
  1. 拓扑优化: 必做, 0 成本
  2. DDP overlap: 必做, 1 行
  3. AMP: 必做, 1 行
  4. bucket size: 调参, 0 成本
  5. FSDP: 内存紧时做
  6. 手动 overlap: 收益小, 复杂
  7. 压缩: 收敛紧时做
  8. 换硬件: ROI 低, 慢
实战优先级:
  1+2+3 = 起步, 通信从 800 -> 350 ms (56% 节省)
  4 = 调参, 0 成本
  5+ = 进阶
  6+7+8 = 极致, 投入大
Ascend 同: 拓扑 + HCCL overlap + 量化 + 算子融合""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:集合通信性能看 4 指标: BW% / Latency / Steps / Time;
  BW% < 50% 必有瓶颈; 跑 nccl-tests 看基线, 配 NCCL env 调优;
  优化路径: 拓扑 + overlap + AMP + bucket size, 起步通信 -50%;
  极致优化: FSDP + 压缩 + 换硬件, 投入大, 收益递减。
- 熟手:5 类瓶颈: 带宽 / 延迟 / 算法 / 拓扑 / 调度;
  NCCL 调优 12 个 env: IB_HCA, BUFFSIZE, MAX_NCHANNELS, ALGO, PROTO;
  定位流程: nccl-tests 测基线 -> msprof 看 timeline -> 分类瓶颈 -> 调优;
  优化收益/难度比: 拓扑 (低) > overlap (低) > AMP (中) > FSDP (中) > 压缩 (高) > 硬件 (硬件);
  Ascend 同, HCCL 前缀, 拓扑 + overlap + 量化 + 算子融合。
【进阶】在 1 个 8 GPU 节点上跑 7B 模型 DP 训练, 完整走优化流程:
  1. 跑 nccl-tests, 测基线 BW;
  2. 配 NCCL env 调优, 测新 BW;
  3. 启 DDP, 测训练 1 步耗时;
  4. 加 AMP, bucket 调, overlap, 测每步耗时;
  5. 对比 FSDP, 找 最优配置 (吞吐 / 内存);
  6. 画 优化路径 vs 通信耗时 柱状图, 量化每步收益。
EOF
echo "############################################################"
