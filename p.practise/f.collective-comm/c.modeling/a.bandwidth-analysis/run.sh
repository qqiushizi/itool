#!/bin/bash
# ============================================================
# 实验: a.bandwidth-analysis
# 说明: 集合通信带宽分析 (节点内/节点间/IB/PCIe)
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 带宽是分布式训练的核心瓶颈, 训练速度 ~ 1/带宽。
# 4 层带宽:
#   - HBM (显存带宽): 3 TB/s (H100), 1 TB/s (A100)
#   - 节点内 (NVLink): 600-900 GB/s
#   - 节点间 (IB): 25-50 GB/s
#   - 跨集群 (WAN): 1-10 GB/s
# 训练 1 步通信量:
#   - DP allreduce: 2N * (P-1)/P bytes (N = 梯度)
#   - TP allreduce: 2N * (T-1)/T bytes (T = TP size)
#   - PP send/recv: 2N * (P-1) bytes (P = PP size)
# 决定扩展效率 = 算 / (算 + 通), 通占比大就慢
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: a.bandwidth-analysis | 集合通信带宽分析"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 硬件带宽分级 ---
hdr(1,TOTAL,"硬件带宽分级")
why("""从最快到最慢, 通信的 4 个层级:""")
out = ["  层级          带宽           延迟      典型设备"]
out.append("  HBM (显存)    1000-3000 GB/s  ~1us     H100 3 TB/s")
out.append("  NVLink        600-900 GB/s   ~1us     H100 NVLink 4.0")
out.append("  HCCS (Ascend) 200-400 GB/s  ~2us     910B HCCS")
out.append("  PCIe Gen4/5   32-64 GB/s    ~3-5us   备选")
out.append("  IB HDR/NDR    25-50 GB/s    ~2us     集群标配")
out.append("  100 GbE       12.5 GB/s     ~10us    备选")
out.append("  25 GbE        3.1 GB/s      ~20us    慢")
out.append("  WAN           0.1-1 GB/s    ~ms      跨集群")
res("\n".join(out))
mea("""带宽对比洞察:
  1. HBM 是 节点内通信的上限 (3 TB/s)
  2. NVLink 是 节点内 GPU 通信的标配 (900 GB/s)
  3. 节点间 IB 25-50 GB/s, 比 NVLink 慢 20-40x
  4. PCIe 32 GB/s, 慢 30x
  5. WAN 1 GB/s, 慢 1000x
分布式训练通信几乎必走 IB, 慢 LAN/WAN 训练会卡死""")

# --- 2. 通信量分析 ---
hdr(2,TOTAL,"训练 1 步的通信量")
why("""以 7B 模型 (FP16, 7B * 2 = 14 GB 梯度) 为例:""")
out = ["  并行方式    每 rank 数据     通信量        通信耗时 (IB 25 GB/s)"]
grad = 7e9 * 2  # 14 GB 梯度, FP16
for P in [8, 64, 128, 512, 1024]:
    grad_per_rank = grad / P
    comm_bytes = 2 * grad * (P-1) / P  # DP allreduce
    time_s = comm_bytes / 25e9
    out.append(f"  DP, P={P:4d}   {grad_per_rank/1e9:.2f} GB      {comm_bytes/1e9:.2f} GB        {time_s*1000:.1f} ms")
out.append(f"  TP, T=8     1/8*grad     ~0.3*grad        0.3*grad*2/T_8/25G=...")
out.append(f"  PP, P=8     1/8*grad     2*grad*(P-1)   跨 step, 看序列长")
res("\n".join(out))
mea("""训练通信瓶颈:
  1. 7B 模型, P=8: 13 GB 通信 / step
  2. P=64: 13.7 GB / step
  3. P=1024: 14 GB / step (接近常数)
  4. 25 GB/s IB: 0.5s / step
  5. 算力 (H100 1 PFLOPS) 算 7B 1 步: 14 TFLOPS / 1 PFLOPS = 14 ms
  6. 通信 0.5s, 算力 14 ms, 通信是瓶颈 35x
解决:
  1. 增加算力 (用 NVLink, 900 GB/s)
  2. 减少通信 (量化, 压缩)
  3. overlap 通信和计算""")

# --- 3. 通信瓶颈分类 ---
hdr(3,TOTAL,"通信瓶颈分类:算力 vs 带宽 vs 延迟")
why("""通信瓶颈有 3 类, 不同原因:""")
out = ["  瓶颈类型     表现              解决              示例"]
out.append("  带宽瓶颈     大消息, 实测 BW 接近上限  算力换通信     7B DP")
out.append("  延迟瓶颈     小消息, BW 远低上限      批处理, 拓扑   频繁 allreduce")
out.append("  算力瓶颈     CPU 处理慢              异步, 优化代码  python 主控")
out.append("  调度瓶颈     kernel 间隔大            通信 overlap   ring + 计算")
out.append("  拓扑瓶颈     路径不佳, 慢链路        rail-opt      异号卡走 PCIe")
out.append("  队列瓶颈     NCCL 队列满            调 buffer pool  1000 卡")
res("\n".join(out))
mea("""瓶颈定位流程:
  1. 实测 BW: torch.distributed all_reduce 1 GB
  2. 对比理论 BW: 算 (25 GB/s IB, 900 GB/s NVLink)
  3. 实测 < 理论 50%: 必有瓶颈
  4. 分类:
     - 带宽瓶颈: 消息大, BW 接近上限, 正常
     - 延迟瓶颈: 消息小, BW 远低上限
     - 算力瓶颈: CPU 时间 > GPU 时间
     - 调度瓶颈: kernel 间隔 > 算子时间""")

# --- 4. 优化策略 ---
hdr(4,TOTAL,"优化策略:省通信时间")
why("""7 个核心优化:""")
out = ["  策略              通信减少     难度   副作用"]
out.append("  1. overlap 通信+算  藏起来      中     内存多")
out.append("  2. 梯度压缩 (FP8)  50%         中     精度掉")
out.append("  3. 梯度稀疏化      90%         高     收敛慢")
out.append("  4. 拓扑优化        路径选优     低     -")
out.append("  5. ZeRO 切分       1/N         中     内存多")
out.append("  6. 通信算法选       ring/tree   低     -")
out.append("  7. 减少同步次数     多次合 1    高     收敛影响")
res("\n".join(out))
mea("""优化优先级:
  1. 拓扑优化: 必做, 0 成本
  2. overlap 通信+算: 必做, 训练标配
  3. ZeRO 切分: 内存紧时做
  4. 通信压缩: 带宽紧时做
  5. 稀疏化: 收敛紧时做
实战: Megatron-LM 7 项全开, 通信从 50% -> 10%
最简起步: torch.distributed + DDP + AMP + overlap
Ascend 同: amp + cann 融合 + 拓扑感知""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:分布式训练通信 4 层: HBM > NVLink/HCCS > IB > WAN;
  节点内 600-900 GB/s, 节点间 25-50 GB/s, 差 20-40x;
  7B 模型 1 步通信 ~14 GB, 占步时间 50-80% (IB);
  优化必做: 拓扑优化 + overlap 通信计算, 进阶: ZeRO + 压缩。
- 熟手:7B FP16 1 步通信 14 GB, P=1024 时接近常数;
  瓶颈分类: 带宽 / 延迟 / 算力 / 调度 / 拓扑 / 队列;
  实测 BW < 理论 50% 必有瓶颈, 找原因;
  优化 7 策略, 优先级: 拓扑 < overlap < ZeRO < 压缩 < 稀疏化;
  Megatron 7 项全开, 通信占比 50% -> 10%。
【进阶】在 1 个 8 GPU 节点上实测 all_reduce 不同大小 (1KB-1GB) 的 BW,
  拟合 T = alpha + N*beta 模型; 对比 节点内 (NVLink) vs 模拟节点间 (PCIe);
  画 BW vs 消息大小 曲线, 找 拐点 (延迟瓶颈 vs 带宽瓶颈)。
EOF
echo "############################################################"
