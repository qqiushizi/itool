#!/bin/bash
# ============================================================
# 实验: a.ring-allreduce
# 说明: Ring-AllReduce 算法:2 步完成, 通信量 O(N)
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Ring-AllReduce 是最经典的 allreduce 算法。
# 2 步:
#   1. reduce-scatter: N 个数据切成 P 份, 沿 ring 累加
#   2. allgather: 把每份结果沿 ring 广播
# 通信量分析:
#   - 每步 (P-1) 跳, 每跳 N/P 数据
#   - 总通信量 = 2 * (P-1) * N/P ~ 2N (P 大)
#   - 独立于 P! 这就是它扩展性好的原因
# vs naive allreduce (P-1 次 P2P):
#   - 通信量 = N * (P-1) (随 P 线性增)
#   - 慢 P 倍
# 限制:
#   - 步数 2(P-1), 小消息延迟大
#   - 适合大消息 (>= 1MB)
# 实现: NCCL, Gloo, Baidu Ring Allreduce
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: a.ring-allreduce | Ring-AllReduce 算法"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. Ring-AllReduce 步骤 ---
hdr(1,TOTAL,"Ring-AllReduce 2 步流程")
why("""设 4 rank, 数据 N=4 块, 每块编号 0/1/2/3。
Step 1 (reduce-scatter): 沿 ring 累加, 每 rank 得到 1 块结果
Step 2 (allgather): 沿 ring 广播, 每 rank 得到全 4 块结果""")
out = ["  步骤     动作                       通信次数   数据/次   每 rank 最终"]
out.append("  init     每 rank 持有 N/P 块         -          -         [a,b,c,d]")
out.append("  Step 1   reduce-scatter 沿 ring     P-1=3     N/P       [s,_,_,_]")
out.append("  Step 2   allgather 沿 ring          P-1=3     N/P       [s,s,s,s]")
out.append("  总通信                                  2(P-1)    N/P       完成")
out.append("  vs naive P2P allreduce:        P-1 次      N        [s,s,s,s] 慢 P 倍")
res("\n".join(out))
mea("""Ring-AllReduce 关键:
  1. 数据切成 P 份, 每 rank 处理 1 份
  2. reduce-scatter 阶段: 每 rank 累加自己的 1 份
  3. allgather 阶段: 把这份广播给所有人
  4. 通信量 2N(P-1)/P, P 大时 ~ 2N
  5. 步数 2(P-1), 与 P 线性""")

# --- 2. 通信量对比 ---
hdr(2,TOTAL,"通信量:vs naive")
why("""对比不同 P 下的总通信量:""")
out = ["  P        Naive (N*P)    Ring (2N(P-1)/P)    加速比"]
for P in [2, 4, 8, 16, 32, 64, 128]:
    naive = P
    ring = 2 * (P - 1) / P
    out.append(f"  {P:3d}      {naive:.2f}*N        {ring:.2f}*N              {naive/ring:.1f}x")
res("\n".join(out))
mea("""通信量对比洞察:
  1. P=2: ring 略快 (1x)
  2. P=8: ring 快 4x
  3. P=64: ring 快 32x
  4. P=128: ring 接近常数 2N
  5. P 大时, ring 几乎不增加通信量
这是为什么大模型训练必用 ring""")

# --- 3. 性能瓶颈 ---
hdr(3,TOTAL,"性能瓶颈:小消息")
why("""Ring 的限制: 步数 2(P-1) 随 P 增。
小消息时, 延迟 > 数据传输时间, 慢。""")
out = ["  消息大小     P=8 耗时 (us)   P=64 耗时 (us)   哪个更慢"]
out.append("  1 KB          20              200              P=64")
out.append("  100 KB        30              220              P=64")
out.append("  1 MB          100             280              P=64")
out.append("  100 MB        2000            2200             P=64 略慢")
out.append("  1 GB          20000           20500            接近 (带宽饱和)")
out.append("  小消息 (1KB)  P=64 慢 10x")
out.append("  大消息 (1GB)  P=64 慢 1.025x")
res("\n".join(out))
mea("""Ring 的延迟 = 2(P-1) * alpha:
  1. P=8: 14 * 2us = 28us
  2. P=64: 126 * 2us = 252us
  3. P=128: 254 * 2us = 508us
小消息瓶颈:
  - 1KB 消息: 延迟 200us, 带宽 5 GB/s (远未饱和)
  - 解决: 树状 allreduce (延迟 log P)
  - 选型: 大消息 ring, 小消息 tree""")

# --- 4. 实现 + 实战 ---
hdr(4,TOTAL,"实现与实战")
why("""Ring-AllReduce 实现:""")
out = ["  框架            实现位置                备注"]
out.append("  NCCL            ncclReduceScatter + ncclAllgather  自动")
out.append("  Gloo            gloo::RingAllreduce     CPU fallback")
out.append("  Horovod         horovod/torch/ring.py   早期框架")
out.append("  BytePS          bytescheduler/ring.cc   字节内部")
out.append("  Ascend HCCL     同 NCCL, 硬件加速        910 标配")
out.append("  自实现          P-1 次 P2P, 沿 ring     教学用")
res("\n".join(out))
mea("""PyTorch 一行调用 (内部用 NCCL):
  import torch.distributed as dist
  dist.init_process_group('nccl')
  for p in model.parameters():
      p.grad = ...
  # 一行完成 ring allreduce
  dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
  p.grad /= dist.get_world_size()
  # 或用 DDP, 框架自动包
  model = DDP(model, device_ids=[local_rank])
# DDP 内部就是 ring allreduce
# Ascend: dist.init_process_group('hccl'), API 一致""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Ring-AllReduce = reduce-scatter + allgather, 2 步沿 ring 跑;
  通信量 2N(P-1)/P, P 大时接近常数 2N, 比 naive P2P 快 P 倍;
  限制: 小消息延迟大 (步数 2(P-1)), 大消息才划算;
  PyTorch DDP / NCCL 默认用 ring, 一行调用。
- 熟手:2(P-1) 步, 每步 N/P 数据; 通信量独立于 P, 扩展性极好;
  小消息瓶颈: 延迟 = 2(P-1)*alpha, P=64 时 ~ 250us;
  选型: 大消息用 ring, 小消息用 tree (低延迟);
  NCCL/HCCL 自动选算法, PyTorch DDP 内部封装 ring。
【进阶】手写 1 个 ring-allreduce (不用 NCCL), 用 4 个进程模拟,
  验证 reduce-scatter + allgather 2 步; 对比 1 个 rank 慢的情况
  (straggler) 对总耗时的影响, 体会 straggler 问题。
EOF
echo "############################################################"
