#!/bin/bash
# ============================================================
# 实验: b.point-to-point
# 说明: 点对点通信 (send/recv)、带宽延迟建模
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 点对点通信 (P2P) = 两个 rank 之间的 send/recv。
# 是其他集合通信 (allreduce/allgather) 的基础。
# 关键概念:
#   - 延迟 (latency): 一次通信的固定开销, ~1-10us
#   - 带宽 (bandwidth): 持续传输速率, GB/s
#   - 通信模型: T = alpha + N * beta
#     alpha = 延迟, beta = 1/带宽, N = 数据量
#   - 双向带宽: send + recv 同时, 接近 2 倍
# 硬件 P2P:
#   - GPUDirect/NPU Direct: 显存直传, 绕过 CPU
#   - RDMA: 跨节点, 0 拷贝
#   - GPU/NPU -> CPU: 慢, 1-2 GB/s
# 同步问题:
#   - send 不等 recv 完成, 要 sync
#   - 死锁: 双方都 send 等 recv, 永远等
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: b.point-to-point | 点对点通信与带宽延迟建模"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. P2P 通信 API ---
hdr(1,TOTAL,"P2P 通信 API:send/recv")
why("""P2P 的两个基本 API:
  - send: 把本地 buffer 发到对方
  - recv: 接收对方 buffer 到本地
  - 阻塞: 同步等 (简单, 慢)
  - 非阻塞: 立即返回, 用 wait/sync 等
  - batched: isend/irecv, 多次通信并行
NCCL 也有 P2P: ncclSend / ncclRecv""")
out = ["  API             阻塞?   用途                    备注"]
out.append("  send/recv       是      简单场景                易死锁")
out.append("  isend/irecv     否      并行 P2P               配 wait()")
out.append("  batched_isend   否      大规模 P2P            NCCL 推荐")
out.append("  ncclSend        异步    NCCL 通信              需 communicator")
out.append("  ncclGroupStart  集合     多个 P2P 一起提交     降低启动开销")
out.append("  cudaMemcpy      是      GPU<->GPU (P2P)        旧 API, 不推荐")
out.append("  ncclSend/Recv   异步    NPU 通信              HCCL 同")
res("\n".join(out))
mea("""P2P API 选型:
  1. 简单 send/recv: 调试用, 易死锁
  2. isend/irecv: 大部分场景
  3. batched: 多对多 P2P (如 MoE 专家)
  4. ncclGroupStart: 一次提交多个, 省启动
  5. 死锁常见: 双方都 send 等 recv
避免死锁: 奇数 rank send, 偶数 rank recv""")

# --- 2. 带宽延迟建模 ---
hdr(2,TOTAL,"带宽延迟模型:T = alpha + N*beta")
why("""通信时间 = 延迟 + 数据量/带宽
  T = alpha + N / BW
实测参数 (NVLink + IB):
  - 节点内: alpha=1us, BW=600 GB/s
  - 节点间: alpha=2us, BW=25 GB/s (200 Gb/s)
用模型估通信时间, 优化决策。""")
import numpy as np
def comm_time(N_bytes, alpha_us, bw_gbps):
    bw_bytes = bw_gbps * 1e9 / 8
    return alpha_us + N_bytes / bw_bytes * 1e6  # us

out = ["  N (MB)   节点内 (us)   节点间 (us)   节点间慢几倍"]
out.append(f"  0.001    {comm_time(1024, 1, 4800):6.1f}      {comm_time(1024, 2, 25):6.1f}        {comm_time(1024, 1, 4800)/comm_time(1024, 2, 25):.1f}x")
for n_mb in [0.1, 1, 10, 100, 1000]:
    t_intra = comm_time(n_mb*1024*1024, 1, 4800)
    t_inter = comm_time(n_mb*1024*1024, 2, 25)
    out.append(f"  {n_mb:7.1f}   {t_intra:9.1f}    {t_inter:9.1f}    {t_inter/t_intra:.1f}x")
res("\n".join(out))
mea("""带宽延迟模型洞察:
  1. 小消息 (1KB): 节点内 1us, 节点间 2us, 慢 2x
  2. 中等 (1MB): 节点内 2us, 节点间 40us, 慢 20x
  3. 大消息 (1GB): 节点内 220us, 节点间 40ms, 慢 200x
  4. 优化方向: 节点内能融就不出节点
  5. 大消息: 节点间带宽是瓶颈, 异步 overlap""")

# --- 3. 死锁与同步 ---
hdr(3,TOTAL,"死锁、同步与重叠")
why("""P2P 最常见的坑: 死锁。""")
out = ["  死锁场景         描述                解决方案"]
out.append("  A 发 -> B 收      OK                 OK")
out.append("  A 发, B 发        A 等 B, B 等 A     用奇偶法, 一边 send 一边 recv")
out.append("  A 收, B 收        A 等数据, B 不发   同上")
out.append("  A 发, B 发 (无buffer) 都需要 recv     isend, 不阻塞")
out.append("  A 发 (大), B 收(小)  大消息分块      多次小 send/recv")
out.append("  A->B->C->A 循环  环死锁             用 ring, 1 次 1 步")
res("\n".join(out))
mea("""P2P 同步的 3 个策略:
  1. 阻塞 send/recv: 简单, 易死锁
  2. 非阻塞 isend/irecv + wait: 灵活
  3. batched (group start): 最优
重叠 (overlap):
  - 计算和通信同时: 大消息分块
  - send chunk 1, 同时算 chunk 2
  - 通信时间藏起来""")

# --- 4. P2P 与集合通信 ---
hdr(4,TOTAL,"P2P vs 集合通信")
why("""P2P 是基础, 集合通信在 P2P 上构建:""")
out = ["  集合通信         内部 P2P 次数    用途"]
out.append("  allreduce        2*(N-1)         梯度同步 (DP)")
out.append("  allgather        N-1             权重广播 (ZeRO-1)")
out.append("  reduce_scatter   N-1             梯度切片 (ZeRO-2)")
out.append("  broadcast        N-1             参数初始化")
out.append("  alltoall         N-1             专家路由 (MoE)")
out.append("  barrier          N-1             同步点")
out.append("  gather           N-1             单点收集")
out.append("  scatter          1               单点散发")
res("\n".join(out))
mea("""为什么用集合通信而不是多次 P2P:
  1. 优化算法: ring-allreduce 比 N-1 次 P2P 快 O(N) 倍
  2. 拓扑感知: 自动选最佳路径
  3. 流水线: 大消息分块, overlap
  4. 错误处理: timeout, retry
  5. 框架封装: torch.distributed 一行调用
实战:
  - 90% 场景用 allreduce / allgather
  - 10% (MoE, 不规则) 用 P2P
  - 自定义算法才手写 P2P""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:P2P = send/recv 两个 rank 之间传数据, 是集合通信的基础;
  通信时间 = 延迟 + 数据量/带宽; 节点内快 200x (NVLink vs IB);
  死锁是大坑, 用奇偶法避免 (一边 send 一边 recv)。
- 熟手:通信模型 T = alpha + N*beta, 节点内 alpha=1us BW=600GB/s, 节点间 alpha=2us BW=25GB/s;
  isend/irecv + wait 是大部分场景默认; ncclGroupStart 批量提交省启动;
  P2P 同步 3 策略: 阻塞 (易死锁) / 非阻塞 + wait (灵活) / batched (最优);
  90% 场景用集合通信, 10% (MoE) 用 P2P。
【进阶】写 1 个 P2P benchmark: 测不同 N (1KB-1GB) 的 send/recv 延迟和带宽,
  拟合 T = alpha + N*beta 模型, 画 节点内 vs 节点间 对比图;
  测试 isend/irecv 与阻塞 send/recv 的 overlap 收益。
EOF
echo "############################################################"
