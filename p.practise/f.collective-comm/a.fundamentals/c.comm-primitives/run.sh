#!/bin/bash
# ============================================================
# 实验: c.comm-primitives
# 说明: 集合通信原语 (allreduce/allgather/reduce-scatter 等)
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 集合通信原语 (collective primitives) 是分布式训练的核心。
# 8 大原语:
#   1. allreduce:   全员求和 (DP 梯度同步)
#   2. allgather:   全员收集 (ZeRO-1)
#   3. reduce_scatter: 归约切片 (ZeRO-2)
#   4. broadcast:   单点广播 (初始化)
#   5. reduce:      单点归约
#   6. alltoall:    全互换 (MoE)
#   7. gather:      单点收集
#   8. scatter:     单点发散
# 通信量 (N = 数据, P = rank 数):
#   - allreduce:     2N * (P-1) / P
#   - allgather:     N * (P-1)
#   - reduce_scatter: N * (P-1) / P (但 sum = allreduce)
# 关键洞察:
#   - allreduce = reduce_scatter + allgather
#   - 三者通信量相等, 但实现差异
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: c.comm-primitives | 集合通信原语"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 8 大原语 ---
hdr(1,TOTAL,"8 大集合通信原语")
why("""分布式训练的 8 大原语, 覆盖 99% 场景:""")
out = ["  原语              输入 -> 输出              用途"]
out.append("  allreduce         [a,b,c,d] -> [s,s,s,s]   DP 梯度同步 (最常用)")
out.append("  allgather         [a,b,c,d] -> [[a],[b],[c],[d]]  ZeRO-1")
out.append("  reduce_scatter    [a,b,c,d] -> [s1,s2,s3,s4] ZeRO-2")
out.append("  broadcast         [a] -> [a,a,a,a]          参数初始化")
out.append("  reduce            [a,b,c,d] -> [s]          单点汇总")
out.append("  alltoall          [a,b,c,d] -> [a,b,c,d] 切 MoE 专家")
out.append("  gather            [a,b,c,d] -> [[a],[b],[c],[d]] 单点收集")
out.append("  scatter           [a,b,c,d] -> [a_i]        单点发散")
res("\n".join(out))
mea("""8 大原语使用频率:
  1. allreduce: 60% (DP 必备)
  2. allgather: 15% (TP, ZeRO-1)
  3. reduce_scatter: 15% (ZeRO-2, TP)
  4. broadcast: 5% (初始化, 检查点)
  5. alltoall: 3% (MoE, 3D parallel)
  6. 其他: 2%
关键洞察:
  - allreduce = reduce_scatter + allgather
  - 三者通信量相等 (N * (P-1))""")

# --- 2. 通信量分析 ---
hdr(2,TOTAL,"通信量分析:N * (P-1) / P")
why("""每个原语的数据传输量, 决定训练吞吐。
设 N = 单 rank 数据, P = rank 数, BW = 带宽。""")
out = ["  原语              单 rank 数据量     聚合通信量 (P=8)"]
out.append(f"  allreduce         N + N(P-1)/P      {1 + 7/8:.2f} * N per rank")
out.append(f"  allgather         N(P-1)              {7} * N per rank")
out.append(f"  reduce_scatter    N(P-1)/P            {7/8:.2f} * N per rank")
out.append(f"  broadcast         N                   {1} * N (单点发)")
out.append(f"  alltoall          N(P-1)              {7} * N per rank")
out.append(f"  reduce            N(P-1)              {7} * N (单点收)")
res("\n".join(out))
mea("""通信量洞察:
  1. allreduce 通信量 = 2 * N * (P-1) / P (双向)
  2. allgather = reduce_scatter * (P-1)
  3. P 大时, allreduce 通信量 ~ 2N (常数)
  4. alltoall 通信量最大 (N*P), 仅 MoE 用
  5. 训练吞吐 = 算力 / (算 + 通), 通是大头
扩展效率 = N / (N + 通信)""")

# --- 3. 实现算法对比 ---
hdr(3,TOTAL,"实现算法:ring / tree / direct")
why("""同一原语有不同实现, 性能差异大:""")
out = ["  原语            ring-allreduce      tree             direct"]
out.append("  allreduce       2N(P-1)/P           log(P)*N         N(P-1)  (差)")
out.append("  allgather       N(P-1)              log(P)*N         N(P-1)")
out.append("  时间            O(N/P)              O(log P)         O(P)")
out.append("  适用 P          任意                大 (>=16)        小 (<=4)")
out.append("  Ascend 优化     HCCL ring+tree      HCCL 双算法      备选")
out.append("  带宽利用率     高 (N/P)            中 (log P)       低 (N*P)")
out.append("  延迟开销        2P 步              log P 步         P 步")
res("\n".join(out))
mea("""算法选择:
  1. P < 8: tree 或 direct 都行
  2. P >= 8: ring 更好 (常数带宽, 大量数据)
  3. P >= 32: tree 更优 (低延迟)
  4. NCCL/HCCL 自带选择 (基于 P 和 N)
  5. ring 不适合小消息 (延迟大)
ring 通信量分析:
  - 2(P-1) 步, 每步 N/P 数据
  - 总 = 2(P-1) * N/P ~ 2N (P 大)""")

# --- 4. 实战调用 ---
hdr(4,TOTAL,"实战调用 (PyTorch + NCCL)")
why("""PyTorch distributed 调集合通信:""")
out = ["  原语             PyTorch API                    备注"]
out.append("  allreduce         dist.all_reduce(t)            同步, in-place")
out.append("  allreduce (sum)   dist.all_reduce(t, op=ReduceOp.SUM) 默认 SUM")
out.append("  allgather         dist.all_gather(t_list, t)    t_list 长度 P")
out.append("  gather            dist.gather(t, t_list, dst=0) 单点收")
out.append("  scatter           dist.scatter(t, scatter_list, src=0) 单点发")
out.append("  reduce_scatter    dist.reduce_scatter(t, t_list) t_list 长度 P")
out.append("  broadcast         dist.broadcast(t, src=0)      单点广播")
out.append("  alltoall          dist.all_to_all(t_list)       t_list 长度 P")
out.append("  barrier           dist.barrier()                同步点")
res("\n".join(out))
mea("""PyTorch 实战技巧:
  1. all_reduce 默认 SUM, 改 op.SUM / op.MAX / op.MIN
  2. all_reduce 是 in-place (改 t 本身)
  3. all_gather 不 in-place, 输出到 t_list
  4. gather / scatter 要指定 dst / src
  5. NCCL 后端: backend='nccl', CPU 用 'gloo'
  6. 异步: 用 dist.isend/irecv 模式
完整例子 (DP 梯度同步):
  for p in model.parameters():
      p.grad = ...
  for p in model.parameters():
      dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
      p.grad /= dist.get_world_size()""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:集合通信 8 大原语, 最常用 allreduce (60% 训练);
  通信量 allreduce = 2N(P-1)/P, P 大时接近 2N 常数;
  allreduce = reduce_scatter + allgather;
  PyTorch 一行 API: dist.all_reduce / all_gather / all_to_all。
- 熟手:8 原语使用频率 allreduce (60%) + allgather (15%) + reduce_scatter (15%);
  算法选择: P>=8 用 ring (带宽优), P>=32 用 tree (延迟低);
  通信量 alltoall 最大 (N*P), 仅 MoE 用; 扩展效率 = 算 / (算+通);
  PyTorch all_reduce in-place, all_gather 不 in-place 到 t_list。
【进阶】写 1 个 mini-distributed 训练脚本 (2-4 GPU), 跑 3 个原语
  (allreduce / allgather / alltoall), 对比不同 P (2/4/8) 下的耗时,
  验证 ring-allreduce 的 O(N/P) 通信量优势, 画扩展效率图。
EOF
echo "############################################################"
