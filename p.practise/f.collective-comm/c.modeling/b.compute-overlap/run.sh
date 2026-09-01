#!/bin/bash
# ============================================================
# 实验: b.compute-overlap
# 说明: 计算与通信 overlap, 通信藏起来
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 通信 overlap = 通信和计算同时进行, 把通信时间藏起来。
# 不 overlap: 总时间 = 算 + 通 (串行)
# overlap: 总时间 = max(算, 通) (并行)
# 收益: 总时间 - min(算, 通)
# 3 个级别:
#   1. 算子级 overlap: 1 个 matmul + 1 个 allreduce, 同时
#   2. layer 级 overlap: 前层 allreduce, 后层 matmul, 同时
#   3. 整图 overlap: 反向时算梯度, 同时 allreduce 上一层
# 实现:
#   - DDP 默认: 反向 + allreduce overlap
#   - FSDP: 多层 overlap
#   - Megatron: TP+DP+PP 三维 overlap
# 关键: 通信是异步的, 启动后不等, 可并行
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: b.compute-overlap | 计算与通信 overlap"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. Overlap 原理 ---
hdr(1,TOTAL,"Overlap 原理:藏通信时间")
why("""不 overlap: 算 100 ms, 通 80 ms, 总 180 ms。
overlap: 算 || 通, 总 max(100, 80) = 100 ms, 省 80 ms (44%)。""")
out = ["  模式        总时间    节省     收益"]
out.append("  串行        算 + 通    0        0%")
out.append("  完全 overlap max(算,通)  min(算,通)  ~30-50%")
out.append("  部分 overlap 算 + 通*0.3  70%*通  ~20-30%")
out.append("  通信主导    max(算,通)  ~算     通信主导时 overlap 必做")
out.append("  算力主导    max(算,通)  ~算     算力主导时 overlap 无效")
res("\n".join(out))
mea("""Overlap 收益条件:
  1. 通信耗时 > 0 (必有通信)
  2. 算力 不 100% 占用 (有空跑通信)
  3. 通信是 异步的 (NCCL 是)
  4. 同步点 不 频繁 (DDP 自动, 手动要小心)
不适用:
  - 算力 100% 占用 (小模型 + 大集群)
  - 同步密集 (DDP 默认 bucket 合并)""")

# --- 2. 3 个级别的 overlap ---
hdr(2,TOTAL,"3 个级别的 overlap")
why("""Overlap 的 3 个粒度:""")
out = ["  级别         粒度        实现                  收益"]
out.append("  算子级        1 op 1 allreduce  手写 + torch stream  5-10%")
out.append("  layer 级     1 layer allreduce + 1 layer 算   DDP 默认  15-25%")
out.append("  整图级        多 layer 异步    FSDP / Megatron  30-50%")
out.append("  反向 + 通信   backward 触发 allreduce   DDP  15-25%")
out.append("  前向 + 通信   forward + 下一块 allreduce  手动  5-10%")
out.append("  PP 双向       1 段前向, 1 段反向         Megatron  30-50%")
res("\n".join(out))
mea("""DDP 的 overlap (最常用):
  1. 启动: bucket 合并梯度, 默认 25 MB / bucket
  2. 反向算到 bucket 边界, 触发 allreduce
  3. allreduce 异步, 同时继续反向下一段
  4. 整段反向结束, 等所有 bucket allreduce
  5. 总时间 ~ max(反向, allreduce) + 最后 wait
FSDP 进一步:
  1. forward 前 allgather 参数
  2. forward 同时, reduce 上一层
  3. 整图级流水线
Megatron 进一步:
  1. TP group 内 allreduce, 与 DP group 重叠
  2. PP 1F1B 调度, 前向反向交叠""")

# --- 3. 异步通信 + Stream ---
hdr(3,TOTAL,"异步通信 + Stream")
why("""异步通信 = 启动后不等, 用 stream / event 同步。""")
out = ["  API              同步?   用途"]
out.append("  dist.all_reduce  同步   简单, 无 overlap")
out.append("  dist.isend/irecv 异步   手写 P2P overlap")
out.append("  dist.batch_isend 异步   batch P2P")
out.append("  torch.cuda.Stream 异步   GPU stream, 通信流")
out.append("  dist.bucket     异步   DDP bucket 化")
out.append("  record_event    异步   同步点")
out.append("  wait_event      阻塞   等流结束")
res("\n".join(out))
mea("""Async overlap 实战 (伪代码):
  # 1. 启动异步 allreduce
  comm_stream = torch.cuda.Stream()
  with torch.cuda.stream(comm_stream):
      handle = dist.all_reduce(grad, op=..., async_op=True)
  # 2. 同时主 stream 继续算
  with torch.cuda.stream(main_stream):
      # 算下一个 matmul
      out = model(next_layer)
  # 3. 同步点
  comm_stream.wait_stream(main_stream)
  main_stream.wait_stream(comm_stream)
  # 此时通信和计算都已完成
  # 总时间 = max(comm, compute)""")

# --- 4. Overlap 调优实战 ---
hdr(4,TOTAL,"Overlap 调优实战")
why("""7B 模型, 8 卡 DP 训练, 优化 overlap:""")
out = ["  优化点                通信耗时 (ms)   overlap 后 (ms)  收益"]
out.append("  默认 (无 overlap)     800             800              0%")
out.append("  DDP 默认 (bucket)     800             600              25%")
out.append("  DDP + AMP             800             500              37%")
out.append("  DDP + AMP + 手动 overlap 800          400              50%")
out.append("  FSDP + 整图 overlap    800             300              62%")
out.append("  Megatron 3D overlap    800             200              75%")
out.append("  Ascend HCCL 同理      800             ~300             62%")
res("\n".join(out))
mea("""调优步骤:
  1. baseline: 不优化, 测总时间
  2. 启用 DDP: 默认 bucket 25 MB, 测时间
  3. 调 bucket 大小: 5/25/50/100 MB, 找最优
  4. 加 AMP: 通信量减半 (FP16 梯度)
  5. 手动 overlap: 关键 sync 点加 wait
  6. 评估 FSDP: 内存紧时收益大
  7. Ascend: 同 HCCL, + 算子融合 + 拓扑感知
经验值: Megatron 3D overlap, 通信占比 50% -> 10%""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Overlap = 通信和计算同时跑, 把通信时间藏起来;
  不 overlap 总时间 = 算 + 通, overlap = max(算, 通), 省 min(算, 通);
  DDP 默认在反向时 overlap allreduce, 收益 25%;
  Megatron 3D overlap 可达 50-75%。
- 熟手:3 个粒度: 算子级 (5-10%) / layer 级 (15-25%) / 整图级 (30-50%);
  DDP bucket 化 (默认 25MB) 触发异步 allreduce;
  Async 通信 + CUDA Stream 是 overlap 的基础;
  调优流程: baseline -> DDP -> bucket size -> AMP -> 手动 overlap -> FSDP;
  Megatron 3D overlap 是大模型训练标配, 通信 50% -> 10%。
【进阶】在 7B 模型 + 8 GPU 上 profile, 对比:
  1. 无 overlap (同步 all_reduce); 2. DDP 默认 (bucket overlap);
  3. DDP + AMP (FP16 通信减半); 4. 手动多 stream overlap;
  画 通信耗时 vs 优化级别 柱状图, 量化每步优化收益。
EOF
echo "############################################################"
