#!/bin/bash
# ============================================================
# 实验: d.alltoall
# 说明: All-to-All 通信:MoE 专家路由、数据并行分片交换
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# All-to-All = 每 rank 都向所有其他 rank 发数据, 也从所有收数据。
# 数据量: 每 rank 发 N/P, 收 N/P (总共 N)。
# 通信量: N * (P-1) / P (双向)
# 应用:
#   1. MoE 专家路由: token 路由到专家所在 rank
#   2. 3D Parallel 切分: TP group 重排
#   3. Sharded 优化器: 跨 rank 切分
# 4 种实现:
#   - naive: 2P 步 (P send + P recv)
#   - ring: 2(P-1) 步, 每步 N/P
#   - pair exchange: P-1 步 (一次 swap)
#   - direct: P-1 步, 全双工
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: d.alltoall | All-to-All 通信 (MoE 路由)"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. All-to-All 通信模式 ---
hdr(1,TOTAL,"All-to-All 通信模式")
why("""P=4 节点, 每节点向其他节点发不同数据。""")
out = ["  源 rank -> 目标 rank   数据大小"]
out.append("  0 -> 0                1 (留 1/4)")
out.append("  0 -> 1                1")
out.append("  0 -> 2                1")
out.append("  0 -> 3                1")
out.append("  1 -> 0                1 (给 rank 0)")
out.append("  1 -> 1                1")
out.append("  ... 等")
out.append("  每 rank 发出 P 份, 收 P 份")
out.append("  总数据 = P * N (N = 每 rank 总量)")
res("\n".join(out))
mea("""All-to-All 特点:
  1. 最复杂的集合通信, 通信量最大
  2. 数据模式任意 (谁发什么, 谁收什么)
  3. 通信量 = N * (P-1) / P (与 allgather 同)
  4. 步数: naive 2P, ring 2(P-1)
  5. 适用: MoE, 3D parallel, expert parallelism""")

# --- 2. MoE 专家路由 ---
hdr(2,TOTAL,"MoE 专家路由场景")
why("""Mixture of Experts (MoE) 训练中, 每 token 路由到 top-k 专家。
专家分布在不同 rank 上, 需要 all-to-all 交换 token。""")
out = ["  步骤         数据                通信"]
out.append("  0. 路由       4 token, top-2 路由   计算, 不通信")
out.append("  1. 准备       切分 token, 标目标 rank  计算")
out.append("  2. alltoall   token -> 目标 rank    all-to-all, N*B")
out.append("  3. 专家计算   每 rank 算 1 个专家    算力")
out.append("  4. alltoall   结果 -> 源 rank       all-to-all, N*B")
out.append("  5. 合并       还原 batch           计算")
out.append("  总通信量       2 * N * B * (P-1) / P  2 次 all-to-all")
out.append("  N = token 数, B = 维度, P = rank 数")
res("\n".join(out))
mea("""MoE 通信开销:
  1. All-to-all 是 MoE 训练最大瓶颈
  2. P 大时 (>= 32), 通信量 ~ 2N*B
  3. 优化: 专家并行 (EP) 减少 P
  4. 优化: 通信计算 overlap
  5. 优化: token drop (负载均衡)
  6. DeepSeek-V3 64 专家, EP=8 缓解""")

# --- 3. 实现算法对比 ---
hdr(3,TOTAL,"实现算法对比")
why("""All-to-All 的 4 种实现:""")
out = ["  算法         步数       通信量      复杂度    适用"]
out.append("  naive        2P         N*(P-1)/P   O(P^2)  小 P")
out.append("  ring         2(P-1)     N*(P-1)/P   O(P)    通用")
out.append("  pair-swap    P-1        N*(P-1)/P   O(P)    中等")
out.append("  direct       P-1        N*(P-1)/P   O(P)    P 小")
out.append("  Bruck        P-1        N*log P/P   O(P log P) 复杂, 不实用")
out.append("  NVSwitch     1          N*(P-1)/P   O(P)    节点内最快")
res("\n".join(out))
mea("""算法选择:
  1. 节点内 (P=8): NVSwitch 1 步搞定
  2. 节点间: ring (与 allreduce 同算法)
  3. P 大 (>= 16): ring, 步数线性
  4. Bruck 通信量 N*log P, 实际不优 (延迟)
  5. NCCL/HCCL: ring all-to-all (默认)
  6. MoE 优化: EP (专家并行) 减少 P""")

# --- 4. 实战调用 ---
hdr(4,TOTAL,"PyTorch 实战调用")
why("""PyTorch All-to-All:""")
out = ["  原语              API                          备注"]
out.append("  alltoall          dist.all_to_all(...)        旧 API, 慢")
out.append("  alltoall_single   dist.all_to_all_single(...)  1 张量, 快")
out.append("  _all_to_all       dist._all_to_all(...)       内部 API")
out.append("  group all_to_all  new_group + all_to_all      自定义 group")
out.append("  NCCL 内部         ncclAlltoAll                 直接 C++")
out.append("  batched           group start + 多 alltoall     多次合并")
res("\n".join(out))
mea("""PyTorch all-to-all 实战 (MoE 例子):
  # 输入: 每 rank 有 N/P token
  # 目标: 把 token 路由到目标专家所在 rank
  input_list = [t for t in input_split]  # 长度 P
  output_list = [torch.empty_like(t) for t in input_list]
  
  # 调用 (新 API)
  torch.distributed.all_to_all_single(
      output,        # 收 (flattened)
      input,         # 发 (flattened)
      input_split_sizes,   # 每段大小
      output_split_sizes,
      group=ep_group      # EP group
  )
  
  # MoE 优化: 通信计算 overlap
  # 1. async start alltoall
  # 2. 同时算路由
  # 3. wait + 专家计算
  # 关键: overlap 才能藏住通信
  # Ascend 同理, 用 hccl.alltoall_single""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:All-to-All = 每 rank 都向所有其他 rank 发/收数据, 通信量最大;
  MoE 专家路由必备, 2 次 all-to-all (token 去 + 结果回);
  通信量 N*(P-1)/P, 与 allgather 同; Naive 2P 步, ring 2(P-1) 步;
  PyTorch all_to_all_single 一行调用。
- 熟手:All-to-All 是 MoE 训练最大瓶颈, P 大时通信量 ~ 2N*B;
  MoE 优化: 专家并行 (EP) 减少 P + 通信计算 overlap + token drop 负载均衡;
  算法选型: 节点内 NVSwitch 1 步, 节点间 ring; NCCL/HCCL 默认 ring;
  Bruck 算法通信量 N*log P, 但延迟高, 实际不用。
【进阶】在 8 GPU 上模拟 MoE all-to-all: 32 专家, 每 token 路由 top-2;
  测不同 P (2/4/8) 下的 all-to-all 耗时, 画 通信耗时 vs P 曲线;
  试不同 EP 配置 (EP=1/2/4/8), 找 MoE 训练最优的 EP 划分。
EOF
echo "############################################################"
