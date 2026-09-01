#!/bin/bash
# ============================================================
# 实验: c.collective-ops
# 说明: 集合通信算子选型与对比 (DP/TP/PP 通信模式)
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 不同并行方式用不同集合通信算子:
#   - DP (data parallel): allreduce
#   - TP (tensor parallel): allreduce + allgather + reduce_scatter
#   - PP (pipeline parallel): P2P send/recv
#   - EP (expert parallel): all-to-all
#   - SP (sequence parallel): allgather + reduce_scatter
#   - ZeRO-1/2/3: reduce_scatter / allgather
# 通信量排名 (1 个 transformer layer, h=hidden):
#   - DP allreduce: 2h
#   - TP allreduce: 2h (列切) + 2h (行切) = 4h
#   - PP send/recv: 2h (1 个 hidden)
#   - EP all-to-all: 2h (路由 + 反向)
# 优化方向: 减少通信次数, 减少通信量
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: c.collective-ops | 集合通信算子选型"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 并行方式 vs 通信算子 ---
hdr(1,TOTAL,"并行方式 vs 通信算子")
why("""主流并行方式用的集合通信:""")
out = ["  并行方式    通信算子                    通信量 (per step)"]
out.append("  DP          allreduce                   2 * G (梯度)  * (P-1)/P")
out.append("  TP (列切)   allreduce                   2 * X * (T-1)/T")
out.append("  TP (行切)   allreduce + reduce_scatter  X * (T-1)/T")
out.append("  PP          send/recv                   2 * H (1 层) * (Pp-1)")
out.append("  EP (MoE)    all-to-all                  2 * T * B * (E-1)/E")
out.append("  SP          allgather + reduce_scatter  X * (T-1)/T")
out.append("  ZeRO-1      allgather (参)              2 * P_size (优化器状态)")
out.append("  ZeRO-2      reduce_scatter (梯度)       1/P * grad")
out.append("  ZeRO-3      allgather (参)              1/P * param")
out.append("  FSDP        allgather + reduce_scatter  1/P * (param + grad)")
res("\n".join(out))
mea("""通信算子分布:
  1. allreduce: DP, TP, ZeRO, 80% 场景
  2. allgather: ZeRO-1/3, SP, 10%
  3. reduce_scatter: ZeRO-2, FSDP, TP, 5%
  4. all-to-all: EP, MoE, 3%
  5. send/recv: PP, 2%
不同并行的通信量级:
  - DP: 2G (大, 7B = 14 GB)
  - TP: 4X (X = activation)
  - PP: 2H (H = hidden, 比 DP 小 10x)
  - EP: 2TB (T = tokens, B = batch)""")

# --- 2. 通信量对比 ---
hdr(2,TOTAL,"7B 模型各并行通信量对比")
why("""7B 模型 (h=4096, L=32 层, bs=32, seq=2048) 各并行方式通信量:""")
out = ["  并行        配置        通信量 / step"]
out.append("  DP          P=8         2 * 14GB * 7/8 = 24.5 GB")
out.append("  DP          P=64        2 * 14GB * 63/64 = 27.6 GB")
out.append("  TP          T=8         4 * 32 MB * 7/8 = 112 MB (per layer)")
out.append("  TP          T=8 全模型  112 MB * 32 = 3.6 GB")
out.append("  PP          P=8         2 * 16 KB * 7 = 224 KB (per step)")
out.append("  PP          P=8 全模型  224 KB * 32 = 7.2 MB (微)")
out.append("  EP (MoE)    E=8         2 * 32 MB = 64 MB (per layer)")
out.append("  ZeRO-3      P=8         14 GB / 8 = 1.75 GB (param) + 通信")
out.append("  FSDP        P=8         2 * 14 GB / 8 = 3.5 GB")
res("\n".join(out))
mea("""通信量洞察:
  1. DP 通信量最大 (14 GB 量级), 难压缩
  2. TP 通信量小 (3.6 GB), 但有 allreduce 次数多
  3. PP 通信量最小 (MB 级), 但有流水线 bubble
  4. EP 通信量小但 all-to-all 慢
  5. ZeRO/FSDP 通信量 = DP / P, 减少但多了 allgather
实战:
  - DP 起步, P 上去用 ZeRO/FSDP 省内存
  - 模型太大 (单卡装不下) 用 TP (单层切)
  - 还装不下用 PP (跨层切)""")

# --- 3. 通信算子选型决策 ---
hdr(3,TOTAL,"选型决策:DP/TP/PP/EP")
why("""选 并行方式 的决策树:""")
out = ["  决策点             选择     原因"]
out.append("  模型能装下单卡?    DP       简单, 通信易 overlap")
out.append("  装不下单卡         TP       单层切, 通信小, 频繁")
out.append("  TP 装不下          PP       跨层切, 通信最小, 有 bubble")
out.append("  模型是 MoE         EP       专家切, all-to-all")
out.append("  内存紧             ZeRO/FSDP  优化器状态切分")
out.append("  大模型 + 训练稳定   3D Parallel  DP+TP+PP 综合")
out.append("  LLM 主流           TP+PP+DP   Megatron-LM 标配")
res("\n".join(out))
mea("""选型实战:
  1. < 13B 参数: DP + ZeRO-2 起步
  2. 13B-70B: TP=8 + DP (FSDP)
  3. 70B-175B: TP=8 + PP=2 + DP (FSDP)
  4. 175B+: TP=8 + PP=8 + DP (FSDP)
  5. MoE (Mixtral 8x7B): TP=4 + EP=2 + DP (FSDP)
  6. Ascend 同, + HCCL 自动拓扑感知
通信量排序 (从大到小):
  DP > FSDP > ZeRO-3 > ZeRO-2 > TP > EP > PP
  (DP 最大, PP 最小)""")

# --- 4. 通信算子调用 ---
hdr(4,TOTAL,"实战调用对照")
why("""各并行方式对应的 PyTorch 调用:""")
out = ["  并行        PyTorch API                                备注"]
out.append("  DP          model = DDP(model)                          1 行")
out.append("  FSDP        model = FSDP(model, sharding_strategy=...)  1 行")
out.append("  TP (手动)   dist.all_reduce + dist.all_gather           手写")
out.append("  TP (DeepSpeed) 引擎配置                                  框架")
out.append("  TP (Megatron) tensor_parallel 内置                       框架")
out.append("  PP (手动)   dist.send + dist.recv                       手写")
out.append("  PP (PyTorch) Pipe (deprecated)                          框架")
out.append("  PP (Megatron) PipelineParallel 内置                    框架")
out.append("  EP (手写)   dist.all_to_all_single                      手写")
out.append("  EP (Megatron) ExpertParallel 内置                       框架")
res("\n".join(out))
mea("""实战建议:
  1. 起步: 1 行 DDP, 跑通
  2. 内存紧: 1 行 FSDP
  3. 模型大: DeepSpeed 引擎 (ZeRO + AMP)
  4. 极大模型: Megatron-LM (3D Parallel)
  5. MoE: DeepSpeed MoE / Megatron EP
  6. Ascend: MindSpeed (类似 Megatron)
不要自己造轮子, 优先用框架:
  - DP 通信 DDP 帮你写
  - TP/PP 通信 DeepSpeed/Megatron 帮你写
  - 自己写容易有 bug (死锁, 同步错)
  - 性能优化交给框架 (NCCL/HCCL)""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:DP 用 allreduce, TP 用 allreduce + allgather + reduce_scatter;
  PP 用 P2P send/recv, EP 用 all-to-all;
  通信量排序 DP > FSDP > ZeRO-3 > ZeRO-2 > TP > EP > PP;
  选型: 模型能装下单卡用 DP, 装不下用 TP/PP, MoE 用 EP。
- 熟手:7B 模型 DP P=8 通信 24.5 GB / step, TP 3.6 GB, PP 7.2 MB (小);
  选型决策: 13B DP+ZeRO, 70B TP+DP+FSDP, 175B+ 3D Parallel;
  各并行的 PyTorch API: DP 1 行 DDP, FSDP 1 行, TP/PP 用 DeepSpeed/Megatron;
  不要造轮子, 用框架, 通信优化交给 NCCL/HCCL。
【进阶】在 1 个 8 GPU 节点上, 用 7B 模型实测 4 种并行的通信耗时:
  1. DDP (P=8); 2. FSDP (P=8); 3. 手动 TP (T=2+DP=4); 4. DeepSpeed ZeRO-2;
  画 通信耗时 vs 并行方式 柱状图, 找 最优并行配置 (吞吐 / 内存 / 收敛)。
EOF
echo "############################################################"
