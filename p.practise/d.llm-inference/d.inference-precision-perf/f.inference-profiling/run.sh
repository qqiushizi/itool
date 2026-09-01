#!/bin/bash
# ============================================================
# 实验: f.inference-profiling
# 说明: 推理性能 profiling 分析方法
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 推理 profiling 与训练不同:
#   - 训练: 1 个 step 包含 5 段(数据/前向/反向/通信/优化器)
#   - 推理: 1 个请求 = 1 prefill + N decode
# 关注点:
#   1. prefill 耗时 vs decode 耗时
#   2. 每个 decode step 内各算子耗时
#   3. KV cache 读写带宽
#   4. 调度开销(GPU 等待时间)
# 工具:
#   - vLLM --profile
#   - nsys profile (timeline)
#   - Nsight Compute (单 kernel)
#   - 火焰图
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: f.inference-profiling | 推理 profiling:prefill/decode/算子"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 推理 step 拆解 ---
hdr(1,TOTAL,"推理 step 拆解:prefill + N × decode")
why("""1 个完整推理请求:
  prefill 1 次: 算 1 次,处理 prompt 所有 token 并行
  decode N 次: 每步算 1 个 token,读 KV
  1 prefill + N decode = 总耗时
  性能分析要分开看 prefill 和 decode""")
res("""请求: prompt=512,生成 200 token,7B 模型, A100:
  prefill:    1 次, ~50ms   ← 看算力
  decode:     200 × 25ms = 5000ms  ← 看访存
  总耗时:    5050ms
  吞吐:      200/5.05 = 40 tok/s 单请求
  占比:      prefill 1%, decode 99%""")
mea("""prefill 时间与 prompt 长度正比(算力密集);
  decode 时间与生成长度正比(访存密集);
  profiling 时要分别看,不能平均。""")

# --- 2. decode step 内算子耗时 ---
hdr(2,TOTAL,"decode step 内算子耗时")
why("""一个 decode step 通常包含:
  1. Attention (Q·K, softmax, P·V)  - 35%
  2. Linear (Q,K,V,O)                - 20%
  3. MLP (gate, up, down)            - 25%
  4. RMSNorm                         - 5%
  5. RoPE                            - 3%
  6. 调度/launch 开销                - 2%
  7. 其他 (residual, copy)            - 10%""")
res("""decode step 算子耗时占比(7B, A100, batch=32):
  算子                  占比     优化
  Attention (FlashAttn) 35%     INT8 KV / 多查询
  Linear (Q,K,V,O)     20%     INT8 / 切 TP
  MLP (gate,up,down)   25%     INT8 / fuse SwiGLU
  RMSNorm              5%      fused kernel
  RoPE                 3%      fused QK
  调度/launch          2%      CUDA Graph
  其他 (residual 等)    10%     fuse""")
mea("""优化优先级:
  1. Attention (35%) - FlashAttn 已经做,但 KV 还能优化
  2. MLP (25%) - INT8 GEMM
  3. Linear (20%) - 切 TP, INT8
  4. 调度 (2%) - CUDA Graph 减少 launch
  5. 小算子 - fused kernel""")

# --- 3. vLLM profiling 实战 ---
hdr(3,TOTAL,"vLLM profiling 命令速查")
why("""vLLM 自带 profiling 工具:""")
res("""# 1. 启 vllm 时开 profiler
vllm serve /path/to/model --enable-profiling --profile-result-dir ./prof

# 2. 或者用 nsys 直接 profile
nsys profile -o output.nsys-rep \
  -t cuda,nvtx,osrt \
  --capture-range=nvtx --nvtx-capture=prefill:decode \
  vllm serve ...

# 3. 离线 benchmark 时 profile
python3 -m vllm.benchmark --profile ./prof ...

# 4. 生成的 trace 可视化
nsys-ui output.nsys-rep  # 或 chrome://tracing for json""")
mea("""vLLM 暴露的 metric 端点 /metrics:
  vllm:gpu_cache_usage       KV 利用率
  vllm:request_success       请求数
  vllm:time_to_first_token   TTFT 分布
  vllm:time_per_output_token TPOT 分布
  vllm:num_requests_swapped  被 swap 的请求数
  Prometheus 抓 + Grafana 画""")

# --- 4. 排查清单 ---
hdr(4,TOTAL,"常见性能问题排查清单")
why("""推理性能问题的 8 大症状:""")
res("""症状                          排查方法
  TTFT 高 (> 500ms)            看 prefill kernel 占比 / prompt 长度 / chunked
  TPOT 高 (> 50ms)             看 decode kernel / KV 大小 / 访存
  GPU 利用率低 (< 50%)         看调度/launch 开销 / batch 不够
  KV cache OOM                 调 max_num_seqs / 开 INT8 KV
  偶发 spike                   看 P99 延迟 / 调度器 / 长尾请求
  抖动大                       看 NCCL / PCIe / 邻居卡
  吞吐低                       看 batch / max_num_seqs / FlashAttn
  功耗高                       看 GEMM 利用率 / 切 FP8 / 降频""")
mea("""通用排查流程:
  1. 抓 metric:TTFT P50/P99, TPOT P50/P99, GPU util
  2. 对比 baseline:同模型/同硬件/同配置
  3. 抓 nsys:看具体 kernel 耗时
  4. 逐项优化:从最大占比开始
  5. 回归测试:每次优化对比基线""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:推理 profiling 分 prefill 和 decode 两侧;decode 内 attention 占 35%
  最大;vLLM 自带 profiling + nsys 看 timeline;/metrics 端点监控 SLO。
- 熟手:从最大占比 kernel 开始优化;CUDA Graph 减 launch 开销;TTFT/TPOT
  P50/P99 + GPU util + KV 利用率是黄金 4 件套;nsys-ui 火焰图找最耗 kernel。
【进阶】在 vllm 启 --enable-profiling,跑 5 分钟请求,nsys-ui 看 prefill 和
  decode 阶段 timeline。
EOF
echo "############################################################"
