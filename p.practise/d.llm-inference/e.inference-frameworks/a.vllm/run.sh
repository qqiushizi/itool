#!/bin/bash
# ============================================================
# 实验: a.vllm
# 说明: vLLM 架构:PagedAttention、Continuous Batching、调度器、KV cache 管理与核心数据流
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# vLLM 是 UC Berkeley 2023 推出的高性能 LLM 推理引擎,核心创新:
#   1. PagedAttention: KV cache 分页 (类似 OS 虚拟内存)
#   2. Continuous Batching: 实时调度,无等待
#   3. Chunked Prefill: 长 prompt 切块,无 spike
#   4. Prefix Cache: 共享 system prompt
#   5. 推测解码: draft 模型加速
# 数据流:
#   HTTP request → Tokenizer → Scheduler → Engine → Worker (GPU)
#   持续 batching 调度, 每步决定哪些请求跑 prefill, 哪些跑 decode
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: a.vllm | 架构: PagedAttention + Continuous Batching"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 整体架构 ---
hdr(1,TOTAL,"vLLM 整体架构")
why("""vLLM 分层:
  ┌────────────────────────────────────┐
  │ HTTP/RPC Server (OpenAI 兼容 API) │
  │ Tokenizer (BPE/SentencePiece)      │
  │ Scheduler (continuous batching)    │
  │ Engine (核心调度逻辑)               │
  │ Worker (单 GPU 上跑模型)            │
  │ Model (HF / Megatron 加载)         │
  │ KV cache manager (PagedAttention)  │
  └────────────────────────────────────┘""")
res("""vLLM 组件:
  - AsyncLLMEngine: 异步主入口
  - Scheduler: 每步调度,决定 prefill/decode
  - BlockManager: KV cache 物理块管理
  - Worker: 单卡执行单元
  - Speculative: 投机解码模块
  - PrefixCache: 前缀缓存(radix tree)
  - ChunkedPrefill: 分块 prefill""")
mea("vLLM V1 重写了 engine,比 V0 简单 30%,性能更好。\n  API 完全兼容 OpenAI,迁移零成本。")

# --- 2. PagedAttention 核心 ---
hdr(2,TOTAL,"PagedAttention:显存利用率 4×")
why("""PagedAttention = 操作系统虚拟内存思路:
  - KV cache 切成固定大小 block (e.g. 16 token)
  - 逻辑 seq → 物理 block 映射 (block_table)
  - block 可共享 (beam search, prefix cache)
  - 物理内存用满才分配,无碎片""")
res("""vLLM PagedAttention 关键参数:
  --block-size 16         每个 block 装 16 个 token
  --gpu-memory-utilization 0.9  GPU 显存利用率上限
  --swap-space 4          CPU 交换空间(GB)
  --max-num-seqs 256      最大并发序列数
  --max-num-batched-tokens 8192  每 iter 最多 token
  
  显存利用: 80-95% (vs HF 的 20-40%)""")
mea("PagedAttention 是 vLLM 的\"绝招\",论文获 SOSP 2023。\n  同样 7B 模型 + A100, HF 跑 8 并发, vLLM 跑 32 并发。")

# --- 3. Continuous Batching 调度 ---
hdr(3,TOTAL,"调度器:Continuous Batching")
why("""vLLM 调度器每步(iteration)做这些事:
  1. 检查 running 队列(正在 decode 的请求)
  2. 检查 waiting 队列(新来 + prefill 未完成)
  3. 决定:
     - prefill 几个新请求(受限 --max-num-batched-tokens)
     - decode 几个旧请求(全部 or 部分)
  4. 拼成 1 个 batch 跑 1 个 forward
  5. 完成的请求释放资源""")
res("""调度循环(简化):
  step t:
    prefills = pick_prefills(waiting, budget)  # 按 chunked 切
    decodes = all_running()                    # 全部继续
    batch = combine(prefills + decodes)
    out = model.forward(batch)
    for req in out:
      if req.done:
        release_blocks(req)
        send_to_client(req)
    t += 1""")
mea("""这种 iteration-level 调度的好处:
  - 短请求不被等, GPU 满载
  - prefill 和 decode 共存, 不会互相饿死
  - chunked prefill 让长 prompt 也能被切块""")

# --- 4. 部署与监控 ---
hdr(4,TOTAL,"部署 + 监控")
why("""vLLM 启动和监控:""")
res("""# 启动服务 (OpenAI 兼容)
vllm serve /path/to/LLaMA-3-8B-Instruct \\
  --host 0.0.0.0 --port 8000 \\
  --tensor-parallel-size 2 \\
  --gpu-memory-utilization 0.9 \\
  --enable-chunked-prefill \\
  --enable-prefix-caching \\
  --max-model-len 8192

# 客户端
curl http://host:8000/v1/chat/completions -d '...'

# 监控端点
curl http://host:8000/metrics  # Prometheus
curl http://host:8000/v1/models

# 关键 metric
vllm:gpu_cache_usage
vllm:time_to_first_token_seconds
vllm:time_per_output_token_seconds
vllm:num_requests_running
vllm:num_requests_waiting""")
mea("""生产部署:
  - 单卡/单机: 直接 vllm serve
  - 多机 TP: --tensor-parallel-size
  - 量化模型: --quantization gptq/awq
  - 监控: Prometheus + Grafana""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:vLLM = PagedAttention + Continuous Batching + Chunked Prefill;
  显存利用率 4× 提升,OpenAI 兼容 API,生产首选。
- 熟手:调度器 iteration-level,PagedAttention 是 vLLM 杀手锏,prefix cache
  + chunked prefill + speculative 解码是进阶优化;--enable-prefix-caching
  是 system prompt 场景必开;Prometheus /metrics 监控 SLO。
【进阶】vllm serve LLaMA-3-8B,跑 guidellm 压测,看 TTFT/TPOT/吞吐。
EOF
echo "############################################################"
