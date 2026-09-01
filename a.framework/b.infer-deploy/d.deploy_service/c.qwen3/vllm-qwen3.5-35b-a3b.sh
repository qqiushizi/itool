#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
MODEL_PATH="/data2/lxy/models/Qwen/Qwen3.5-35B-A3B"

vllm serve $MODEL_PATH \
--host 0.0.0.0 \
--port 5172 \
--data-parallel-size 1 \
--tensor-parallel-size 8 \
--enable-expert-parallel \
--seed 1024 \
--served-model-name qwen3.5 \
--max-num-seqs 16 \
--max-model-len 133000 \
--max-num-batched-tokens 16384 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--enable-prefix-caching \
--async-scheduling \
--speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_cpu_binding":true}' \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder
