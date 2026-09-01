#!/usr/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=8
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export USE_MULTI_GROUPS_KV_CACHE=1

export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=512

export USE_MULTI_BLOCK_POOL=1

sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

vllm serve /data2/lxy/model/DeepSeek-V4-Flash-w8a8-mtp \
  --safetensors-load-strategy 'prefetch' \
  --max-model-len 135168 \
  --max-num-batched-tokens 4096 \
  --served-model-name ds \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 16 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --port 7000 \
  --block-size 128 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --async-scheduling \
  --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":false}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2,4,6,8,10,12,14,16,18,20,22,24,32,36,40]}' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
  --speculative-config '{"num_speculative_tokens": 1,"method": "mtp"}'
