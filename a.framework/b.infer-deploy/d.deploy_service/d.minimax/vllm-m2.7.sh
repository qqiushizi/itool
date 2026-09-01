export VLLM_ENGINE_READY_TIMEOUT_S=1800
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=10
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1

export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1

vllm serve /data2/lxy/models/MiniMax-M2.7 \
    --served-model-name "MiniMax-M2.7" \
    --host 0.0.0.0 \
    --port 5172 \
    --trust-remote-code \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}' \
    --enable-expert-parallel \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --max-num-seqs 8 \
    --max-model-len 40690 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.92 \
    --speculative_config '{"enforce_eager": true, "method": "eagle3", "model": "/data2/lxy/models/MiniMax-M2.7-EAGLE3", "num_speculative_tokens": 3}' \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think
