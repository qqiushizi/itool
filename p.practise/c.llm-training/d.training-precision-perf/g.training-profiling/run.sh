#!/bin/bash
# ============================================================
# 实验: g.training-profiling
# 说明: 训练性能 profiling 分析方法
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 训练 step 耗时 = 计算 + 通信 + 访存 + 调度。
# 找瓶颈:
#   1. CPU/GPU 利用率(nvidia-smi / psutil)
#   2. 算子级时间(哪些 kernel 占最多)
#   3. 通信时间(NCCL/HCCl 占比)
#   4. 内存带宽(roofline: 算力 vs 带宽)
# 工具:
#   - PyTorch Profiler (torch.profiler)
#   - NVIDIA Nsight Systems / Nsys
#   - DeepSpeed: --profile_step
#   - Ascend: msprof
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: g.training-profiling | 训练耗时拆解与瓶颈定位"
echo "############################################################"

python3 <<'PY'
import numpy as np, time
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 5

# --- 1. 拆解一个 step 的时间组成 ---
hdr(1,TOTAL,"一个训练 step 的 5 段拆解")
why("""典型 1 step = 前向 + 反向 + AllReduce + 优化器 + 数据加载(异步)。
大模型这 5 段大致占比:前向 25%, 反向 35%, 通信 25%, 优化器 10%, 杂 5%。""")
seg = [("数据加载", 5), ("前向", 25), ("反向", 35), ("梯度AllReduce", 25), ("优化器更新", 10)]
total = 100
print()
out = (
    "  段                  占比  典型瓶颈\n  " + "-"*50 + "\n  "
    + "\n  ".join(f"{n:<14s}  {p:>3d}%   {b}" for n,p,b in [
        ("数据加载", 5, "DataLoader 慢 / CPU 跟不上"),
        ("前向", 25, "模型太大 / 算子效率低"),
        ("反向", 35, "FlashAttn 未开 / 算子低效"),
        ("AllReduce", 25, "通信量大 / 重叠不充分"),
        ("优化器", 10, "AdamW 慢 / ckpt 粒度细"),
        ("其它", 5, "日志 / checkpoint 写盘"),
    ])
)
res(out)
mea("""'优化器 10%' 看着不多,但 LLM 里梯度 fp32 cast + AdamW 双精度更新,
是出名的隐藏杀手。先看哪段占比最大,再针对性优化。""")

# --- 2. 模拟 matmul vs elementwise 的时间差 ---
hdr(2,TOTAL,"算力密集 vs 访存密集:numpy 实测")
why("""理论:matmul 算力密集,elementwise 访存密集。同样 FLOPs 算前者快。
我们在 CPU 上跑 1024×1024 matmul 和 elementwise,看实际差距。""")
def matmul_bench():
    A = np.random.randn(512,512); B = np.random.randn(512,512)
    t=time.perf_counter()
    for _ in range(5): C = A @ B
    return (time.perf_counter()-t)/5
def ew_bench():
    A = np.random.randn(1024,1024)
    t=time.perf_counter()
    for _ in range(20): A = A*0.5 + 1.0
    return (time.perf_counter()-t)/20
mm = matmul_bench(); ew = ew_bench()
# FLOPs
mm_flops = 2*512**3
ew_flops = 1024**2  # 一次加一次乘
mm_gflops = mm_flops/mm/1e9
ew_gflops = ew_flops/ew/1e9
res(f"""CPU 实测 (numpy 单线程):
  matmul  512×512: {mm*1000:.2f} ms,  {mm_gflops:.2f} GFLOPS
  elementwise 1024×1024 (×2 ops): {ew*1000:.3f} ms,  {ew_gflops:.2f} GFLOPS
  → 算力差 ~{mm_gflops/ew_gflops:.0f}×""")
mea("""matmul 算力密集(CPU 也有 BLAS 加速),elementwise 受内存带宽限制。
A100 上 matmul ~312 TFLOPS(FP16),elementwise 只 ~1.5 TF/s(访存 2 TB/s / 4B)。
因此:小算子、激活函数、LayerNorm 是访存瓶颈;Linear/Conv 是算力瓶颈。""")

# --- 3. 通信模拟 ---
hdr(3,TOTAL,"通信时间建模:AllReduce 8 卡 bf16")
why("""N 卡 AllReduce 通信量 = 2(P-1)/P * data_size。8 卡 bf16 32MB 张量:
  vol = 2*(8-1)/8 * 32MB = 56 MB
NVLink 600 GB/s 下时间 ≈ 56/600/1000 ≈ 0.1 ms
但 PCIe/NCCL 启动开销 ~5-10 μs,小张量下是瓶颈。""")
P, size, bw = 8, 32*1024*1024, 600e9
vol = 2*(P-1)/P * size
t_comm = vol/bw
res(f"""AllReduce 32MB, 8 卡 NVLink 600GB/s:
  通信量: {vol/1e6:.1f} MB
  理论耗时: {t_comm*1e6:.1f} μs
  NCCL launch overhead: ~5-10 μs
  → 小张量下 launch 开销占主导""")
mea("""优化通信:
  1. 梯度 bucket 合并(默认 25MB 桶)→ 减少 launch 次数
  2. 计算-通信 overlap(下一层算子的反向和这一层 AllReduce 并行)
  3. 拓扑感知(同 node 内 NVLink,跨 node IB/RoCE)""")

# --- 4. 瓶颈定位决策树 ---
hdr(4,TOTAL,"瓶颈定位决策树")
why("""按下面流程逐项排查,从最大占比的段开始:""")
res("""step 慢 → 哪段最慢?
  ├─ 数据加载占比 > 20%
  │   → DataLoader num_workers 不够,开 pin_memory,异步 prefetch
  ├─ 前向/反向 占比 > 50%
  │   ├─ matmul/conv 占比大 → 算力瓶颈 → 开 TF32/FP16/BF16,CUDA Graphs
  │   ├─ elementwise 占比大 → 访存瓶颈 → 算子融合(FLASHATTN/Flash LayerNorm)
  │   └─ attention 占比大 → 开 FlashAttn
  ├─ AllReduce 占比 > 20%
  │   → overlap 不充分 → 开 DDP bucket + bucket_cap_mb
  │   → 通信量大 → 开 FSDP/ZeRO 分片,通信量减小
  └─ 优化器 占比 > 10%
      → 切 8bit 优化器 / fused AdamW (apex / transformers)""")
mea("""真机:跑 torch.profiler → Chrome trace → 看 kernel 占比。
N 卡间 timeline 看 GPU0/1 是否同步(bubble 越大越慢)。""")

# --- 5. profiling 工具速查 ---
hdr(5,TOTAL,"profiling 工具速查")
why("""不同工具擅长不同场景:""")
res("""工具                       适用场景                          优势
  -------------------------------------------------------------------
  torch.profiler              PyTorch 模型全栈                 易用,导出 Chrome trace
  NVIDIA Nsight Systems       GPU kernel 级 + NVTX 标记         性能最权威
  NVIDIA Nsight Compute       单个 kernel 优化 (SASS 级)       寄存器/共享内存分析
  DeepSpeed timeline          分布式 step timeline             多卡同步可视化
  msprof (Ascend)             昇腾 AI Core 级 timeline          Cube/Vector/MTE 占比
  py-spy / scalene            Python 端瓶颈                    CPU 代码级
  nvidia-smi / dmon           实时 GPU 利用率/显存              运维监控""")
mea("""先 torch.profiler 跑一个 step,看 Chrome trace 找最大 kernel;
再针对性上 Nsight Compute 分析这个 kernel。
Ascend 用 msprof --output=...;导出 timeline.json 配 ASCEND_HOME/tools 工具。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:训练慢要先知道慢在哪——拆 step 为 5 段(数据/前向/反向/通信/优化器)。
  数据加载慢加 num_workers,前向慢开 FlashAttn,通信慢开 overlap。
- 熟手:先 torch.profiler 看 kernel 分布,再上 Nsight Compute 优化瓶颈
  kernel;MFU(Model FLOPs Utilization) = 实际 FLOPS / 峰值 FLOPS,>50% 算优秀。
【进阶】学看 Chrome trace 颜色(紫色=计算,绿色=内存,黄色=通信);自己模型
  跑一次 torch.profiler.schedule 找出占比 top-3 的 kernel。
EOF
echo "############################################################"
