#!/bin/bash
# ============================================================
# 实验: c.host-device
# 说明: Host 入队 / Device 执行、任务下发
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 异构计算: Host (CPU) 调度, Device (NPU) 执行。
#   Host: 申请内存, 搬数据, 调 aclnn 接口, 等结果
#   Device: 真正算的硬件
# 任务下发:
#   1. Host 申请 device 内存 (aclrtMalloc)
#   2. Host 拷数据 host->device (aclrtMemcpy)
#   3. Host 调算子 (aclnnXxx)
#   4. Device 算 (异步, 不阻塞 host)
#   5. Host 拷结果 device->host
#   6. Host 释放 device 内存
# 关键: Host/Device 异步, 用 stream 排队
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: c.host-device | Host 调度 + Device 执行 + Stream"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. Host/Device 关系 ---
hdr(1,TOTAL,"Host 与 Device 的边界")
why("""Host (CPU) 和 Device (NPU) 各自职责:
  Host:
    - 业务逻辑 (Python/C++)
    - 申请 device 内存
    - 准备数据 (cpu tensor)
    - 调算子
    - 处理结果
  Device:
    - 真正计算的硬件
    - 接收 kernel 指令
    - 异步执行 (不阻塞 host)""")
res("""典型代码流:
  // 1. 申请 device 内存
  void* dev_x = nullptr;
  aclrtMalloc(&dev_x, size, ACL_MEM_MALLOC_HUGE_FIRST);

  // 2. host -> device
  aclrtMemcpy(dev_x, dev_size, host_x, host_size, ACL_MEMCPY_HOST_TO_DEVICE);

  // 3. 调算子
  aclnnAddGetWorkspaceSize(dev_x, dev_y, dev_z, &ws_size, &executor);
  void* workspace; aclrtMalloc(&workspace, ws_size, ...);
  aclnnAdd(workspace, ws_size, executor, stream);

  // 4. 同步
  aclrtSynchronizeStream(stream);

  // 5. device -> host
  aclrtMemcpy(host_z, host_size, dev_z, dev_size, ACL_MEMCPY_DEVICE_TO_HOST);

  // 6. 释放
  aclrtFree(dev_x);""")
mea("""与 CUDA 类比:
  - aclrtMalloc   ≈ cudaMalloc
  - aclrtMemcpy   ≈ cudaMemcpy
  - aclnnXxx      ≈ cuLaunchKernel
  - stream        ≈ cudaStream
  - workspace     ≈ cu 函数 workspace
  - aclrtSync     ≈ cudaStreamSynchronize""")

# --- 2. Stream 与异步 ---
hdr(2,TOTAL,"Stream:Host/Device 异步的关键")
why("""Stream = 任务队列, host 提交到 stream, device 按序执行。
  - 多个 kernel 可在同 stream 串行
  - 不同 stream 可并行 (理论上, 需硬件支持)
  - 不阻塞 host, host 可继续准备下批数据
  - aclrtSynchronizeStream 强制同步等结果""")
res("""Stream 用法:
  // 创建 stream
  aclrtStream stream;
  aclrtCreateStream(&stream);

  // 算子入队
  aclnnAdd(workspace, ..., stream);   // 立即返回
  aclnnMul(workspace, ..., stream);   // 立即返回
  aclnnRelu(workspace, ..., stream);  // 立即返回

  // 这时 host 可继续干别的; device 串行跑这 3 个

  // 拿结果时同步
  aclrtMemcpy(host_out, dev_out, size, ACL_MEMCPY_DEVICE_TO_HOST);
  // 默认隐式同步 device->host 前一操作
  // 显式:
  aclrtSynchronizeStream(stream);

  // 销毁
  aclrtDestroyStream(stream);""")
mea("""Stream 的核心价值: overlap。
  1. 数据搬运 与 计算 overlap (关键性能来源)
  2. 多 stream 让多组 kernel 并行
  3. host 在 device 算时继续准备数据""")

# --- 3. aclnn 接口 ---
hdr(3,TOTAL,"aclnn 接口:统一算子调用规范")
why("""aclnn = Ascend CL NN (类比 cuBLAS/cuDNN):
  算子通过 aclnn 接口调用,无需手写 kernel。
  接口分两步:
    1. aclnnXxxGetWorkspaceSize: 准备, 算 workspace 大小
    2. aclnnXxx: 真正执行
  workspace 是中间内存, host 申请, device 用。""")
res("""常用 aclnn 接口 (选摘):
  aclnnAdd / aclnnMul / aclnnSub / aclnnDiv     元素级
  aclnnMatmul                                    矩阵乘
  aclnnSoftmax / aclnnLayerNorm / aclnnBatchNorm  归一化
  aclnnConv2d / aclnnMaxPool                     CNN
  aclnnFlashAttentionScore                       FlashAttn
  aclnnAllReduce / aclnnAllGather                集合通信
  
  算子查询:  CANN 提供 \"算子大全\" 文档""")
mea("""aclnn 是高层 API, 通常调它就够, 极致性能才手写 AscendC。\n  框架层 (torch_npu / MindSpore) 已经封装好 aclnn, 用户不用手写。""")

# --- 4. 性能优化技巧 ---
hdr(4,TOTAL,"Host/Device 性能优化")
why("""常见优化点:""")
res("""优化点                  做法
  数据搬运隐藏          双 stream (一个算, 一个搬下一批)
  Workspace 复用        一次申请, 多次复用 (避免反复 malloc)
  device 内存复用       算子完成后立即释放, 不要 cache
  异步                    不要每步都 aclrtSynchronizeStream
  batch 大               一次发大 batch, 减少 launch 次数
  算子融合              用 fused 算子 (Linear+ReLU 一次)
  Pin memory            host 端用 pinned memory, 搬得更快""")
mea("""高级: 零拷贝 (zero-copy)
  - Host 内存直接映射到 device 地址空间
  - 设备读写不需 memcpy, 性能极佳
  - 限制: 大小对齐 + 性能可预期
  - 用 aclrtHostRegister / aclrtHostUnregister
""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Host (CPU) 调度 + Device (NPU) 执行;aclnn 接口两步走
  (workspace size + execute);Stream 让 host/device 异步;数据搬运和计算
  可 overlap。
- 熟手:双 stream 隐藏数据搬运;workspace 复用;用 aclnn 高层 API 即可,
  极致性能才手写 AscendC;zero-copy 用 HostRegister 减少 memcpy。
【进阶】用 aclnn 写一个 batch matmul,对比同步 vs 双 stream 异步的性能。
EOF
echo "############################################################"
