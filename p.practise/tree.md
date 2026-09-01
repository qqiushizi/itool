# p.practise — AI 第一性原理学习实验目录

> 目标:从第一性原理出发,用「原理讲解 + 自动化模拟实验 + 注释讲解 + 小白→熟手递进结论」的形式,
> 帮助初学者一步步提升至熟手。每个叶目录含 `run.sh`,纯 CPU + numpy/标准库即可模拟,
> 有 NPU/GPU 硬件时自动启用真实算子。
>
> 命名沿用仓库约定:顶层 `字母.主题`,子目录 `字母.名称`,叶目录放 `run.sh`。
> `itool.sh` 按目录前缀字母排序导航,`run.sh` 作为选项 0。

---

## 总览

```
p.practise/
├── a.ai-fundamentals/          AI 基础(数学→经典ML→神经网络→优化泛化)
├── b.model-architectures/      不同模态的典型架构与算法
├── c.llm-training/             大模型训练及精度性能
├── d.llm-inference/            大模型推理及精度性能
├── e.ascend-operator/          昇腾算子开发
└── f.collective-comm/          集合通信原理与算法
```

---

## A. AI 基础  `p.practise/a.ai-fundamentals/`

### a.math-basics/  数学基础
- `a.linear-algebra/` — 向量/矩阵/张量、秩、SVD 直观理解(numpy 模拟分解与降维)
- `b.probability-stats/` — 分布、贝叶斯、期望方差、采样与大数定律可视化
- `c.calculus-optimization/` — 导数/梯度/链式法则、梯度下降收敛轨迹
- `d.information-theory/` — 熵、KL 散度、交叉熵从投硬币推导到分类损失

### b.classical-ml/  经典机器学习
- `a.linear-regression/` — 最小二乘 vs 正规方程 vs 梯度下降对比
- `b.logistic-regression/` — 决策边界、损失曲线、概率校准
- `c.trees-boosting/` — 决策树分裂/信息增益、GBDT 直觉
- `d.clustering/` — KMeans 迭代过程、距离度量与初始化敏感性
- `e.dimensionality-reduction/` — PCA 推导与实现、与 SVD 的关系

### c.neural-network-basics/  神经网络基础
- `a.perceptron-mlp/` — 感知机→MLP 前向/反向手写实现
- `b.activation-functions/` — 各激活函数形状、梯度、饱和区对比
- `c.backprop-from-scratch/` — 计算图 + 手算反向传播
- `d.loss-functions/` — MSE/CE/Focal/对比损失 推导与曲线对比
- `e.regularization/` — L1/L2/Dropout/BatchNorm 过拟合抑制效果

### d.optimization-generalization/  优化与泛化
- `a.optimizers/` — SGD/Momentum/AdaGrad/RMSProp/Adam 优化轨迹对比
- `b.lr-schedule/` — warmup/cosine/余弦重启 学习率曲线
- `c.generalization/` — 过/欠拟合、偏差-方差、学习曲线
- `d.evaluation-metrics/` — 准确率/精确召回/F1/AUC/混淆矩阵

---

## B. 模型架构与算法  `p.practise/b.model-architectures/`

### a.nlp-llm/  NLP 与语言模型
- `a.word-embeddings/` — one-hot→word2vec→上下文向量、相似度演变
- `b.rnn-lstm/` — 序列建模、门控、梯度消失/爆炸模拟
- `c.transformer-core/` — 自注意力 QKV、多头、位置编码手算
- `d.transformer-block/` — 残差/LayerNorm/FFN 组装、参数量计算
- `e.decoder-llm/` — GPT 式 decoder、因果 mask、KV-cache 直觉

### b.vision/  视觉
- `a.cnn-basics/` — 卷积/池化/感受野、特征图与参数量
- `b.classic-backbones/` — ResNet 残差/Inception/VGG 演进对比
- `c.vit/` — 图像分块→Transformer、cls token
- `d.detection-segmentation/` — 检测头/anchor、分割上采样直觉
- `e.dit/` — Diffusion Transformer、adaLN-Zero、patchify 去噪
- `f.omni/` — 全模态、文本/图像/音频 token 统一、跨模态注意力
- `g.world-model/` — 世界模型、潜空间动力学、动作条件、未来帧预测

### c.multimodal/  多模态
- `a.clip/` — 图文对齐、对比学习、双塔结构
- `b.vlm/` — 视觉编码器 + LLM 融合、cross-attention
- `c.diffusion-basics/` — 前向加噪/反向去噪、DDPM 直觉、U-Net
- `d.audio-speech/` — Mel 特征、语音识别/合成架构概览

### d.generative-algos/  生成与对齐算法
- `a.autoregressive-sampling/` — 贪心/beam/top-k/top-p/temperature 解码对比
- `b.gan-basics/` — 生成器/判别器博弈、模式坍塌
- `c.vae/` — 编码/解码、重参数化、ELBO
- `d.rlhf-grpo/` — RLHF/PPO、奖励模型、GRPO 算法概览
- `e.moe/` — 专家路由、负载均衡、稀疏激活

---

## C. 大模型训练及精度性能  `p.practise/c.llm-training/`

### a.pretraining/  预训练
- `a.tokenization/` — BPE/WordPiece/SentencePiece 分词实验
- `b.data-pipeline/` — 数据打包、packing、mask 构造
- `c.scaling-laws/` — 模型/数据/计算 scaling law 模拟与损失预测
- `d.compute-metrics/` — FLOPs 估算、tokens/s、MFU 概念与计算

### b.parallel-strategies/  训练并行
- `a.data-parallel/` — DP/梯度聚合概念与通信量
- `b.tensor-parallel/` — TP 列/行切分、通信量分析
- `c.pipeline-parallel/` — PP、气泡、1F1B 调度
- `d.sequence-parallel/` — SP/Context 并行、长序列切分
- `e.zero-ckpt/` — ZeRO-1/2/3 显存占用建模
- `f.megatron-cfg/` — 组合并行配置、通信代价估算

### c.finetuning/  微调
- `a.full-finetune/` — 全参微调显存/算力分析
- `b.lora-qlora/` — LoRA/QLoRA 低秩、量化基座、效果
- `c.peft-methods/` — Adapter/Prefix/Prompt tuning 对比
- `d.sft-data/` — 指令数据构造、loss masking

### d.training-precision-perf/  训练精度与性能
- `a.numerics/` — FP32/FP16/BF16/FP8 表示范围与溢出模拟
- `b.mixed-precision/` — AMP、loss scaling、溢出检测
- `c.gradient-checkpoint/` — 梯度重计算 显存-算力权衡
- `d.optimizer-states/` — 优化器状态显存建模(Adam)
- `e.convergence-debug/` — 损失震荡/发散、梯度范数监控、调参
- `f.flash-attention/` — Attention 复杂度、IO-aware、显存节省
- `g.training-profiling/` — 训练性能 profiling 分析方法:msprof/PyTorch profiler 采集、计算-通信-访存分解、step 耗时拆解、瓶颈定位

### e.training-frameworks/  训练框架介绍(架构为主)
- `a.llama-factory/` — Llama-Factory 架构:统一微调框架、数据模板/数据流、WebUI、模块化的 LoRA/全参/DPO/RLHF 支持设计
- `b.ms-swift/` — ms-swift 架构:魔搭 SWIFT 训练-推理-部署一体化、插件化模型适配、多模态扩展点
- `c.mindspeed-llm/` — MindSpeed-LLM 架构:基于 Megatron-LM 的昇腾训练加速、并行策略封装、HCCL/算子适配层
- `d.mindspeed-mm/` — MindSpeed-MM 架构:多模态训练框架、模态融合管线、与 MindSpeed-LLM 的复用关系
- `e.verl/` — veRL 架构:RL 训练框架、Actor-Critic + Ray + Colocated 架构、与 SFT 训练框架解耦的接入方式
- `f.train-framework-comparison/` — 训练框架对比:定位/适用场景/并行能力/昇腾支持差异、选型决策

---

## D. 大模型推理及精度性能  `p.practise/d.llm-inference/`

### a.decoding/  解码
- `a.prefill-decode/` — prefill vs decode 阶段计算特性
- `b.kv-cache/` — KV 缓存显存建模、增长与命中
- `c.sampling-decoding/` — 采样解码对比、重复惩罚
- `d.speculative/` — 投机解码/草稿模型、加速比

### b.serving/  服务化
- `a.continuous-batching/` — 连续批处理、调度、吞吐
- `b.paged-attention/` — 分页 KV、碎片、显存利用率
- `c.chunked-prefill/` — 分块预填充、TTFT/TPOT 优化
- `d.load-metrics/` — 吞吐/延迟/并发、SLO 概念

### c.quantization/  量化
- `a.quant-basics/` — 对称/非对称量化、scale/zero-point
- `b.ptq/` — 训练后量化、校准、per-channel/per-tensor
- `c.weight-only/` — W8A16/W4A16、AWQ/GPTQ 直觉
- `d.int8-inference/` — INT8 推理精度/性能权衡
- `e.quant-error/` — 量化误差度量、敏感层分析
> 量化/反量化的**概念与策略**在本节;其**算子实现**见 `e.ascend-operator/c.common-ops/e.quant-op/`,
> **融合加速**(DeQuant+MatMul+Quant,W8A8 夹心 kernel)见 `e.ascend-operator/e.fusion-operator/b.common-fusion-ops/g.quant-fusion/`。

### d.inference-precision-perf/  推理精度与性能
- `a.precision-eval/` — 量化前后 perplexity/准确率对比
- `b.kernel-fusion/` — 算子融合、RoPE 融合、减少访存
- `c.memory-bound/` — 访存密集 vs 计算密集、roofline
- `d.benchmark-method/` — ais-bench、压测脚本、指标采集
- `e.long-context/` — 长上下文显存/延迟、YaRN/外推
- `f.inference-profiling/` — 推理性能 profiling 分析方法:逐层耗时、prefill/decode 分解、算子级 trace、访存带宽利用率、瓶颈算子定位

### e.inference-frameworks/  推理框架介绍(架构为主)
- `a.vllm/` — vLLM 架构:PagedAttention、Continuous Batching、调度器、KV cache 管理与核心数据流
- `b.vllm-ascend/` — vLLM-Ascend 架构:昇腾适配层、NPU 算子/通信映射、与 vLLM 主线的关系与演进
- `c.vllm-omni/` — vLLM-Omni 架构:多模态推理扩展、模态接入点、与 vLLM 的复用关系
- `d.infer-framework-comparison/` — 推理框架对比:功能/性能/硬件支持/部署差异、选型决策

---

## E. 昇腾算子开发  `p.practise/e.ascend-operator/`

### a.architecture/  硬件架构
- `a.ascend-hardware/` — 达芬奇架构、AI Core、Cube/Vector/AI CPU
- `b.memory-hierarchy/` — L1/L0/UB/HBM 层次、搬运效率
- `c.compute-paradigm/` — 计算范式、同步、流水

### b.development-flow/  开发流程
- `a.ascendc-basics/` — Ascend C 编程模型、tiling、API
- `b.kernel-tiling/` — tiling 切分策略、数据搬运
- `c.host-device/` — Host 入队 / Device 执行、任务下发
- `d.build-deploy/` — 编译、部署、aclnn 接口

### c.common-ops/  常见算子
- `a.matmul-op/` — 矩阵乘算子、Cube、tiling
- `b.elementwise-op/` — Elementwise/激活、Vector
- `c.reduce-op/` — Reduce/LayerNorm、分块归约
- `d.conv-op/` — Im2col/直接卷积、Cube 加速
- `e.quant-op/` — Quant/DeQuant 算子:Vector 单元 Cast+scale、per-channel、tiling

### d.tuning-debug/  调优与调试
- `a.profiling/` — msprof/profiling、瓶颈定位
- `b.profiling-analysis/` — profiling 分析方法:时间轴解读、算子耗时排序、Cube/Vector 占比、流水气泡、瓶颈定位决策树
- `c.memory-optimize/` — UB 复用、双缓冲、bank conflict
- `d.precision-debug/` — 算子精度对齐、dump 对比
- `e.performance-model/` — roofline、算力利用率、调优决策


### e.fusion-operator/  融合算子
- `a.fusion-intro/` 融合算子介绍
  - `a.why-fusion/` — 为什么要融合:减少访存、kernel launch 开销、中间结果落盘
  - `b.fusion-classification/` — 融合分类:计算密集/访存密集、elementwise/规约/线性融合
  - `c.fusion-benefits/` — 融合收益分析:访存带宽、IO、kernel launch 量化对比
  - `d.fusion-limits/` — 融合限制:寄存器/UB 占用、计算密度、可融合性判断
- `b.common-fusion-ops/` 常见融合算子
  - `a.layernorm-fusion/` — LayerNorm 融合(mean+var+normalize+scale+shift 一体)
  - `b.activation-fusion/` — 激活融合(Linear+GELU/SiLU/ReLU 等)
  - `c.attention-fusion/` — FlashAttention 融合(QK^T+softmax+PV、IO-aware)
  - `d.normalization-fusion/` — RMSNorm/BN 融合
  - `e.rope-fusion/` — RoPE 融合(旋转位置嵌入与 matmul 融合)
  - `f.loss-fusion/` — 损失融合(CE+softmax 数值稳定)
  - `g.quant-fusion/` — DeQuant+MatMul(+Quant) 融合:反量化夹心、W8A8/W4A8、FP16 中间量留 UB
- `c.fusion-development/` 融合算子开发
  - `a.ascendc-fusion/` — Ascend C 融合算子开发流程、tiling、临时变量
  - `b.fusion-tiling/` — 融合算子 tiling 策略(多算子联合切分)
  - `c.register-op/` — 算子注册、图融合、buffer 分配
  - `d.compile-deploy-fusion/` — 融合算子编译部署、aclnn 接口
- `d.fusion-tuning/` 融合算子调优
  - `a.fusion-profiling/` — 融合算子 profiling、与未融合对比
  - `b.fusion-precision/` — 融合算子精度对齐(消除中间结果的数值差异)
  - `c.fusion-tradeoff/` — 融合 vs 不融合 的性能/显存/精度权衡决策

### f.ascend-stack/  算子开发栈对照(四栈同一算子)
> 以 `vector_add` 为基准算子,横切 AscendC / PyPTO / Triton / TileLang 四种算子开发栈,
> 对照编程模型、tiling、自动生成与编译方式的异同。
- `a.ascendc/` — AscendC:kernel/tiling/双缓冲、5 段流水线同步
- `b.pypto/` — PyPTO:Python DSL 描述张量计算、自动算子生成
- `c.triton/` — Triton:@triton.jit 自动编译、block-level 编程(昇腾后端 triton-ascend)
- `d.tilelang/` — TileLang:Tile 优先 Python DSL、多后端(CUDA/HIP/AscendC)
- `e.compare/` — 四栈对照:同一算子的代码量/tiling 抽象/性能/易用性对比(待补)

---

## F. 集合通信原理与算法  `p.practise/f.collective-comm/`

### a.fundamentals/  通信基础
- `a.topology/` — 节点/卡/RoCE/拓扑、带宽与延迟
- `b.point-to-point/` — send/recv、阻塞/非阻塞
- `c.comm-primitives/` — Broadcast/Reduce/AllReduce/AllGather/ReduceScatter 语义

### b.algorithms/  集合通信算法
- `a.ring-allreduce/` — Ring AllReduce 带宽最优、分步模拟
- `b.tree-allreduce/` — Tree 算法、延迟最优
- `c.hierarchical/` — 分层(节点内+节点间)、HCCL 概念
- `d.alltoall/` — AllToAll、MoE 通信、transpose

### c.modeling/  建模与优化
- `a.bandwidth-analysis/` — 通信量/时间建模、σ-λ 模型
- `b.compute-overlap/` — 计算-通信重叠、双流
- `c.collective-ops/` — HCCL/MPI op 调用、规模扩展

### d.debug/  调试
- `a.hang-debug/` — 通信悬挂、超时、死锁定位
- `b.performance/` — 通信耗时占比、bandwidth test、调优

---

## 每个 run.sh 的统一模板

1. `set -euo pipefail` + 依赖自检(缺 numpy 等时自动 `pip install` 或给出提示)
2. **原理讲解**段:从第一性原理推导本实验要点
3. **实验步骤**段:自动生成数据 → 模拟计算 → 打印中间结果 → 可视化/表格
4. 行内中文注释解释每一步为什么这么做
5. **结论与进阶**:小白结论 + 熟手视角的调优/延伸思考
6. 不强依赖 NPU/GPU:纯 CPU + numpy/标准库即可模拟,有硬件时自动启用真实算子

---

## 规模统计

| 模块 | 子模块 | 叶实验 |
|---|---|---|
| A. AI 基础 | 4 | 18 |
| B. 模型架构与算法 | 4 | 21 |
| C. 大模型训练及精度性能 | 5 | 27 |
| D. 大模型推理及精度性能 | 5 | 23 |
| E. 昇腾算子开发 | 6 | 40 |
| F. 集合通信原理与算法 | 4 | 12 |
| **合计** | **28** | **141** |

