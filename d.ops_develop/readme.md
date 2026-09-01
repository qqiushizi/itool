# d.ops_develop — 算子开发工作流

> 场景：在客户机器上开发昇腾算子，或基于已有算子源码做改造。
> 输出形态：脚本 / 脚本文件夹（沿用 itool 约定：`字母.名称` 目录 + 叶子 `run.sh`）。
> 全程按「环境检查 → 环境搭建 → 需求分析 → 脚手架」四步推进，可单独执行任意一步。

## 总览

```
d.ops_develop/
├── a.env_check/          ① 环境检查
│   ├── a.check_env/      检查组件 + 环境变量 + 修复建议
│   └── b.download_cann/  自动下载 CANN 包(toolkit / kernel)
├── b.env_setup/          ② 环境搭建
│   ├── a.pull_vllm_ascend/ 用户指定 CANN 版本 → 拉取 vllm-ascend 镜像
│   └── b.run_container/    实例化容器
├── c.design/             ③ 算子设计需求分析
│   └── a.op_spec/        交互收集(功能/数据类型/典型shape) → 生成 op.json + spec.md
└── d.scaffold/           ④ 算子脚手架
    ├── a.msopgen/        基于 msopgen(轻量) 生成算子工程
    ├── b.ops_transformer/ 拉取 ops-transformer(完善) 源码
    └── c.torchbind/      torchbind(CPU+NPU) / vllm_ascend 接入工程
```

## 快速开始

```bash
# 1. 检查环境（会先打印环境变量，再检查 TOOLKIT/OPP/torch/torch_npu/pybind/cmake/gcc）
bash d.ops_develop/a.env_check/a.check_env/run.sh

# 2. 若缺少 CANN，下载（版本可配）
CANN_VERSION=8.1.RC1 CHIP=910b bash d.ops_develop/a.env_check/b.download_cann/run.sh

# 3. 拉镜像 + 起容器
CANN_VERSION=8.1.rc1 CHIP=910b bash d.ops_develop/b.env_setup/a.pull_vllm_ascend/run.sh
bash d.ops_develop/b.env_setup/b.run_container/run.sh cann-910b:8.1.rc1 my_ascend

# 4. 需求分析 → 生成 op.json
bash d.ops_develop/c.design/a.op_spec/run.sh

# 5. 生成工程
bash d.ops_develop/d.scaffold/a.msopgen/run.sh op.json
bash d.ops_develop/d.scaffold/b.ops_transformer/run.sh
bash d.ops_develop/d.scaffold/c.torchbind/run.sh AddCustom
```

## 说明

- 所有脚本尽量少依赖、纯 bash + 标准命令，面向昇腾客户机（Linux）。
- 涉及网络下载的脚本（CANN 包、镜像、git clone）默认使用公开源，均可通过环境变量覆盖为内部镜像。
- 脚本兼容 bash 3.2（不使用 bash4 语法），避免在低版本 bash 上踩坑。
