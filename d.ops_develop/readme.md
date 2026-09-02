# d.ops_develop — 算子开发工作流

> 场景：在客户机器上开发昇腾算子，或基于已有算子源码做改造。
> 输出形态：脚本 / 脚本文件夹（沿用 itool 约定：`字母.名称` 目录 + 叶子 `run.sh`）。
> 全程按「环境准备 ①→⑤ → 需求分析 ⑥ → 脚手架 ⑦」推进，可单独执行任意一步。

## 总览

```
d.ops_develop/
├── a.env_check/                       环境准备(宿主机)
│   ├── a.check_cann/       ① 服务器 CANN 检查(安装目录/版本/驱动/激活)
│   ├── b.download_cann/      下载 CANN 包(toolkit / kernels / 合一包)
│   └── c.install_cann/     ② CANN 安装(安装方式/位置/source 激活)
├── b.env_setup/                       容器化开发环境
│   ├── a.pull_image/       ③ 镜像拉取(quay.io 可视化选 tag)
│   ├── b.run_container/    ④ 容器实例化(交互输入 + 生成可编辑起容器脚本)
│   └── c.check_in_container/⑤ 进容器检查软件包 → 确认可开始算子开发
├── c.design/               ⑥ 算子设计需求分析 → op.json + op_spec.md
└── d.scaffold/             ⑦ 算子脚手架(msopgen / ops-transformer / torchbind)
```

## 快速开始

```bash
# ① 服务器 CANN 检查(汇总: 安装目录/版本/驱动/激活状态)
bash d.ops_develop/a.env_check/a.check_cann/run.sh

# (可选) 下载 CANN 包
CANN_VERSION=8.1.RC1 CHIP=910b bash d.ops_develop/a.env_check/b.download_cann/run.sh
MODE=combined  CANN_VERSION=8.1.RC1 bash d.ops_develop/a.env_check/b.download_cann/run.sh   # 合一包

# ② 交互式安装 CANN(选择方式/位置/自动 source 激活)
bash d.ops_develop/a.env_check/c.install_cann/run.sh

# ③ 拉镜像(可视化选择 tag)
bash d.ops_develop/b.env_setup/a.pull_image/run.sh

# ④ 实例化容器(输入容器名等, 生成 start_container.sh 可自行修改)
bash d.ops_develop/b.env_setup/b.run_container/run.sh

# ⑤ 进容器检查软件包, 确认可开始算子开发
bash d.ops_develop/b.env_setup/c.check_in_container/run.sh asc_dev
```

## 说明

- 所有脚本尽量少依赖、纯 bash + 标准命令，面向昇腾客户机（Linux），兼容 bash 3.2。
- 涉及网络下载/查询的脚本（CANN 包、quay 镜像）默认使用公开源，均可通过环境变量覆盖为内部镜像。
- `①` 为纯检查脚本，不修改系统；`②` 安装脚本会执行 `.run` 安装包（非 root 自动加 `sudo`）。
- `④` 会在工作目录生成 `start_container.sh`，客户可自行修改后反复使用。
