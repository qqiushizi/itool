# d.ops_develop — 算子开发工作流

> 场景：在客户机器上开发昇腾算子，或基于已有算子源码做改造。
> 输出形态：脚本 / 脚本文件夹（沿用 itool 约定：`字母.名称` 目录 + 叶子 `run.sh`）。
> 全程按「环境准备 ①→④ → 需求分析 ⑤ → 脚手架 ⑥」推进，可单独执行任意一步。

## 总览

```
d.ops_develop/
├── a.env_check/                       环境检查(宿主机 / 容器通用, 只读)
│   ├── run.sh   ① 环境检查(芯片型号识别 + 软件版本匹配矩阵)
│   └── readme.md      功能说明
├── b.env_setup/                       环境搭建(下载 / 安装 / 镜像 / 容器)
│   ├── a.install_cann/     ② CANN toolkit 安装(下载+安装合并, 仅 toolkit)
│   │   ├── a.cann-9.1.0/    9.1.0 (推荐)
│   │   ├── b.cann-9.0.0/    9.0.0
│   │   ├── c.cann-8.2.RC1/  8.2.RC1 (旧芯片兼容)
│   │   ├── d.cann-8.1.RC1/  8.1.RC1 (旧芯片兼容)
│   │   └── readme.md        版本说明
│   ├── b.pull_image/       ③ 镜像拉取(quay.io 可视化选 tag)
│   └── c.run_container/    ④ 容器实例化(交互输入 + 生成可编辑起容器脚本)
├── c.design/               ⑤ 算子设计需求分析 → op.json + op_spec.md
└── d.scaffold/             ⑥ 算子脚手架(msopgen / ops-transformer / torchbind)
```

## 快速开始

```bash
# ① 环境检查(只读: 识别 910/950/310 芯片, 检查 Python/CANN/torch/torch_npu 版本匹配)
bash d.ops_develop/a.env_check/run.sh

# ② CANN toolkit 安装(选择版本, 下载+安装一步完成, 仅 toolkit)
bash d.ops_develop/b.env_setup/a.install_cann/a.cann-9.1.0/run.sh

# ③ 拉镜像(可视化选择 tag)
bash d.ops_develop/b.env_setup/b.pull_image/run.sh

# ④ 实例化容器(输入容器名等, 生成 start_container.sh 可自行修改)
bash d.ops_develop/b.env_setup/c.run_container/run.sh

# 进入容器后建议再次运行同一个环境检查
#   docker exec -it asc_dev bash
#   bash d.ops_develop/a.env_check/run.sh

# ⑤ 需求分析 → 生成 op.json / op_spec.md
bash d.ops_develop/c.design/a.op_spec/run.sh

# ⑥ 生成工程
bash d.ops_develop/d.scaffold/a.msopgen/run.sh op.json   # msopgen 轻量
bash d.ops_develop/d.scaffold/b.ops_transformer/run.sh   # ops-transformer 完善
bash d.ops_develop/d.scaffold/c.torchbind/run.sh AddCustom
```

## 说明

- 所有脚本尽量少依赖、纯 bash + 标准命令，面向昇腾客户机（Linux），兼容 bash 3.2。
- `a.env_check` 合并了原来的「① 服务器 CANN 检查」与「⑤ 进容器检查软件包」，只在一个当前环境内做只读检查；宿主机和容器通用。
- `a.env_check` 只保留环境版本匹配所需的项目：芯片型号、Python、CANN、torch、torch_npu，并输出兼容性矩阵结论；不做修复/安装，不扩展额外检查项。
- 涉及网络下载/查询的脚本（CANN 包、quay 镜像）默认使用公开源，均可通过环境变量覆盖为内部镜像。
- `a.env_check` 只负责检查(只读)；`b.env_setup`(下载/②/③/④) 负责实际动作：下载、`.run` 安装(非 root 自动 `sudo`)、拉镜像、起容器。
- `④` 会在工作目录生成 `start_container.sh`，客户可自行修改后反复使用。
