# d.ops_develop — 算子开发工作流

> 场景：在客户机器上开发昇腾算子，或基于已有算子源码做改造。
> 输出形态：脚本 / 脚本文件夹（沿用 itool 约定：`字母.名称` 目录 + 叶子 `run.sh`）。
> 全程按「环境检查 ① → CANN 安装 ② → 镜像/容器 ③ → 需求分析 ④ → 脚手架 ⑤」推进，可单独执行任意一步。

## 总览

```
d.ops_develop/
├── a.env_check/                       环境检查(宿主机 / 容器通用, 只读)
│   ├── run.sh                 ① 环境检查(芯片型号识别 + 软件版本匹配矩阵)
│   └── readme.md              功能说明
├── b.env_setup/                       CANN toolkit 安装(下载 + 安装合并)
│   └── a.install_cann/
│       ├── a.cann-9.1.0/      9.1.0 (推荐)
│       ├── b.cann-9.0.0/      9.0.0
│       ├── c.cann-8.2.RC1/    8.2.RC1 (旧芯片兼容)
│       ├── d.cann-8.1.RC1/    8.1.RC1 (旧芯片兼容)
│       └── readme.md
├── c.container/                        ③ 镜像拉取 + 容器实例化(合并)
│   ├── run.sh                 镜像 tag 可视化选择 / docker pull / 生成起容器脚本
│   └── readme.md              功能说明
├── d.design/                           ④ 算子设计需求分析 → op.json + op_spec.md
└── e.scaffold/                         ⑤ 算子脚手架(msopgen / ops-transformer / torchbind)
```

## 快速开始

```bash
# ① 环境检查(只读: 识别 910/950/310 芯片, 检查 Python/CANN/torch/torch_npu 版本匹配)
bash d.ops_develop/a.env_check/run.sh

# ② CANN toolkit 安装(推荐 9.1.0, 下载+安装一步完成, 仅 toolkit)
bash d.ops_develop/b.env_setup/a.install_cann/a.cann-9.1.0/run.sh

# ③ 镜像拉取 + 容器实例化(合并)
bash d.ops_develop/c.container/run.sh

# 进入容器后建议再次运行同一个环境检查
#   docker exec -it asc_dev bash
#   bash d.ops_develop/a.env_check/run.sh

# ④ 需求分析 → 生成 op.json / op_spec.md
bash d.ops_develop/d.design/a.op_spec/run.sh

# ⑤ 生成工程
bash d.ops_develop/e.scaffold/a.msopgen/run.sh op.json   # msopgen 轻量
bash d.ops_develop/e.scaffold/b.ops_transformer/run.sh   # ops-transformer 完善
bash d.ops_develop/e.scaffold/c.torchbind/run.sh AddCustom
```

## 说明

- 所有脚本尽量少依赖、纯 bash + 标准命令，面向昇腾客户机（Linux），兼容 bash 3.2。
- `a.env_check` 合并了原来的宿主机/容器检查，只在当前环境内做只读检查。
- `a.env_check` 只保留环境版本匹配所需的项目：芯片型号、Python、CANN、torch、torch_npu，并输出兼容性矩阵结论；不做修复/安装，不扩展额外检查项。
- `b.env_setup/a.install_cann` 将下载与安装合并，只安装 toolkit，不安装 kernels/ops、合一包和驱动。
- `c.container` 将镜像拉取与容器实例化合并：先选 tag / 拉镜像，再生成可编辑 `start_container.sh` 并启动容器。
- `c.container` 会在工作目录生成 `start_container.sh`，客户可自行修改后反复使用。
