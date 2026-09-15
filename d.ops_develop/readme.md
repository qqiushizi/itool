# d.ops_develop — 算子开发工作流

> 场景：在客户机器上开发昇腾算子，或基于已有算子源码做改造。
> 输出形态：脚本 / 脚本文件夹（沿用 itool 约定：`字母.名称` 目录 + 叶子 `run.sh`）。
> 全程按「镜像/容器 ① → 环境检查 ② → 按需安装 CANN ③ → 需求分析 ④ → 脚手架 ⑤」推进；第 ②③ 步在容器内执行。

## 总览

```
d.ops_develop/
├── a.image_container/                 ① 镜像拉取 + 容器实例化
│   ├── run.sh                 查询官方 tag 作参考 / 手动输入 tag 镜像 / 检查或拉取
│   └── readme.md              功能说明
├── b.env_check/                       ② 容器内环境检查
│   ├── run.sh                 芯片型号识别 + 软件版本匹配矩阵，只读输出建议
│   └── readme.md              功能说明
├── c.install_cann/                    ③ 按需安装/补装 CANN toolkit
│   ├── a.cann-9.1.0/run.sh    9.1.0 (推荐)
│   ├── b.cann-9.0.0/run.sh    9.0.0
│   ├── c.cann-8.2.RC1/run.sh  8.2.RC1 (旧芯片兼容)
│   ├── d.cann-8.1.RC1/run.sh  8.1.RC1 (旧芯片兼容)
│   └── readme.md              版本说明 + 容器内安装要求
├── d.op_design/                       ④ 算子设计需求分析 → op.json + op_spec.md
│   ├── run.sh                 本地/外部 LLM / 手动填写 → workspace/op_design_<算子名>/
│   └── readme.md              功能说明
└── e.op_build/                        ⑤ 算子工程构建
    ├── run.sh                 读取 workspace 下 op.json，用 msopgen 生成算子工程
    └── readme.md              功能说明
```

## 快速开始

```bash
# ① 拉镜像 + 建容器
bash d.ops_develop/a.image_container/run.sh

# 进入刚创建的容器
docker exec -it asc_dev bash

# ② 容器内环境检查
cd /workspace/itool
bash d.ops_develop/b.env_check/run.sh

# ③ 如果环境检查报告建议“安装/更换 CANN”，再执行（按需，可跳过）
bash d.ops_develop/c.install_cann/a.cann-9.1.0/run.sh

# ④ 需求分析(本地/外部 LLM / 手动填写) → 生成 op.json / op_spec.md
bash d.ops_develop/d.op_design/run.sh

# ⑤ 生成算子工程
bash d.ops_develop/e.op_build/run.sh   # 默认自动读取 workspace 下的 op.json
```

## 说明

- 所有脚本尽量少依赖、纯 bash + 标准命令，面向昇腾客户机（Linux），兼容 bash 3.2。
- `a.image_container` 是工作流起点：查询 quay.io 官方仓库（vllm-ascend / cann）tag 作为参考，以用户手动输入的 tag/完整镜像为准；随后检查/拉取镜像，通过 A/B 选项配置容器参数，把宿主机工作目录挂载到容器 `/workspace`，并把当前 itool 仓库挂载到 `/workspace/itool`。
- d/e 功能的输入输出统一在 `d.ops_develop/workspace/` 下归档；d 生成 `op_design_<算子名>/`，e 生成 `op_build_<算子名>/`。
- `a.image_container` 容器参数均可用环境变量预设（`NAME`、`WORK_DIR`、`SHM_SIZE`、`NET_MODE`、`PRIVILEGED`），未预设时用选项引导，不要求用户手输复杂参数。
- `start_container.sh` 生成后不会自动执行，会先询问 `y/N`；确认后才启动容器。
- `b.env_check` 只做只读检查，不修复、不安装；缺 CANN 时只提示进入 `c.install_cann`。
- `c.install_cann` 默认只能在容器内执行，默认安装到 `/workspace/Ascend/ascend-toolkit-<version>`，容器重建后仍保留。
- `c.install_cann` 只安装 toolkit，不安装 kernels/ops、合一包和驱动。
- `a.image_container` 会在 `~/ascend_ops_workspace`（可用 `WORK_DIR` 覆盖）生成当前机器专用的 `start_container.sh`，同一台机器可反复使用；换机器应重新运行本步骤。
