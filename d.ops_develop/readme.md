# d.ops_develop — 算子开发工作流

> 场景：在客户机器上开发昇腾算子，或基于已有算子源码做改造。
> 输出形态：脚本 / 脚本文件夹（沿用 itool 约定：`字母.名称` 目录 + 叶子 `run.sh`）。
> 全程按「镜像/容器 ① → 环境检查 ② → 按需安装 CANN ③ → 需求分析 ④ → 脚手架 ⑤」推进；第 ②③ 步在容器内执行。

## 总览

```
d.ops_develop/
├── a.image_container/                 ① 镜像拉取 + 容器实例化
│   ├── run.sh                 选择镜像 tag / docker pull / 生成 start_container.sh
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
│   └── a.op_spec/run.sh       交互询问，输出 op.json / op_spec.md
└── e.op_scaffold/                     ⑤ 算子脚手架
    ├── a.msopgen/run.sh       msopgen 生成 AscendC 工程
    ├── b.ops_transformer/run.sh 拉取 ops-transformer 源码
    └── c.torchbind/run.sh     torchbind(CPU+NPU) 接入工程
```

## 快速开始

```bash
# ① 拉镜像 + 建容器
bash d.ops_develop/a.image_container/run.sh

# 进入刚创建的容器
docker exec -it asc_dev bash

# ② 容器内环境检查
bash d.ops_develop/b.env_check/run.sh

# ③ 如果环境检查报告建议“安装/更换 CANN”，再执行（按需，可跳过）
bash d.ops_develop/c.install_cann/a.cann-9.1.0/run.sh

# ④ 需求分析 → 生成 op.json / op_spec.md
bash d.ops_develop/d.op_design/a.op_spec/run.sh

# ⑤ 生成工程
bash d.ops_develop/e.op_scaffold/a.msopgen/run.sh op.json   # msopgen 轻量
bash d.ops_develop/e.op_scaffold/b.ops_transformer/run.sh   # ops-transformer 完善
bash d.ops_develop/e.op_scaffold/c.torchbind/run.sh AddCustom
```

## 说明

- 所有脚本尽量少依赖、纯 bash + 标准命令，面向昇腾客户机（Linux），兼容 bash 3.2。
- `a.image_container` 是工作流起点：拉镜像、建容器，并把宿主机工作目录挂载到容器 `/workspace`。
- `b.env_check` 只做只读检查，不修复、不安装；缺 CANN 时只提示进入 `c.install_cann`。
- `c.install_cann` 默认只能在容器内执行，默认安装到 `/workspace/Ascend/ascend-toolkit-<version>`，容器重建后仍保留。
- `c.install_cann` 只安装 toolkit，不安装 kernels/ops、合一包和驱动。
- `a.image_container` 会在工作目录生成 `start_container.sh`，客户可自行修改后反复使用。
