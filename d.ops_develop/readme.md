# d.ops_develop — 算子开发工作流

> 场景：在客户机器上开发昇腾算子，或基于已有算子源码做改造。
> 输出形态：脚本 / 脚本文件夹（沿用 itool 约定：`字母.名称` 目录 + 叶子 `run.sh`）。
> 所有脚本的输入/输出默认归档在 `d.ops_develop/workspace/` 下。

## 总览

```
d.ops_develop/
├── a.llm_config/                     ① 大模型配置管理
│   ├── run.sh                 新建/修改/删除/测试模型配置，供后续 LLM 功能选择
│   └── readme.md              功能说明
├── b.image_container/                 ② 镜像拉取 + 容器实例化
│   ├── run.sh                 查询官方 tag 作参考 / 手动输入 tag 镜像 / 检查或拉取
│   └── readme.md              功能说明
├── c.env_check/                       ③ 容器内环境检查
│   ├── run.sh                 芯片型号识别 + 软件版本匹配矩阵，只读输出建议
│   └── readme.md              功能说明
├── d.install_cann/                    ④ 按需安装/补装 CANN toolkit
│   ├── a.cann-9.1.0/run.sh    9.1.0 (推荐)
│   ├── b.cann-9.0.0/run.sh    9.0.0
│   ├── c.cann-8.2.RC1/run.sh  8.2.RC1 (旧芯片兼容)
│   ├── d.cann-8.1.RC1/run.sh  8.1.RC1 (旧芯片兼容)
│   └── readme.md              版本说明 + 容器内安装要求
├── e.op_design/                       ⑤ 算子设计需求分析 → op.json + op_spec.md
│   ├── run.sh                 选择模型配置 / 手动填写 → workspace/op_design_<算子名>/
│   └── readme.md              功能说明
├── f.op_build/                        ⑥ 算子工程构建
│   ├── run.sh                 读取 workspace 下 op.json，用 msopgen 生成 AscendC 算子工程
│   └── readme.md              功能说明
├── g.op_fix/                          ⑦ 大模型辅助修改算子工程代码
│   ├── run.sh                 选择工程 + 模型配置，多轮对话修改代码
│   └── readme.md              功能说明
└── h.op_agent/                        ⑧ OpenCode/CANNBot Agent 安装
    ├── readme.md              说明
    ├── a.install_opencode/    安装 opencode 绿色版
    └── b.install_cannbot_skill/ 接入 CANNBot skills/agents
```

## 快速开始

```bash
# ① 首次使用：先创建大模型配置（后续每个 LLM 功能可选配置）
bash d.ops_develop/a.llm_config/run.sh

# ② 拉镜像 + 建容器
bash d.ops_develop/b.image_container/run.sh

# 进入刚创建的容器
docker exec -it asc_dev bash

# ③ 容器内环境检查
cd /workspace/itool
bash d.ops_develop/c.env_check/run.sh

# ④ 如果环境检查报告建议“安装/更换 CANN”，再执行（按需，可跳过）
bash d.ops_develop/d.install_cann/a.cann-9.1.0/run.sh

# ⑤ 需求分析：选择模型配置或手动填写 → op.json / op_spec.md
bash d.ops_develop/e.op_design/run.sh

# ⑥ 生成算子工程
bash d.ops_develop/f.op_build/run.sh

# ⑦ 大模型辅助修改算子工程
bash d.ops_develop/g.op_fix/run.sh

# ⑧ 安装 OpenCode + CANNBot skills（深度算子开发 agent）
bash d.ops_develop/h.op_agent/a.install_opencode/run.sh
bash d.ops_develop/h.op_agent/b.install_cannbot_skill/run.sh
```

## 说明

- 所有脚本尽量少依赖、纯 bash + 标准命令，面向昇腾客户机（Linux），兼容 bash 3.2。
- `a.llm_config` 生成和选择大模型配置，后续 `e.op_design` / `g.op_fix` 只选择，不重复输入 API。
- `workspace/` 默认被 Git 忽略，客户本机的生成物不会提交到 GitHub。
- `b.image_container` 是环境工作流起点；把宿主机工作目录挂载到容器 `/workspace`，并把当前 itool 仓库挂载到 `/workspace/itool`。
- `c.env_check` 只做只读检查，不修复、不安装；缺 CANN 时只提示进入 `d.install_cann`。
- `d.install_cann` 默认只能在容器内执行，只安装 toolkit，不安装 kernels/ops、合一包和驱动。
- `f.op_build` 固定用 AscendC/C++ 模板生成算子工程。
- `g.op_fix` 按算子工程保存上下文：`workspace/op_build_<算子名>/.itool/op_fix_history.jsonl`。
- `h.op_agent` 使用 repo 内自带 opencode 绿色包；`b.install_cannbot_skill` 将 CANNBot 74 个 skills、18 个 agents 接入用户配置目录。
