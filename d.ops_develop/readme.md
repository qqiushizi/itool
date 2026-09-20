# d.ops_develop — 算子开发工作流

> 场景：在客户机器上开发昇腾算子，或基于已有算子源码做改造。
> 输出形态：脚本目录 `字母.名称` + 叶子 `run.sh`。
> 生成物都归档在 `d.ops_develop/workspace/` 下。

## 总览

```
d.ops_develop/
├── a.llm_config/                     ① 大模型配置管理（可选，直接 LLM 扩展用）
│   ├── run.sh
│   └── readme.md
├── b.image_container/                 ② 镜像拉取 + 容器实例化
│   ├── run.sh                 输入镜像地址 / 拉取 / 生成 start_container.sh
│   └── readme.md
├── c.env_check/                       ③ 容器内环境检查（只读）
│   ├── run.sh                 芯片型号识别 + 版本兼容矩阵 + 建议
│   └── readme.md
├── d.install_cann/                    ④ 按需安装/补装 CANN toolkit
│   ├── a.cann-9.1.0/run.sh
│   ├── b.cann-9.0.0/run.sh
│   ├── c.cann-8.2.RC1/run.sh
│   ├── d.cann-8.1.RC1/run.sh
│   └── readme.md
├── e.install_opencode/                ⑤ 安装 OpenCode + CANNBot skills
│   ├── a.install_opencode/
│   ├── b.install_cannbot_skill/
│   └── readme.md
└── f.op_agent/                        ⑥ 算子 Agent 开发/测试/报告
    ├── run.sh                 打开 OpenCode 开发 → 生成测试 → 测试 → 报告
    └── readme.md
```

## 快速开始

```bash
# ① 拉镜像 + 建容器
bash d.ops_develop/b.image_container/run.sh

# 进入容器
docker exec -it asc_dev bash
cd /workspace/itool

# ② 容器内环境检查（只读，给结论和建议）
bash d.ops_develop/c.env_check/run.sh

# ③ 若检查报告提示需要安装/更换 CANN，再执行（可跳过）
bash d.ops_develop/d.install_cann/a.cann-9.1.0/run.sh

# ④ 安装 OpenCode 运行时 + CANNBot skills/agents
bash d.ops_develop/e.install_opencode/a.install_opencode/run.sh
bash d.ops_develop/e.install_opencode/b.install_cannbot_skill/run.sh

# ⑤ 打开算子开发 Agent（开发 → 生成测试 → 测试 → 报告）
bash d.ops_develop/f.op_agent/run.sh
```

`a.llm_config` 为可选项，仅当后面还有需要直接调用大模型 API 的扩展功能时才需要用；`f.op_agent` 使用 OpenCode，不依赖它。

## 说明

- 所有脚本尽量少依赖、纯 bash + 标准命令，面向昇腾客户机（Linux），兼容 bash 3.2。
- `workspace/` 默认被 Git 忽略，客户机器上的生成物不会提交到 GitHub。
- `c.env_check` 只做只读检查，不修复、不安装。
- `d.install_cann` 默认只能在容器内执行，只安装 toolkit，不安装 kernels/ops、合一包和驱动。
- `e.install_opencode` 使用仓库自带 opencode 绿色包，优先官方 CANNBot 安装，离线时回退内置 skills。
- `f.op_agent` 在 `workspace/op_build_<算子名>/` 下开发，测试报告归档到 `workspace/op_build_<算子名>/report/`。
