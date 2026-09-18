# h.op_agent — OpenCode / CANNBot Agent 安装

> 算子开发工作流第 8 步：安装 OpenCode 绿色运行时，并从官方 CANNBot 仓库拉取/接入 skills、agents。

## 目录

```text
h.op_agent/
├── a.install_opencode/         安装 opencode 绿色版（二进制 + 配置）
│   ├── opencode.tar.gz
│   └── run.sh
└── b.install_cannbot_skill/    安装 CANNBot skills/agents
    └── run.sh
```

## 使用

```bash
# 1. 解压并启动 opencode（便携模式，不修改系统）
bash d.ops_develop/h.op_agent/a.install_opencode/run.sh

# 2. 安装 CANNBot skills/agents
bash d.ops_develop/h.op_agent/b.install_cannbot_skill/run.sh
```

`b.install_cannbot_skill` 默认优先走官方 CANNBot 安装助手：

- 若本机有 `install-helper`，直接调用
- 否则用 `npx @cannbot-ai/install-helper`
- 否则走官方 `install.sh`

网络不可用或官方安装失败时，自动回退到 `a.install_opencode/opencode.tar.gz` 内置的 CANNBot skills/agents。

## 常用环境变量

```bash
CANNBOT_TOOL=opencode          # 目标 agent 工具，默认 opencode
CANNBOT_LEVEL=global           # project 或 global，默认 global
CANNBOT_USE_BUNDLE=1           # 强制使用本地绿色包，不走网络
CANNBOT_ARGS="--all"           # 传给 install-helper 的额外参数
```

示例：

```bash
# 强制使用绿色包内置 skills，不联网
CANNBOT_USE_BUNDLE=1 bash d.ops_develop/h.op_agent/b.install_cannbot_skill/run.sh

# 只安装到项目目录
CANNBOT_LEVEL=project bash d.ops_develop/h.op_agent/b.install_cannbot_skill/run.sh
```

## 官方来源

CANNBot 官方安装方式：

```bash
# 官方一键安装脚本
curl -fsSL https://raw.gitcode.com/cann/cannbot-skills/raw/master/install.sh | bash

# 或者 npm 安装助手
npx -y @cannbot-ai/install-helper install --all --tool opencode --level global --yes
```

安装后的 skills/agents 位于：

```text
${XDG_CONFIG_HOME:-$HOME/.config}/opencode
```

绿色包内置内容：

```text
skills : 74 个
agents : 18 个
版本   : CANNBot 1.1.0
```
