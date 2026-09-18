# h.op_agent — OpenCode / CANNBot Agent 安装

> 算子开发工作流第 8 步：为 `g.op_fix` 之后的深度算子开发安装 OpenCode 绿色运行时，
> 并把绿色包内预置的 CANNBot skills / agents 接入到用户配置目录。

## 目录

```text
h.op_agent/
├── a.install_opencode/         安装 opencode 绿色版（二进制 + 配置）
│   ├── opencode.tar.gz
│   └── run.sh
└── b.install_cannbot_skill/    接入 CANNBot skills / agents
    └── run.sh
```

## 使用

```bash
# 1. 解压并启动 opencode（便携模式，不修改系统）
bash d.ops_develop/h.op_agent/a.install_opencode/run.sh

# 2. 安装 opencode + CANNBot skills/agents 到用户配置目录
bash d.ops_develop/h.op_agent/b.install_cannbot_skill/run.sh
```

如果想只安装 skills/agents，不装 opencode 命令：

```bash
SKILL_ONLY=1 bash d.ops_develop/h.op_agent/b.install_cannbot_skill/run.sh
```

非 root 用户会默认安装到：

```text
$HOME/.local/bin/opencode
```

CANNBot 配置使用户配置目录：

```text
${XDG_CONFIG_HOME:-$HOME/.config}/opencode
```

绿色包内已包含：

```text
skills : 74 个
agents : 18 个
版本   : CANNBot 1.1.0
```

## 说明

- `a.install_opencode` 是便携运行；如果需要让系统全局出现 `opencode` 命令，可执行 `b.install_cannbot_skill`。
- `opencode.tar.gz` 为绿色自包含包，无需 npm / Node，不污染已有 opencode 安装。
