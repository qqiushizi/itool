# h.op_agent — OpenCode / CANNBot Agent 安装与更新

> 算子开发工作流第 8 步：安装 OpenCode 绿色运行时，并接入最新官方 CANNBot skills、agents。

## 目录

```text
h.op_agent/
├── a.install_opencode/         安装 opencode 绿色版（二进制 + 配置）
│   ├── opencode.tar.gz
│   └── run.sh
└── b.install_cannbot_skill/    安装最新 CANNBot skills/agents
    └── run.sh
```

## 使用

```bash
# 1. 解压并启动 opencode（便携模式，不修改系统）
bash d.ops_develop/h.op_agent/a.install_opencode/run.sh

# 2. 安装最新 CANNBot skills/agents
bash d.ops_develop/h.op_agent/b.install_cannbot_skill/run.sh
```

### `b.install_cannbot_skill` 行为说明

**目标只有一个：安装官方最新版。失败即报错并说明原因，绝不自动回退到旧快照**（避免误以为装到的是最新版）。

执行顺序：

1. 本机有 `install-helper` → 直接调用它安装/更新
2. 否则有 `npx`（Node 18+）→ 用 `npx @cannbot-ai/install-helper`
3. 否则 → 用官方 `install.sh` 引导安装 helper

上述任一失败 → 打印明确的失败原因（网络问题 / 缺 Node / 缺 helper），并以非零退出码结束，**不改动已有配置**。

## 常用环境变量

```bash
CANNBOT_TOOL=opencode          # 目标 agent 工具，默认 opencode
CANNBOT_LEVEL=global           # project 或 global，默认 global
CANNBOT_USE_BUNDLE=1           # 显式转为离线快照安装（见下，需手动触发）
CANNBOT_ARGS="--all"           # 传给 install-helper 的额外参数
```

### 关于 CANNBOT_USE_BUNDLE（离线快照，非默认）

默认情况**不会**走离线快照。只有在你明确设置 `CANNBOT_USE_BUNDLE=1` 时才会安装仓库内置的旧快照，且会明确提示"这是 v1.1.0 旧版，非最新版"：

```bash
# 显式使用绿色包离线快照（旧版，仅适用于无法联网的环境）
CANNBOT_USE_BUNDLE=1 bash d.ops_develop/h.op_agent/b.install_cannbot_skill/run.sh
```

## 官方来源

CANNBot 官方安装 / 更新方式：

```bash
# 官方一键安装脚本
curl -fsSL https://raw.gitcode.com/cann/cannbot-skills/raw/master/install.sh | bash

# 用 npm 安装助手安装
npx -y @cannbot-ai/install-helper install --all --tool opencode --level global --yes

# 用 npm 安装助手更新到最新
npx -y @cannbot-ai/install-helper update --tool opencode --level global --yes
```

安装后的 skills/agents 位于：

```text
${XDG_CONFIG_HOME:-$HOME/.config}/opencode
```

绿色包内置内容（离线快照）：

```text
skills : 74 个
agents : 18 个
版本   : CANNBot 1.1.0
```

> 注意：绿色包是 2026-07-07 打包的 1.1.0 快照，仅作手动离线兜底。在线环境请使用 `install-helper`/`npx` 获取官方最新版。