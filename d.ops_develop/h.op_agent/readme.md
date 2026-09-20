# h.op_agent — OpenCode / CANNBot Agent 安装与更新

> 算子开发工作流第 8 步：安装 OpenCode 绿色运行时，并接入 / 更新官方 CANNBot skills、agents。

## 目录

```text
h.op_agent/
├── a.install_opencode/         安装 opencode 绿色版（二进制 + 配置）
│   ├── opencode.tar.gz
│   └── run.sh
└── b.install_cannbot_skill/    检查 / 更新 CANNBot skills/agents
    └── run.sh
```

## 使用

```bash
# 1. 解压并启动 opencode（便携模式，不修改系统）
bash d.ops_develop/h.op_agent/a.install_opencode/run.sh

# 2. 检查 / 更新 CANNBot skills/agents
bash d.ops_develop/h.op_agent/b.install_cannbot_skill/run.sh
```

`b.install_cannbot_skill` 的定位是 **检查当前版本 + 有新版则更新**：

- 首次使用（尚未安装）→ 自动先走完整安装，再进入更新检查
- 已安装 → 读取当前 `cannbot-manifest.json` 版本，调用官方 `install-helper update` 检查并升级到最新
- 网络不可用 / 无 `install-helper` → 提示无法在线检查，可 `CANNBOT_USE_BUNDLE=1` 回退到绿色包离线安装

> 说明：CANNBot 官方为滚动更新（按日期迭代，无固定版本号文件），"检查是否有新版"由官方 `install-helper update` 完成，itool 不自行解析版本号。

## 常用环境变量

```bash
CANNBOT_TOOL=opencode          # 目标 agent 工具，默认 opencode
CANNBOT_LEVEL=global           # project 或 global，默认 global
CANNBOT_USE_BUNDLE=1           # 强制使用本地绿色包，不走网络（离线兜底）
CANNBOT_ARGS="--all"           # 传给 install-helper 的额外参数
```

示例：

```bash
# 检查并更新到官方最新（默认行为）
bash d.ops_develop/h.op_agent/b.install_cannbot_skill/run.sh

# 强制使用绿色包离线安装（不走网络）
CANNBOT_USE_BUNDLE=1 bash d.ops_develop/h.op_agent/b.install_cannbot_skill/run.sh

# 只安装到项目目录
CANNBOT_LEVEL=project bash d.ops_develop/h.op_agent/b.install_cannbot_skill/run.sh
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

绿色包内置内容（离线兜底快照）：

```text
skills : 74 个
agents : 18 个
版本   : CANNBot 1.1.0
```

> 注意：绿色包是 2026-07-07 打包的 1.1.0 快照，仅作离线兜底。在线环境请优先使用 `install-helper update` 获取官方最新版。