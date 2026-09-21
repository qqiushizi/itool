# h.op_agent — OpenCode 运行时与 Skill 管理

> 算子开发工作流第 8 步：安装 OpenCode 绿色运行时，并提供离线的 Skill 仓库（选装）与联网获取最新 Skill 的能力。

## 背景与定位

itool 的算子开发依赖 opencode（提供工具调用的 agent 运行时）+ 昇腾官方 skill（提供领域知识）。

官方 skill 的分发体系分三层：

```
Ascend/agent-skills（官方总仓/插件市场，覆盖全模块）
   └── official/CANNBot（submodule，算子开发 skill+agent，指向 cann/cannbot-skills）
```

- **Skills** 是"能力"，按 `description` 语义匹配、按需加载；**不是** agent 运行时。
- 本目录提供一个**离线 skill 仓库**（应对昇腾客户机连不上 gitcode 的情况）和一个**联网更新入口**。

## 目录

```text
h.op_agent/
├── a.install_opencode/         安装 opencode 绿色版（二进制 + 配置，即开即用）
│   ├── opencode.tar.gz         = opencode + CANNBot 1.1.0（74 skill + 18 agent 快照）
│   └── run.sh
├── b.skill_store/              离线 Skill 仓库（选装，不依赖网络）
│   ├── skills.tar.gz           = 官方总仓 304 个 skill（已带模块前缀 + 规范命名）
│   └── run.sh
└── c.update_skill/             联网获取/更新官方总仓 Skill（需 Node 18+ 与网络）
    └── run.sh
```

## 使用

### 1. 安装 opencode 运行时（离线，即开即用）

```bash
bash d.ops_develop/h.op_agent/a.install_opencode/run.sh
```

解压后即拥有 opencode + CANNBot 1.1.0 基础能力（74 skill + 18 agent），无需联网。

### 2. 离线选装更多 skill（推荐，应对无网络客户机）

```bash
bash d.ops_develop/h.op_agent/b.skill_store/run.sh          # 交互式按模块勾选
bash d.ops_develop/h.op_agent/b.skill_store/run.sh list     # 列出仓库全部 skill
bash d.ops_develop/h.op_agent/b.skill_store/run.sh status   # 查看已装 skill
bash d.ops_develop/h.op_agent/b.skill_store/run.sh install <skill名>   # 装指定 skill
bash d.ops_develop/h.op_agent/b.skill_store/run.sh uninstall <skill名> # 卸载
```

- 仓库 skill 已加模块前缀（如 `cannbot-ascendc-precision-debug`、`mindcluster-ascend-env-check`），**避免同名覆盖**。
- 全离线，解压到 `b.skill_store/repo/`，选中后复制进 `~/.config/opencode/skills/`。

### 3. 联网获取最新 skill（开发机，需 Node 18+ 且 gitcode 可达）

```bash
bash d.ops_develop/h.op_agent/c.update_skill/run.sh             # 装全部最新
bash d.ops_develop/h.op_agent/c.update_skill/run.sh list        # 列出可装 skill
bash d.ops_develop/h.op_agent/c.update_skill/run.sh <skill名>   # 装指定 skill
```

> 说明：客户机若连 gitcode 超时/418（常见），请用 `b.skill_store` 离线选装，而非本功能。

## 关于 Skill 与 agent 的辨析

- **Skill**（能力）≠ **agent**（运行时）。散装 skill 装进去后，由 opencode 主 agent 通过 `SKILL.md` 的 `description` 语义匹配、按需调用。
- CANNBot 是成体系的（skill + agent 配套）；官方总仓其他模块（PyTorch/MindStudio/vllm-ascend 等）多为散装 skill。
- `openCode` 对 skill 名称的约束：`name` 必须与目录名一致，且只能小写字母数字 + 单连字符（不能含 `/`、下划线、大写）。

## 官方来源

- 官方总仓：https://gitcode.com/Ascend/agent-skills
- CANNBot（算子开发）：https://gitcode.com/cann/cannbot-skills
- Skill 开放标准：https://agentskills.io
- opencode skills 文档：https://opencode.ai/docs/skills/

安装后的 skills 位于：

```text
${XDG_CONFIG_HOME:-$HOME/.config}/opencode/skills/
```

## 打包说明（维护用）

`b.skill_store/skills.tar.gz` 由官方总仓 304 个 skill 抽取、重命名（加模块前缀 + 规范 `name`）而来。重新生成时使用仓库外的 `_build_skill_store.py` + `_tar.py` 脚本；重命名映射见 tar 内 `entries.json`。