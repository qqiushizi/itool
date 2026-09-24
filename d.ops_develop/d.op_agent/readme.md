# d.op_agent — OpenCode 运行时与 Skill 管理

> 算子开发工作流第 4 步：安装 OpenCode 绿色运行时，并提供离线的 Skill 仓库（选装）与联网更新仓库的能力。

## 背景与定位

itool 的算子开发依赖 opencode（提供工具调用的 agent 运行时）+ 昇腾官方 skill（提供领域知识）。

官方 skill 的分发体系：

```
Ascend/agent-skills（官方总仓/插件市场，覆盖全模块）
   └── official/CANNBot（submodule，算子开发 skill+agent，指向 cann/cannbot-skills）
```

- **Skills** 是"能力"，按 `description` 语义匹配、按需加载；**不是** agent 运行时。
- 本目录的三个功能职责划分：
  - **a** = 装 opencode 运行时，即开即用（离线）
  - **b** = 离线 skill 仓库，用户自己决定装什么
  - **c** = 联网更新 skill 仓库本身（不是直接装 opencode）

## 目录

```text
d.op_agent/
├── a.install_opencode/         安装 opencode 绿色版（二进制 + 配置，即开即用）
│   ├── opencode.tar.gz         = opencode + CANNBot 1.1.0（74 skill + 18 agent 快照）
│   └── run.sh
├── b.skill_store/              离线 Skill 仓库（选装，不依赖网络）
│   ├── skills.tar.gz           = 官方总仓技能（按模块目录分类，name 保持原名）
│   ├── run.sh
│   └── .gitignore
└── c.update_skill/             联网更新 skill_store 仓库（需 git + python3 + 网络）
    ├── run.sh
    └── build_store.py          # 打包脚本（从总仓抽取 → 目录分类 → 打 tar）
```

## 关键设计：目录分类 + 原名

opencode 只认 `skills/<name>/SKILL.md` 一层扁平结构，且 `name` 须与目录名一致、不能含 `/`。因此采用：

- **仓库侧**（skills.tar.gz 解压后）：`<模块>/<原名>/SKILL.md`，用目录做分类，`name` 字段保持原名
- **安装侧**（装进 opencode）：把 skill 按**原名**复制进 skills 目录，不拼前缀、不改名
- **同名直接覆盖**：可用来更新 tar 包自带的旧 skill（如 CANNBot 的 74 个）

模块目录（11 个）：`cannbot`、`pytorch`、`common`、`community-op`、`mindstudio`、`mindspeed`、`mindcluster`、`verl`、`mindseries-sdk`、`vllm-ascend`、`community-tools`

## 使用

### 1. 安装 opencode 运行时（离线，即开即用）

```bash
bash d.ops_develop/d.op_agent/a.install_opencode/run.sh
```

解压后即拥有 opencode + CANNBot 1.1.0 基础能力（74 skill + 18 agent），无需联网。

> 如需**只解压、不启动**（供其它脚本调用）：`bash d.ops_develop/d.op_agent/a.install_opencode/run.sh ensure`

> 注意：便携版 opencode 由 `a.install_opencode/run.sh` 启动，脚本内部把 `XDG_CONFIG_HOME`
> 指向绿色包内的 `config/`，因此其真实 skills 目录是 `opencode/config/opencode/skills/`
> （而非 `~/.config/opencode/skills/`）。

### 2. 离线选装 skill（推荐，应对无网络客户机）

```bash
bash d.ops_develop/d.op_agent/b.skill_store/run.sh          # 交互式按模块勾选
bash d.ops_develop/d.op_agent/b.skill_store/run.sh list     # 列出仓库全部 skill（按模块）
bash d.ops_develop/d.op_agent/b.skill_store/run.sh status   # 查看已装 skill
bash d.ops_develop/d.op_agent/b.skill_store/run.sh install <skill原名>   # 装指定 skill
bash d.ops_develop/d.op_agent/b.skill_store/run.sh uninstall <skill原名> # 卸载
```

- 全离线，解压到 `b.skill_store/repo/`，选中后按原名复制进 opencode 的 `config/opencode/skills/`
- 同名 skill 会选择覆盖（用于更新基础包里的旧版）
- **顺序无关**：若 opencode 绿色包尚未解压，本脚本会自动调用 `a.install_opencode/run.sh ensure` 先解压，再装 skill

### 3. 联网更新 skill 仓库（开发机，需 git + python3 + 网络）

```bash
bash d.ops_develop/d.op_agent/c.update_skill/run.sh
```

- 从官方总仓浅克隆/更新后，重新构建 `b.skill_store/skills.tar.gz`
- 更新的是**仓库**，之后仍需 `b.skill_store/run.sh` 选装

> 说明：客户机连 gitcode 超时/418（常见），请用 `b.skill_store` 离线选装。

## 关于 Skill 与 agent 的辨析

- **Skill**（能力）≠ **agent**（运行时）。散装 skill 装进去后，由 opencode 主 agent 通过 `SKILL.md` 的 `description` 语义匹配、按需调用。
- CANNBot 是成体系的（skill + agent 配套）；官方总仓其他模块（PyTorch/MindStudio/vllm-ascend 等）多为散装 skill。
- opencode 对 skill 名称的约束：`name` 必须与目录名一致，且只能小写字母数字 + 单连字符（不能含 `/`、下划线、大写）。

## 官方来源

- 官方总仓：https://gitcode.com/Ascend/agent-skills
- CANNBot（算子开发）：https://gitcode.com/cann/cannbot-skills
- Skill 开放标准：https://agentskills.io
- opencode skills 文档：https://opencode.ai/docs/skills/

## 打包说明（维护用）

`b.skill_store/skills.tar.gz` 由官方总仓抽取、按模块目录分类、name 保持原名、同名覆盖而来。重新生成方式：

```bash
# 联网重新打包（推荐）
bash d.ops_develop/d.op_agent/c.update_skill/run.sh

# 或手动（已有总仓 clone）
python3 c.update_skill/build_store.py <总仓路径> b.skill_store/skills.tar.gz
```

映射关系（模块 → 总仓路径）见 `c.update_skill/build_store.py` 的 `MODULE_MAP`。