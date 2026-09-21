# d.ops_develop — 算子开发工作流

> 场景：在客户机器上开发昇腾算子，或基于已有算子源码做改造。
> 输出形态：`字母.名称` 目录 + 叶子 `run.sh`。
> 生成物归档在 `d.ops_develop/workspace/` 下。

## 总览

```
d.ops_develop/
├── a.llm_api/                     ① 配置 OpenCode 的 API
│   ├── run.sh                 录入 baseURL / apiKey / model / provider
│   └── readme.md              功能说明
├── b.image_container/                 ② 镜像拉取 + 容器实例化
│   ├── run.sh                 查询/输入镜像、检查或拉取、生成 start_container.sh
│   └── readme.md              功能说明
├── c.env_check/                       ③ 容器内环境检查（只读）
│   ├── run.sh                 芯片型号识别 + 版本兼容矩阵 + 建议
│   └── readme.md              功能说明
└── d.op_agent/                        ④ OpenCode 运行时 + Skill 管理
    ├── a.install_opencode/   安装 opencode 绿色版（即开即用）
    ├── b.skill_store/        离线 Skill 仓库（选装/卸载）
    ├── c.update_skill/       联网更新 skill 仓库
    └── readme.md             说明
```

## 快速开始

```bash
# ① 设置 OpenCode 的 API（先录入一张模型配置）
bash d.ops_develop/a.llm_api/run.sh

# ② 拉镜像 + 建容器
bash d.ops_develop/b.image_container/run.sh

# 进入容器
docker exec -it asc_dev bash
cd /workspace/itool

# ③ 容器内环境检查（只读，给结论和建议）
bash d.ops_develop/c.env_check/run.sh

# ④ 安装 OpenCode 运行时 + 选装 Skill
bash d.ops_develop/d.op_agent/a.install_opencode/run.sh      # 安装 opencode 绿色版
bash d.ops_develop/d.op_agent/b.skill_store/run.sh           # 离线选装 Skill
bash d.ops_develop/d.op_agent/c.update_skill/run.sh          # 联网更新 skill 仓库（可选）
```

## 说明

- 所有脚本尽量少依赖、纯 bash + 标准命令，面向昇腾客户机（Linux），兼容 bash 3.2。
- `workspace/` 默认被 Git 忽略，客户机器上的生成物不会提交到 GitHub。
- `c.env_check` 只做只读检查，不修复、不安装。
- `d.op_agent` 是 OpenCode 的运行时 + 昇腾官方 skill 的离线仓库/在线更新；CANN toolkit、torch 等环境依赖由「容器 + 环境检查」提供建议，按需到 `f.installation` 补装。
