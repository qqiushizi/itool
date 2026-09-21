# my-skill — 用户自定义 skill 目录

这里放你自己写的 skill，会被 `b.skill_store/run.sh` 的选装功能**自动识别**（与 tar 解压的官方 `repo/` 一起扫描）。

## 怎么放

按「模块 / skill」两级目录组织，和官方 `repo/` 结构一致：

```
my-skill/
└── <你的模块名>/          # 比如 a5、my-tools、company-ops ...
    └── <skill名>/         # 目录名 = skill 名
        └── SKILL.md       # 必须有规范的 frontmatter
```

## SKILL.md 必须满足

1. 有 YAML frontmatter，且含 `name` 和 `description` 两个必填字段
2. `name` 必须与所在目录名**完全一致**
3. `name` 只能小写字母数字 + 单连字符（不能含 `/`、下划线、大写）

示例：

```yaml
---
name: my-op-perf-analysis
description: 分析自定义算子的性能瓶颈，输出优化建议
---
# 正文……
```

## 注意事项

- 本目录随 git 跟踪，更新/重打包官方仓库（`c.update_skill`）**不会**覆盖它
- 重名时会覆盖 opencode 里已装的同名 skill
- 写好后执行 `bash ../run.sh list` 即可看到它出现在对应模块下

## 已内置的 A5 skill（模块 a5）

本目录已预置 3 个昇腾 A5（Ascend 950 系列）相关 skill，均基于昇腾官方文档编写：

- `a5-arch-overview` — A5 架构与 SIMD/SIMT 混合编程模型概览
- `a5-simt-programming` — SIMT 与 SIMD/SIMT 混合编程选型指导
- `a5-env-check` — A5 型号确认与环境检查