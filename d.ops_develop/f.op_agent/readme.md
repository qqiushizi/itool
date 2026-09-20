# f.op_agent — 算子 Agent 开发/测试/报告

> 算子开发工作流第 6 步：直接打开 OpenCode 进行算子开发，生成测试用例，运行测试，并输出测试报告。

## 运行

```bash
bash d.ops_develop/f.op_agent/run.sh
```

运行后按菜单选择：

```text
1) 选择/新建算子工程
2) 打开 OpenCode 开发
3) 生成测试用例
4) 编译工程
5) 运行测试
6) 生成测试报告
Q) 退出
```

## 工作流

```text
选择/新建 op_build_<算子名>
        ↓
打开 OpenCode（携带 AGENTS.md 任务说明）
        ↓
OpenCode 开发算子和测试脚本
        ↓
执行 build.sh / 测试脚本
        ↓
生成测试报告
```

## 生成物

```text
d.ops_develop/workspace/op_build_<算子名>/
├── AGENTS.md              # 交给 OpenCode 的任务说明
├── ...                    # OpenCode 生成的算子工程
└── report/
    ├── build.log
    ├── test.log
    └── test_report.md
```

## 前置条件

- 已进入算子开发容器
- 已安装 OpenCode + CANNBot skills，见 `e.install_opencode`
- 已安装并激活 CANN toolkit，见 `d.install_cann`

## 环境变量

```bash
# 自动模式：一条命令走完新建工程 → 编译 → 生成测试 → 编译 → 测试 → 报告
MODE=auto bash d.ops_develop/f.op_agent/run.sh AddCustom

# 指定 opencode 命令或 start.sh
OPCODE_CMD=opencode bash d.ops_develop/f.op_agent/run.sh

# 指定 workspace 路径
OPS_WORKSPACE=/data/ops_workspace bash d.ops_develop/f.op_agent/run.sh
```

## 说明

- `f.op_agent` 自动在工程目录写入 `AGENTS.md`，约束 OpenCode 只修改当前算子工程。
- 测试报告统一归档到 `report/test_report.md`。
- 若 `workspace/op_build_<算子名>/` 已存在，会复用该工程，不会覆盖。
