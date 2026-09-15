# g.op_fix — 大模型辅助修改算子工程

> 工作流第 7 步：对 `f.op_build` 生成的算子工程进行多轮对话式代码修改。

## 运行

```bash
bash d.ops_develop/g.op_fix/run.sh
```

脚本会先让你选择：

1. 要修改的算子工程
2. 要使用的大模型配置

然后进入对话模式：

```text
[op_fix] > 请帮我检查 host 侧 shape 推导
[AI]     > ...
```

## 能力

- 自动读取算子工程中的 `.cpp` / `.h` / `CMakeLists.txt` 等文件
- 保持多轮上下文
- 模型输出文件修改块后，可选择性应用
- 应用前自动备份原文件为 `.bak`

## 上下文记忆

每个工程单独保存上下文：

```text
workspace/op_build_<算子名>/.itool/op_fix_history.jsonl
```

重新进入时会恢复历史上下文。

## 命令

| 命令 | 作用 |
|---|---|
| 直接输入需求 | 让大模型分析/修改工程 |
| `files` | 查看当前工程文件 |
| `history` | 查看上下文消息数 |
| `help` | 查看帮助 |
| `quit` / `exit` | 退出 |
