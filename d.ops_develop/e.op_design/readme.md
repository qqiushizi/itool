# e.op_design — 算子需求分析

> 工作流第 4 步：把算子需求转成 `op.json` 与 `op_spec.md`。
> 支持：本地大模型、外部 API、手动填写。

## 运行

```bash
bash d.ops_develop/e.op_design/run.sh
```

运行后会先选择：

```text
请选择分析方式:
  [1] 本地大模型 (昇腾宿主机/容器内 vLLM-ascend 服务)
  [2] 外部 API (OpenAI Chat Completions 兼容接口)
  [3] 手动填写
```

选择 `[1]` 或 `[2]` 后，脚本会按对应方式逐个引导你完成大模型变量的声明，再收集算子需求并调用模型。

## 方式 1 / 2：使用已有大模型配置

先在 `a.llm_config` 中创建好模型配置，然后 `e.op_design` 运行后会列出配置让你选择。

没有可用配置时会提示：

```bash
bash d.ops_develop/a.llm_config/run.sh
```

## 方式 3：手动填写

选择 `[3]` 后，按提示填写算子名称、类型、输入/输出、shape、属性等。也可以通过环境变量直接进入：

```bash
OP_SPEC_MODE=manual bash d.ops_develop/e.op_design/run.sh
```

## 输出

所有输出统一归档到：

```text
d.ops_develop/workspace/
└── op_design_<算子名>/
    ├── op.json
    └── op_spec.md
```

也可以手动指定输出目录：

```bash
OUT_DIR=/workspace/my_op_design bash d.ops_develop/e.op_design/run.sh
```

`op.json` 可直接交给下一步 msopgen 生成工程：

```bash
bash d.ops_develop/f.op_build/run.sh d.ops_develop/workspace/op_design_<算子名>/op.json

# 或者直接自动查找 workspace 下的 op.json
bash d.ops_develop/f.op_build/run.sh
```
