# d.op_design — 算子需求分析

> 工作流第 4 步：把算子需求转成 `op.json` 与 `op_spec.md`。
> 支持：本地大模型、外部 API、手动填写。

## 运行

```bash
bash d.ops_develop/d.op_design/run.sh
```

运行后会先选择：

```text
请选择分析方式:
  [1] 本地大模型 (昇腾宿主机/容器内 vLLM-ascend 服务)
  [2] 外部 API (OpenAI Chat Completions 兼容接口)
  [3] 手动填写
```

选择 `[1]` 或 `[2]` 后，脚本会按对应方式逐个引导你完成大模型变量的声明，再收集算子需求并调用模型。

## 方式 1：本地大模型

适用于昇腾宿主机/容器内已经启动 vLLM-ascend 服务的场景。脚本会引导声明：

```text
ITOOL_LLM_API_BASE = http://127.0.0.1:8000/v1
ITOOL_LLM_MODEL    = Qwen/Qwen2.5-7B-Instruct   # 或 vLLM served model name
ITOOL_LLM_API_KEY  = (可留空)
```

在容器/宿主机上运行时，`127.0.0.1` 指向这台昇腾宿主机自身。

## 方式 2：外部 API

适用于 OpenAI Chat Completions 兼容的外部接口。脚本会引导声明：

```text
ITOOL_LLM_API_BASE = https://api.deepseek.com/v1
ITOOL_LLM_MODEL    = deepseek-chat
ITOOL_LLM_API_KEY  = sk-xxxx
```

也可使用其他兼容 `/v1/chat/completions` 的服务。

## 环境变量

脚本支持通过环境变量跳过交互：

```bash
export OP_SPEC_MODE=llm                 # llm | manual
export LLM_PROVIDER=local              # local | external；OP_SPEC_MODE=llm 时生效
export OP_DESC_LLM='实现矩阵乘法 MatMulCustom...'
export ITOOL_LLM_API_BASE=http://127.0.0.1:8000/v1
export ITOOL_LLM_API_KEY=sk-xxxx      # 本地 vLLM 服务可留空
export ITOOL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export ITOOL_LLM_TIMEOUT=120
```

### 本地大模型跳过交互示例

```bash
OP_SPEC_MODE=llm LLM_PROVIDER=local OP_DESC_LLM='实现矩阵乘法 MatMulCustom，输入 A/B 为 fp16，输出 C 为 fp16' ITOOL_LLM_API_BASE=http://127.0.0.1:8000/v1 ITOOL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct bash d.ops_develop/d.op_design/run.sh
```

### 外部 API 跳过交互示例

```bash
OP_SPEC_MODE=llm LLM_PROVIDER=external OP_DESC_LLM='实现矩阵乘法 MatMulCustom，输入 A/B 为 fp16，输出 C 为 fp16' ITOOL_LLM_API_BASE=https://api.deepseek.com/v1 ITOOL_LLM_API_KEY=sk-xxxx ITOOL_LLM_MODEL=deepseek-chat bash d.ops_develop/d.op_design/run.sh
```

## 方式 3：手动填写

选择 `[3]` 后，按提示填写算子名称、类型、输入/输出、shape、属性等。也可以通过环境变量直接进入：

```bash
OP_SPEC_MODE=manual bash d.ops_develop/d.op_design/run.sh
```

## 输出

任一方式都会生成：

```text
op_design_<算子名>/
├── op.json
└── op_spec.md
```

`op.json` 可直接交给下一步 msopgen 生成工程：

```bash
bash d.ops_develop/e.op_scaffold/a.msopgen/run.sh op_design_<算子名>/op.json
```
