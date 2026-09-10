# a.analysis — 算子需求分析

> 工作流第 4 步：把算子需求转成 `op.json` 与 `op_spec.md`。
> 支持两种分析方式：大模型分析、手动填写。

## 运行

```bash
bash d.ops_develop/d.op_design/a.analysis/run.sh
```

运行后会先选择：

```text
请选择分析方式:
  [1] 大模型分析
  [2] 手动填写
```

## 方式 1：大模型分析

用自然语言描述算子，由大模型提取结构化定义：

```text
实现矩阵乘法 MatMulCustom，输入 A/B 为 fp16，shape 支持 [M,K] 和 [K,N]，输出 C 为 fp16，shape [M,N]。
```

脚本会调用 OpenAI Chat Completions 协议接口，解析并展示结果；用户确认后才生成文件。

兼容：

- 外部 API
- 使用工具的昇腾宿主机上已启动的 vLLM 服务
- 其他暴露 `/v1/chat/completions` 的服务

常用环境变量：

```bash
ITOOL_LLM_API_BASE=http://127.0.0.1:8000/v1
ITOOL_LLM_API_KEY=sk-xxxx   # 宿主机 vLLM 服务可留空
ITOOL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
ITOOL_LLM_TIMEOUT=120
```

昇腾宿主机上已有 vLLM-ascend 服务时：

```bash
export ITOOL_LLM_API_BASE=http://127.0.0.1:8000/v1
# 在容器/宿主机上运行时，127.0.0.1 指向这台昇腾宿主机自身
export ITOOL_LLM_MODEL=/path/to/model

bash d.ops_develop/d.op_design/a.analysis/run.sh
```

外部 API 示例：

```bash
export ITOOL_LLM_API_BASE=https://api.deepseek.com/v1
export ITOOL_LLM_API_KEY=sk-xxxx
export ITOOL_LLM_MODEL=deepseek-chat

bash d.ops_develop/d.op_design/a.analysis/run.sh
```

也支持跳过交互：

```bash
OP_SPEC_MODE=llm OP_DESC_LLM='实现矩阵乘法 MatMulCustom...' ITOOL_LLM_API_BASE=http://127.0.0.1:8000/v1 ITOOL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct bash d.ops_develop/d.op_design/a.analysis/run.sh
```

## 方式 2：手动填写

选择 `[2]` 后，按提示填写算子名称、类型、输入/输出、shape、属性等。逻辑与原版保持一致。

也可以通过环境变量直接进入手动模式：

```bash
OP_SPEC_MODE=manual bash d.ops_develop/d.op_design/a.analysis/run.sh
```

## 输出

两种方式都会生成：

```text
op_design_<算子名>/
├── op.json
└── op_spec.md
```

`op.json` 可直接交给下一步 msopgen 生成工程：

```bash
bash d.ops_develop/e.op_scaffold/a.msopgen/run.sh op_design_<算子名>/op.json
```
