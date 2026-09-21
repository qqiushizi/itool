# a.opencode_api — 配置 OpenCode 的 API

> 算子开发工作流第 1 步：给 OpenCode 录入 OpenAI 兼容接口（baseURL / apiKey / model / provider），
> 写入 opencode 的配置文件，供 `d.op_agent` 安装的 opencode 直接调用大模型。

## 做什么

- 交互式录入 `provider / baseURL / apiKey / model / timeout`
- 写入 opencode 标准配置 `~/.config/opencode/opencode.json`
- 支持查看当前配置、测试连接（访问 `baseURL/models`）

## 运行

```bash
bash d.ops_develop/a.opencode_api/run.sh              # 菜单
bash d.ops_develop/a.opencode_api/run.sh config       # 直接配置
bash d.ops_develop/a.opencode_api/run.sh view         # 查看当前配置
bash d.ops_develop/a.opencode_api/run.sh test         # 测试连接
```

菜单：

```text
A) 配置 API(baseURL / apiKey / model / provider)
B) 查看当前配置
C) 测试连接(访问 baseURL/models)
Q) 退出
```

## 非交互方式

```bash
OPENCODE_API_BASE=https://api.deepseek.com/v1 \
OPENCODE_API_KEY=sk-xxx \
OPENCODE_MODEL=deepseek-chat \
OPENCODE_PROVIDER=deepseek \
bash d.ops_develop/a.opencode_api/run.sh config
```

支持环境变量：`OPENCODE_API_BASE` / `OPENCODE_API_KEY` / `OPENCODE_MODEL` /
`OPENCODE_PROVIDER`(或 `OPENCODE_TAG`) / `OPENCODE_TIMEOUT` / `OPENCODE_CONFIG`。

## 写入的 opencode.json 结构

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "deepseek/deepseek-chat",
  "provider": {
    "deepseek": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "deepseek",
      "options": {
        "baseURL": "https://api.deepseek.com/v1",
        "timeout": 120000,
        "apiKey": "sk-xxx"
      },
      "models": {
        "deepseek-chat": { "name": "deepseek-chat" }
      }
    }
  },
  "tools": { "bash": true, "edit": true, "read": true, "write": true },
  "autoupdate": false
}
```
