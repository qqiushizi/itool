# a.llm_config — 大模型配置管理

> 工作流第 1 步：管理大模型接入配置，供 `e.op_design`、`g.op_fix` 等 LLM 功能复用。

## 配置目录

```text
d.ops_develop/workspace/llm_configs/
├── deepseek.json
├── local_vllm.json
└── ...
```

## 功能

- 新建配置
- 修改配置
- 删除配置
- 列出配置
- 测试配置是否可访问
- 设置某个配置为“当前使用配置”

## 运行

```bash
bash d.ops_develop/a.llm_config/run.sh
```

按菜单选择：

```text
A) 将某个配置设为当前使用
B) 新建配置
C) 修改配置
D) 删除配置
E) 测试配置
Q) 退出
```

## 生成文件示例

```json
{
  "provider": "external",
  "api_base": "https://api.deepseek.com/v1",
  "api_key": "sk-xxxx",
  "model": "deepseek-chat",
  "timeout": 120
}
```
