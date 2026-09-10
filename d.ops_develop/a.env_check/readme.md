# a.env_check — 算子开发环境检查

> 单脚本、只读、宿主机/容器通用。

## 功能

直接运行：

```bash
bash d.ops_develop/a.env_check/run.sh
```

脚本会在**当前环境**完成两件核心事：

1. **芯片型号识别汇总**
   - 优先解析 `npu-smi info`
   - 识别 `910B / 910A / 910C / 950 / 310P` 等常见昇腾芯片
   - 汇总 NPU 设备数

2. **软件版本兼容性矩阵判断**
   - Python ↔ torch
   - torch ↔ torch_npu
   - 芯片型号 ↔ CANN ↔ torch_npu

同时输出：

- Python / CANN / torch / torch_npu 版本
- CANN 安装目录、`set_env.sh` 激活脚本
- 综合结论：`匹配` 或 `存在缺失/不匹配`
- 下一步建议命令（只提示，不执行）

## 使用场景

- 宿主机上装完环境后自查
- 进入算子开发容器后再次确认容器内环境
- 客户现场交付时快速生成环境检查报告
