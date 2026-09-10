# b.env_check — 算子开发环境检查

> 算子在开发容器内执行；单脚本、只读，只给结论和建议，不安装/修复。

## 功能

直接运行：

```bash
bash d.ops_develop/b.env_check/run.sh
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

- 运行 `a.image_container` 并进入容器后，作为工作流第 2 步执行
- 客户现场交付时快速生成容器内环境检查报告
- 仅在宿主机上排查时也可直接运行，但工作流推荐容器内执行
