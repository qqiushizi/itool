# c.env_check — 算子开发环境检查

> 算子在开发容器内执行；单脚本、只读，只给结论和建议，不安装/修复。

## 功能

直接运行：

```bash
bash d.ops_develop/c.env_check/run.sh
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
- 容器内若同时存在多个 Python（例如 `/usr/bin/python3` 与 `/usr/local/python3.x`），
  脚本会优先选择**能成功 `import torch`** 的解释器做 torch / torch_npu 检测，并在报告中提示实际使用的解释器
- 不会删除、重装或修改系统中已有的 Python
- CANN 安装目录、版本文件、`set_env.sh` 激活脚本
- CANN 版本识别会自动从 `ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME`、`ASCEND_OPP_PATH`
  以及 `/usr/local/Ascend` 下的目录查找 `version.cfg` / `version`
- `version.cfg` 中优先读取 `toolkit_running_version`，不会把 `cann_running_version` 等包版本字段误报为 toolkit
- 根目录的 `version.info` 只作为低优先级补充；不会递归采用 `share/info/mindstudio-debugger`
  等子组件目录里的 `version.info`，避免把子模块版本误当成 CANN 版本
- 多版本共存时，优先识别当前激活目录（`ASCEND_TOOLKIT_HOME` / `latest`）；无法定位时再取 CANN 版本号最高者
- 仍找不到版本文件时，会从 `cann-9.1.0`、`ascend-toolkit/9.1.0` 这类目录名中推断版本
- 综合结论：`匹配` 或 `存在缺失/不匹配`
- 下一步建议命令（只提示，不执行）

## 使用场景

- 运行 `b.image_container` 并进入容器后，作为工作流第 2 步执行
- 客户现场交付时快速生成容器内环境检查报告
- 仅在宿主机上排查时也可直接运行，但工作流推荐容器内执行
