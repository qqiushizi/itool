# c.env_check — 算子开发环境检查

> 算子在开发容器内执行；检查阶段**只读**、只给结论和建议；检查结束后可选跳转到 `d.install_cann` 下载/安装缺失环境。

## 功能

直接运行：

```bash
bash d.ops_develop/c.env_check/run.sh
```

脚本会在**当前环境**完成：

1. **芯片型号识别汇总**
   - 优先解析 `npu-smi info`
   - 识别 `910B / 910A / 910C / 950 / 310P` 等常见昇腾芯片
   - 汇总 NPU 设备数

2. **软件版本兼容性矩阵判断**
   - Python ↔ torch
   - torch ↔ torch_npu
   - 芯片型号 ↔ CANN ↔ torch_npu

3. **环境补齐 / 下载安装（新）**
   - 检查结束后进入交互菜单，客户可选择下载/安装哪个 CANN 版本，跳转 `d.install_cann` 对应脚本
   - 提供 `按推荐下载安装` 选项：按芯片型号自动选择推荐 CANN 版本，自动下载+安装
   - 推荐安装失败时，显示失败原因与脚本最近输出，并列出**目前已安装环境版本**

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

## 环境补齐 / 下载安装菜单

检查汇总报告输出后，会进入一个交互菜单：

```text
请选择要下载/安装的环境组件（跳转 d.install_cann）:
  1) CANN 9.1.0   (较新稳定)
  2) CANN 9.0.0   (稳定)
  3) CANN 8.2.RC1 (旧芯片兼容)
  4) CANN 8.1.RC1 (旧芯片兼容)
  r) 按推荐下载安装 (CANN <推荐版本>)
  s) 跳过，不安装
```

- `1/2/3/4`：跳转 `d.install_cann/<版本目录>/run.sh`，由该安装脚本接管交互（下载 → 安装 → 激活）。
- `r`：按芯片型号自动选择推荐 CANN 版本，并以 `ITOOL_AUTO_DL=1 ITOOL_AUTO_INSTALL=1` 自动下载安装。
  - 推荐规则：`950 → 9.0.0`；`910A/B/C → 8.1.RC1`；`310P → 8.1.RC1`；未知芯片 → `9.1.0`。
  - 成功：显示安装成功的关键日志。
  - 失败：显示失败原因与脚本最近输出，并列出目前环境已安装版本。
- `s`：跳过，不安装。

安装完成后可重新运行本脚本复查版本。

## 使用场景

- 运行 `b.image_container` 并进入容器后，作为工作流第 2 步执行
- 客户现场交付时快速生成容器内环境检查报告，并在检查后直接补齐缺失环境
- 仅在宿主机上排查时也可直接运行，但工作流推荐容器内执行
