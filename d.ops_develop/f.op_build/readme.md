# f.op_build — 算子工程构建

> 工作流第 5 步：用 CANN `msopgen` 结合 `op.json` 建立算子工程。
> 输入、输出统一放在 `d.ops_develop/workspace/` 下。

## 运行

```bash
# 自动在 d.ops_develop/workspace/ 下查找 op.json
bash d.ops_develop/f.op_build/run.sh
```

如果有多个 `op.json`，脚本会列出让你选择。

也可以直接指定：

```bash
bash d.ops_develop/f.op_build/run.sh d.ops_develop/workspace/op_design_AddCustom/op.json
```

指定 chip arch：

```bash
bash d.ops_develop/f.op_build/run.sh d.ops_develop/workspace/op_design_AddCustom/op.json 910B
```

工程生成方式固定为 AscendC/C++：

```bash
msopgen gen -i <op.json> -f tf -lan cpp -c <compute_unit> -out <输出目录>
```

arch 会自动转换成 msopgen 需要的格式：

```text
910B          -> ai_core-ascend910B
Ascend910B    -> ai_core-ascend910B
310P          -> ai_core-ascend310P
ai_core-ascend910B -> ai_core-ascend910B（保持不变）
```

## 输出

```text
d.ops_develop/workspace/
└── op_build_<算子名>/
    └── ...  # msopgen 生成的算子工程
```

## 环境变量

```bash
OPS_WORKSPACE=/data/ops_workspace   # 默认 d.ops_develop/workspace
OUT_DIR=/data/custom_build          # 覆盖 msopgen 输出目录
REPO_ROOT=/path/to/itool            # 默认自动识别 itool 仓库根目录
```

## 依赖

- 已安装并激活 CANN toolkit（`source set_env.sh`）
- 能找到 `msopgen`
- 已由 `e.op_design` 生成 `op.json`
