# workspace — ops_develop 统一工作区

`d.ops_develop` 下所有脚本生成的内容默认都归档在这里：

```text
workspace/
├── op_design_<算子名>/    # e.op_design 输出：op.json + op_spec.md
├── op_build_<算子名>/     # f.op_build 输出：msopgen 算子工程
└── ...
```

- 各脚本默认从这里写入和读取。
- 可通过环境变量 `OPS_WORKSPACE` 覆盖。
- 本目录默认被 Git 忽略，客户机器上的生成物不会提交到 GitHub。
