# a.install_cann — CANN toolkit 安装

> 进入本目录后，按 `a` / `b` / `c` / `d` 选择版本；每个版本都是独立可执行的 `run.sh`。
> 本目录只安装 `Ascend-cann-toolkit`，不安装 kernels/ops、合一包和驱动。

## 版本目录

| 目录 | CANN 版本 | 说明 |
|---|---|---|
| `a.cann-9.1.0/` | `9.1.0` | 推荐 |
| `b.cann-9.0.0/` | `9.0.0` | 稳定 |
| `c.cann-8.2.RC1/` | `8.2.RC1` | 旧芯片兼容 |
| `d.cann-8.1.RC1/` | `8.1.RC1` | 旧芯片兼容 |

## 运行

```bash
# 推荐版本
bash d.ops_develop/b.env_setup/a.install_cann/a.cann-9.1.0/run.sh

# 9.0.0
bash d.ops_develop/b.env_setup/a.install_cann/b.cann-9.0.0/run.sh
```

每个版本脚本会完成：查包/自动下载 → 安装到指定目录 → `source set_env.sh` → 验证 `acl` → 可选写入 `~/.bashrc`。

## 常用非交互变量

```bash
# 只探测官方 URL，不下载/不安装
CHECK_ONLY=1 bash d.ops_develop/b.env_setup/a.install_cann/a.cann-9.1.0/run.sh

# 自动下载 + 自动安装
ITOOL_AUTO_DL=1 ITOOL_AUTO_INSTALL=1 bash d.ops_develop/b.env_setup/a.install_cann/a.cann-9.1.0/run.sh

# 指定安装目录 / 包目录
INSTALL_DIR=/opt/Ascend PKG_DIR=/opt/cann_pkgs bash d.ops_develop/b.env_setup/a.install_cann/b.cann-9.0.0/run.sh
```
