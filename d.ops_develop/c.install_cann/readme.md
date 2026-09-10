# c.install_cann — CANN toolkit 安装

> 默认场景：你已经运行完 `a.image_container`，并通过 `docker exec` 进入算子开发容器。
> 进入本目录后，按 `a` / `b` / `c` / `d` 选择版本；每个版本都是独立可执行的 `run.sh`。
> 本目录只安装 `Ascend-cann-toolkit`，不安装 kernels/ops、合一包和驱动。

## 版本目录

| 目录 | CANN 版本 | 说明 |
|---|---|---|
| `a.cann-9.1.0/` | `9.1.0` | 推荐 |
| `b.cann-9.0.0/` | `9.0.0` | 稳定 |
| `c.cann-8.2.RC1/` | `8.2.RC1` | 旧芯片兼容 |
| `d.cann-8.1.RC1/` | `8.1.RC1` | 旧芯片兼容 |

## 前置条件

先确保已经进入容器：

```bash
docker exec -it <容器名> bash
```

> 未进入容器直接运行时，脚本会提示并退出；仅做宿主机 URL 探测可加 `ITOOL_ALLOW_HOST=1`。

## 运行

```bash
# 推荐版本
bash d.ops_develop/c.install_cann/a.cann-9.1.0/run.sh

# 9.0.0
bash d.ops_develop/c.install_cann/b.cann-9.0.0/run.sh
```

每个版本脚本会完成：查包/自动下载 → 安装到指定目录 → `source set_env.sh` → 验证 `acl` → 可选写入 `~/.bashrc`。

## CPU 架构自动识别

脚本会按以下顺序识别架构：

1. `uname -m`
2. `lscpu` 的 `Architecture`
3. `dpkg --print-architecture`

并归一化为：

- `x86_64` → `Ascend-cann-toolkit_*_linux-x86_64.run`
- `aarch64` → `Ascend-cann-toolkit_*_linux-aarch64.run`

如果识别不到，会询问用户手动确认架构。

## 安装目录防覆盖

- 容器内默认建议安装到：`/workspace/Ascend/ascend-toolkit-<version>`
- `/workspace` 由 `a.image_container` 从宿主机工作目录挂载而来，容器重建后仍能保留安装结果。
- 如果用户输入的目录已经存在 CANN 标识文件（`set_env.sh` / `version.cfg` / `version` / `latest`），默认直接拒绝覆盖并退出。
- 只有显式 `ITOOL_FORCE=1` 并再次确认后，才允许写入已有 CANN 目录。

## 常用非交互变量

```bash
# 在宿主机只探测官方 URL，不下载/不安装
ITOOL_ALLOW_HOST=1 CHECK_ONLY=1 bash d.ops_develop/c.install_cann/a.cann-9.1.0/run.sh

# 自动下载 + 自动安装
ITOOL_AUTO_DL=1 ITOOL_AUTO_INSTALL=1 bash d.ops_develop/c.install_cann/a.cann-9.1.0/run.sh

# 指定安装目录 / 包目录（必须在容器内）
INSTALL_DIR=/opt/Ascend PKG_DIR=/opt/cann_pkgs bash d.ops_develop/c.install_cann/b.cann-9.0.0/run.sh

# 非强制覆盖已有 CANN 目录
# 默认禁止; 仅当你明确知道后果才加 ITOOL_FORCE=1
ITOOL_FORCE=1 bash d.ops_develop/c.install_cann/a.cann-9.1.0/run.sh
```
