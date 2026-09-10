# a.install_cann — CANN toolkit 安装（下载 + 安装合并）

> 本目录只安装 `Ascend-cann-toolkit`，不安装 kernels/ops、合一包和驱动。

## 支持的稳定版本

| 版本 | 默认 | 说明 |
|---|---|---|
| `9.1.0` | ✅ 推荐 | 已验证官方 OBS URL 可达 |
| `9.0.0` | | 已验证官方 OBS URL 可达 |
| `8.2.RC1` | | 旧芯片兼容，已验证 URL 可达 |
| `8.1.RC1` | | 旧芯片兼容，已验证 URL 可达 |

## 运行

```bash
bash d.ops_develop/b.env_setup/a.install_cann/run.sh
```

## 常用非交互变量

```bash
# 仅探测官网 URL，不下载/不安装
CANN_VERSION=9.1.0 CHECK_ONLY=1 bash d.ops_develop/b.env_setup/a.install_cann/run.sh

# 自动下载 + 自动安装
CANN_VERSION=9.1.0 ITOOL_AUTO_DL=1 ITOOL_AUTO_INSTALL=1 bash d.ops_develop/b.env_setup/a.install_cann/run.sh

# 指定安装目录和包目录
CANN_VERSION=9.0.0 INSTALL_DIR=/opt/Ascend PKG_DIR=/opt/cann_pkgs bash d.ops_develop/b.env_setup/a.install_cann/run.sh
```
