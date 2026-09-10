# c.image_container — 镜像拉取 + 容器实例化

> 本目录合并了原来的 `b.pull_image` 和 `c.run_container`。

## 功能

```bash
bash d.ops_develop/c.image_container/run.sh
```

脚本会依次完成：

1. 查询 `quay.io/ascend/cann` 镜像 tag
2. 可按 芯片 / CANN 版本 / 系统 / Python 筛选
3. `docker pull` 选中镜像，并打本地短标签
4. 收集容器名 / 工作目录 / 共享内存
5. 枚举 `/dev/davinci*` 设备并生成 `start_container.sh`
6. 立即启动容器

## 快速指定镜像

```bash
IMAGE=quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.10 \
NAME=asc_dev WORK_DIR=/data/ops SHM_SIZE=16g \
bash d.ops_develop/c.image_container/run.sh
```
