# a.image_container — 镜像拉取 + 容器实例化

> 本目录是算子开发工作流的第 1 步：给一个镜像地址，脚本负责检查/拉取镜像，并生成当前机器专用的起容器脚本。

## 运行

```bash
bash d.ops_develop/a.image_container/run.sh
```

脚本会询问镜像地址，例如：

```text
quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.10
```

也支持非交互：

```bash
IMAGE=quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.10 bash d.ops_develop/a.image_container/run.sh
```

或：

```bash
bash d.ops_develop/a.image_container/run.sh quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.10
```

## 自动流程

1. 检查本机是否已有该镜像：`docker image inspect`
2. 存在则直接用；不存在则自动执行 `docker pull`
3. 获取镜像稳定 ID：`docker image inspect -f '{{.Id}}'`
4. 检测当前机器存在的 NPU 设备和驱动挂载目录
5. 自动创建宿主机工作目录，默认是：

```text
~/ascend_ops_workspace
```

也支持覆盖：

```bash
WORK_DIR=/data/ops bash d.ops_develop/a.image_container/run.sh
```

6. 生成 `start_container.sh` 到该工作目录
7. 立即执行 `start_container.sh` 启动容器

## 容器挂载

生成脚本会按当前机器实际存在的内容自动挂载：

| 类型 | 路径 |
|---|---|
| NPU 设备 | `/dev/davinci*` |
| 管理设备 | `/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc` |
| 驱动 | `/usr/local/Ascend/driver` |
| DCMI | `/usr/local/dcmi` |
| npu-smi | `/usr/local/bin/npu-smi` |
| 工作目录 | `$WORK_DIR:/workspace` |

容器默认工作目录为 `/workspace`。

## 起容器脚本

生成的脚本位于：

```text
~/ascend_ops_workspace/start_container.sh
```

它是**当前机器专用**的起容器模板，适用于同一台机器反复重启容器。换机器后建议重新运行：

```bash
bash d.ops_develop/a.image_container/run.sh
```

## 常用环境变量

```bash
IMAGE=...    # 镜像地址
NAME=asc_dev # 容器名
WORK_DIR=    # 宿主机工作目录
SHM_SIZE=16g # 共享内存
EXTRA_ARGS=  # 额外 docker run 参数
```

## 下一步

容器启动后进入容器，再运行环境检查：

```bash
docker exec -it asc_dev bash
bash d.ops_develop/b.env_check/run.sh
```
