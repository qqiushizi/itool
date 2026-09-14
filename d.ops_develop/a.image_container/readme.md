# a.image_container — 镜像拉取 + 容器实例化

> 算子开发工作流第 1 步：查询 `quay.io/ascend/cann` 官方 tag，客户可视化选择镜像；
> 本机已有则复用，没有则 `docker pull`；生成当前机器专用的 `start_container.sh`，确认后再启动。

## 运行

```bash
bash d.ops_develop/a.image_container/run.sh
```

交互流程：

1. 查询官方仓库可用 tag
2. 输入筛选关键字（例如 `9.1.0`、`910b`、`py3.10`、`devel`），选择 tag
3. 配置容器名、宿主机工作目录、共享内存、网络模式、是否 `--privileged`
4. 生成 `start_container.sh` 并预览
5. 询问是否立即启动，默认 `N`

## 非交互运行

显式指定镜像：

```bash
IMAGE=quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.10 bash d.ops_develop/a.image_container/run.sh
```

如果 `IMAGE` 只有 tag，会自动补 `quay.io/ascend/cann:` 前缀：

```bash
IMAGE=9.1.0-910b-ubuntu22.04-py3.10 bash d.ops_develop/a.image_container/run.sh
```

也可以同时指定镜像和容器名：

```bash
bash d.ops_develop/a.image_container/run.sh   quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.10   asc_dev
```

## 自动流程

1. 连接 `quay.io` API，查询 `ascend/cann` 仓库可达 tag
2. 按关键字筛选并让用户选择
3. 本机已存在该镜像则复用；不存在则执行 `docker pull`
4. 获取镜像稳定 ID：`docker image inspect -f '{{.Id}}'`
5. 检测当前机器的 NPU 设备与 Ascend 驱动挂载
6. 生成容器启动脚本并询问是否立即执行

## 容器挂载

| 类型 | 宿主机路径 | 容器路径 |
|---|---|---|
| NPU 设备 | `/dev/davinci*` | `/dev/davinci*` |
| 管理设备 | `/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc` | 同名 |
| 驱动 | `/usr/local/Ascend/driver` | `/usr/local/Ascend/driver` |
| DCMI | `/usr/local/dcmi` | `/usr/local/dcmi` |
| npu-smi | `/usr/local/bin/npu-smi` | `/usr/local/bin/npu-smi` |
| 用户工作目录 | `$WORK_DIR` | `/workspace` |
| itool 仓库 | 当前仓库根目录 | `/workspace/itool` |

容器默认工作目录为 `/workspace`。

## 起容器脚本

生成位置：

```text
$WORK_DIR/start_container.sh
```

默认 `$WORK_DIR` 为 `~/ascend_ops_workspace`。

脚本是**当前机器专用**模板，同一台机器可反复使用。换机器后应重新运行：

```bash
bash d.ops_develop/a.image_container/run.sh
```

## 常用环境变量

```bash
CANN_TAG_FILTER=9.1.0 # 查询时预填筛选关键字
IMAGE=...             # 显式指定镜像；只有 tag 时自动补官方前缀
NAME=asc_dev          # 容器名
WORK_DIR=/data/ops    # 宿主机工作目录
SHM_SIZE=16g          # 共享内存
NET_MODE=host         # host 或 bridge
PRIVILEGED=yes        # yes 或 no
EXTRA_ARGS=           # 额外 docker run 参数
```

## 下一步

```bash
docker exec -it asc_dev bash
cd /workspace/itool
bash d.ops_develop/b.env_check/run.sh
```
