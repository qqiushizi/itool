# b.image_container — 镜像拉取 + 容器实例化

> 算子开发工作流第 1 步：从 quay.io 官方仓库查询 tag，客户可视化选择镜像；
> 本机已有则复用，没有则 `docker pull`；生成当前机器专用的 `start_container.sh`，确认后再启动。

## 支持的官方仓库

运行脚本后可以选择：

| 选项 | 仓库 | 用途 |
|---|---|---|
| A | `quay.io/ascend/vllm-ascend` | vLLM Ascend 推理容器 |
| B | `quay.io/ascend/cann` | CANN 算子开发容器 |

也可以通过环境变量直接指定：

```bash
QUAY_REPO=quay.io/ascend/vllm-ascend bash d.ops_develop/b.image_container/run.sh
```

## 运行

```bash
bash d.ops_develop/b.image_container/run.sh
```

交互流程：

1. 选择官方仓库：`vllm-ascend` / `cann`
2. 查询官方仓库可用 tag（仅作参考，最大查询 5 页；失败也不会卡住）
3. **以手动输入为主**：输入 tag 或完整镜像均可以，也可输入编号选择参考项
4. 通过 A/B/C/D 选项配置：容器名、宿主机工作目录、共享内存、网络模式、`--privileged`
5. 生成 `start_container.sh`
6. 询问是否立即启动，默认 `N`

## 非交互运行

指定完整镜像：

```bash
IMAGE=quay.io/ascend/vllm-ascend:v0.27.1-910b-ubuntu22.04-py3.10 bash d.ops_develop/b.image_container/run.sh
```

指定仓库和 tag：

```bash
QUAY_REPO=quay.io/ascend/vllm-ascend IMAGE=v0.27.1-910b-ubuntu22.04-py3.10 bash d.ops_develop/b.image_container/run.sh
```

只给 tag 时，默认通过 `QUAY_REPO` 补全仓库前缀；没有 `QUAY_REPO` 时会交互选择仓库。

## 自动流程

1. 选择/确定 quay.io 官方仓库
2. 连接 quay.io API 查询可达 tag（参考项，最多展示 20 个；可通过 `TAG_FILTER` 预筛）
3. 用户手动输入 tag 或完整镜像，或按编号选择参考项
4. 本机已存在该镜像则复用；不存在则执行 `docker pull`
5. 获取镜像稳定 ID：`docker image inspect -f '{{.Id}}'`
6. 检测当前机器的 NPU 设备与 Ascend 驱动挂载
7. 生成容器启动脚本并询问是否立即执行

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
bash d.ops_develop/b.image_container/run.sh
```

## 常用环境变量

```bash
QUAY_REPO=quay.io/ascend/vllm-ascend  # 官方仓库，默认为 vllm-ascend；可选 quay.io/ascend/cann
QUAY_MIRROR=                          # 拉取失败时使用的国内镜像，默认 m.daocloud.io/quay.io 和 quay.nju.edu.cn
TAG_FILTER=v0.27                      # 查询时预填筛选关键字
CANN_TAG_FILTER=                      # 兼容旧变量名
IMAGE=...                             # 显式指定镜像；只有 tag 时自动补 QUAY_REPO 前缀
NAME=asc_dev                          # 容器名
WORK_DIR=/data/ops                    # 宿主机工作目录
SHM_SIZE=16g                          # 共享内存
NET_MODE=host                         # host 或 bridge
PRIVILEGED=yes                        # yes 或 no
EXTRA_ARGS=                           # 额外 docker run 参数
```

## 下一步

```bash
docker exec -it asc_dev bash
cd /workspace/itool
bash d.ops_develop/c.env_check/run.sh
```
