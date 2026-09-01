# Basic run
docker run -it --rm \
    mindspeed-llm:master-910b-openeuler24.03-py3.11-aarch64 bash

# Run with NPU device (Example: /dev/davinci1)
# Assume NPU device /dev/davinci1 and NPU driver installed at /usr/local/Ascend
docker run -it --rm \
    --name mindspeed-llm \
    --privileged \
        --device=/dev/davinci0 \
        --device=/dev/davinci1 \
        --device=/dev/davinci2 \
        --device=/dev/davinci3 \
        --device=/dev/davinci4 \
        --device=/dev/davinci5 \
        --device=/dev/davinci6 \
        --device=/dev/davinci7 \
        --device=/dev/davinci8 \
        --device=/dev/davinci9 \
        --device=/dev/davinci10 \
        --device=/dev/davinci11 \
        --device=/dev/davinci12 \
        --device=/dev/davinci13 \
        --device=/dev/davinci14 \
        --device=/dev/davinci15 \
    --network host \
    --ipc=host \
    --device=/dev/davinci1 \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home/:/home/ \
    -v /data:/data \
    -v /mnt:/mnt \
    mindspeed-llm:master-910b-openeuler24.03-py3.11-aarch64 \
    /bin/bash

# Enter the running container
docker exec -it mindspeed-llm /bin/bash
