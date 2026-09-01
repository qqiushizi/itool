#!/bin/bash
echo "========== Docker / Container Info =========="
echo ""

echo "[Docker Version]"
docker --version 2>/dev/null || echo "Docker not installed"
docker info 2>/dev/null | head -10 || echo "Docker daemon not accessible"

echo ""
echo "[Container Images]"
docker images 2>/dev/null | grep -iE "ascend|pytorch|mindspeed|llm" || echo "No Ascend/PyTorch images found"

echo ""
echo "[Running Containers]"
docker ps 2>/dev/null | grep -i ascend || echo "No running Ascend containers"

echo ""
echo "[Docker Environment Variables]"
echo "DOCKER_REGISTRY: $DOCKER_REGISTRY"
echo "ASCEND_DOCKER_IMAGE: $ASCEND_DOCKER_IMAGE"

echo ""
echo "[Linux Container Info]"
if command -v ctr 2>/dev/null; then
    ctr images list 2>/dev/null | grep -iE "ascend|pytorch" || echo "No container images"
elif command -v podman 2>/dev/null; then
    podman images 2>/dev/null | grep -iE "ascend|pytorch" || echo "No podman images found"
fi

echo ""
echo "=============================================="
