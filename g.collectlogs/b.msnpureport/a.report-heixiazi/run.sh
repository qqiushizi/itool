source ./setenv.sh

is_docker() {
    if [ -f "/proc/1/cgroup" ]; then
        grep -Eq 'docker|kubepods' /proc/1/cgroup
        return $?
    fi
    return 1
}

# 交互选择提示
echo "检测环境：当前是否运行在Docker容器中？"
echo "1) 是  2) 否"
read -p "请输入数字(1/2): " opt

if [ "$opt" = "1" ]; then
    echo "=== 执行容器内专属命令 msnpureport -f --docker ==="
    $msnpureport -f --docker    
elif [ "$opt" = "2" ]; then
    echo "=== 执行宿主机专属命令 msnpureport -f ==="
    $msnpureport -f
else
    echo "输入错误，仅支持 1 或 2"
    exit 1
fi

