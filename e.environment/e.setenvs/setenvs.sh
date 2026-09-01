#!/bin/bash

# private env
# sed -i 's/\r$//' ienv/isetenv.sh
export LXY_ROOT_PATH=/data2/lxy
export IENV_FILE=$LXY_ROOT_PATH/isetenv.sh
export IENV_CONFIG_FILE=$LXY_ROOT_PATH/icfg.ini
export LXY_ASCEND_TOOLKIT=/usr/local/Ascend/ascend-toolkit/

if [ ! -d $LXY_ROOT_PATH ]; then
	exit 1
fi

export LLL=lxy
# example: idocker pytorch:lxy1.0 lxy /home/lxy
idocker() {
	IMAGE_ID=$1
	CONTAINER_NAME=$2
	WORK_PATH=$3

	docker stop $CONTAINER_NAME
	docker rm $CONTAINER_NAME

	docker run -itd \
	--ipc=host \
	--network=host \
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
	--device=/dev/davinci_manager \
	--device=/dev/devmm_svm \
	--device=/dev/hisi_hdc \
	-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
	-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
	-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
	-v /usr/local/sbin/:/usr/local/sbin/ \
	-v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
	-v /var/log/npu/slog/:/var/log/npu/slog \
	-v /var/log/npu/profiling/:/var/log/npu/profiling \
	-v /var/log/npu/dump/:/var/log/npu/dump \
	-v /var/log/npu/:/usr/slog \
        -v /data2/lxy/set_env.sh:/usr/local/Ascend/cann-8.5.0/share/info/ascendnpu-ir/bin/set_env.sh \
	-v $WORK_PATH:$WORK_PATH \
	--name=$CONTAINER_NAME \
	$IMAGE_ID \
	bash

	__save_env iexec "docker exec -ti $CONTAINER_NAME /bin/bash -c \"cd $WORK_PATH && exec /bin/bash\""
}


__save_env() {
    local key=$1
    local value="${@:2}"
    local file="$IENV_CONFIG_FILE"

    alias "$key"="$value"

    [ ! -f "$file" ] && touch "$file"

    # update key and value
    grep -q "^$key=" "$file" && sed -i "/^$key=/c\\$key=$value" "$file" || echo "$key=$value" >> "$file"
}

# example set env
#save_and_set_env AAA 111

__read_env() {
    local key=$1
	local value="${@:2}"
    local file="$IENV_CONFIG_FILE"

	if [[ -f "$file" ]]; then
		value=$(grep "^$key=" "$file" | cut -d'=' -f2-)
		if [[ -n "$value" ]]; then
			alias "$key"="$value"
			echo "alias $key=$value"
		fi
	else
		echo "File $file does not exist."
	fi
}

__init_env() {
	local file="$IENV_CONFIG_FILE"
	if [[ ! -f "$file" ]]; then
		return
	fi
	while IFS='=' read -r key value; do
		key=$(echo $key | xargs)
		value=$(echo $value | xargs)
		if [[ -z "$key" || $key == \#* ]]; then
			continue
		fi
		alias "$key"="$value"
	done < "$file"
}

icmd() {
	last_command=$(history | tail -n 2 | head -n 1 | awk '{$1=""; print $0}')
	__save_env $1 "$last_command"
}

ipath() {
	__save_env $1 "cd $PWD"
}

iproxy() {
	local PROXY_IP_PORT="127.0.0.1:7890"
	# 如果$1不为空，PROXY_IP_PORT=$1
	if [[ -n "$1" ]]; then
		PROXY_IP_PORT=$1
	fi
	export http_proxy=http://$PROXY_IP_PORT
	export https_proxy=$http_proxy
	export no_proxy="localhost,127.0.0.1,.local,.internal"

	export GIT_SLL_NO_VERIFY=1
	git config --global --replace-all http.proxy $http_proxy
	git config --global --replace-all https.proxy $http_proxy
	git config --global http.sslverify false
}

iproxytx() {
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"
}

alias icproxy="export http_proxy=;export https_proxy="

iutf8(){
    # export LANG=en_US.UTF-8
    # export LANGUAGE=en_US.UTF-8
    # export LC_ALL=en_US.UTF-8

    # 在vi中支持utf8
    echo -e "set encoding=utf-8" >> ~/.exrc
    echo -e "set termencoding=utf-8" >> ~/.exrc
    echo -e "set fileencoding=utf-8" >> ~/.exrc
    echo -e "set fileencodings=utf-8,gbk,gb2312,big5,latin1" >> ~/.exrc

    # 在vim中支持utf8
    echo -e "set encoding=utf-8" >> ~/.vimrc
    echo -e "set termencoding=utf-8" >> ~/.vimrc
    echo -e "set fileencoding=utf-8" >> ~/.vimrc
    echo -e "set fileencodings=utf-8,gbk,gb2312,big5,latin1" >> ~/.vimrc
}

alias iroot="cd $LXY_ROOT_PATH"
alias itool="cd $LXY_ROOT_PATH/itool;source ./itool.sh"
alias ienv="vi $IENV_FILE"
alias icfg="vi $IENV_CONFIG_FILE"
alias isource="source $IENV_FILE" 
alias iisource="source /usr/local/Ascend/ascend-toolkit/set_env.sh"
alias iwget="wget -c --no-check-certificate"
alias itail="tail -200f"
alias ihead="head -200f"
alias igrep="ps -ef | grep"
alias ipip="export PIP_INDEX_URL=https://pypi.org/simple/;export PIP_EXTRA_INDEX_URL=http://mirrors.huaweicloud.com/ascend/repos/pypi;pip install "

# public env
export HF_ENDPOINT=https://hf-mirror.com

alias inpu="npu-smi info"
alias idriver="echo Driver Version;cat /usr/local/Ascend/driver/version.info | grep Version"
alias ifw="cat /usr/local/Ascend/firmware/version.info | grep Version"
alias iplog="cd /root/ascend/log/debug/plog;ls -ltr"
alias icann="cat $LXY_ASCEND_TOOLKIT/latest/version.cfg | grep toolkit_running_version"
alias itorch="pip list | grep torch"
alias iascend="idriver;ifw;icann;itorch"

ihccl(){
	rank_per_node=16
	rank_max=$(($rank_per_node - 1))

	# 检查ip配置和tls状态
	if [[ "$1" == "" ]]; then
		echo "检查ip配置和tls状态"
		for i in $(seq 0 $rank_max); 
		do 
			hccn_tool -i $i -ip -g; 
			hccn_tool -i $i -gateway -g; 
			hccn_tool -i $i -netdetect -g; 
			echo--------------------;
		done
		for i in $(seq 0 $rank_max); do hccn_tool -i $i -tls -g | grep 'tls switch'; done
	fi

	# 检查网卡健康状态
	if [[ "$1" == "health" ]]; then
		echo "检查网卡健康状态"
		for i in $(seq 0 $rank_max); do hccn_tool -i $i -net_health -g; done
	fi

	# 检查连接状态
	if [[ "$1" == "link" ]]; then
		echo "检查连接状态"
		for i in $(seq 0 $rank_max); do hccn_tool -i $i -link -g; done
	fi

	# 检查光模块状态
	if [[ "$1" == "opt" ]]; then
		echo "检查光模块状态"
		for i in $(seq 0 $rank_max); do hccn_tool -i $i -optical -g; done | grep present
	fi

	# 重置npu状态
	if [[ "$1" == "reset" ]]; then
		echo "重置npu状态"
		echo "npu-smi set -t reset -i index -c 0"
	fi

	if [[ "$1" == "ping" ]]; then
		echo "检查网卡到其它IP连接"
		ping_ip = $2
		for i in $(seq 0 $rank_max); do hccn_tool -i $i -ping -g address $ping_ip pkt 128; done
	fi
}

impirun() {
	MPIRUN_PATH="/usr/local/mpich-4.1.3"

	export PATH=$MPIRUN_PATH/bin:$PATH
	export MANPATH=$MPIRUN_PATH/man:$MANPATH
	source $LXY_ASCEND_TOOLKIT/set_env.sh

	# 进入打流测试目录
	cd $LXY_ASCEND_TOOLKIT/latest/tools/hccl_test

	# 如果不存在all_reduce， 则开始编译
	if [[ ! -f "./bin/all_reduce_test" ]]; then
		export LD_LIBRARY_=$LD_LIBRARY_PATH:$LXY_ASCEND_TOOLKIT/latest/lib64:$MPI_PATH/lib/
		make MPI_HOME=$MPIRUN_PATH ASCEND_DIR=$LXY_ASCEND_TOOLKIT/latest
	fi

	# IFNAME是网卡名称的前缀，最好精确匹配
	export HCCL_SOCKET_IFNAME=eth
	export HCCL_SOCKET_FAMILY=AF_INET
	export HCCL_CONNECT_TIMEOUT=600
	export HCCL_BUFFSIZE=4096
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LXY_ASCEND_TOOLKIT/latest/aarch64-linux/lib64:$MPI_PATH/lib/

	npu_all_node=$1
	npu_per_node=$2
	echo "mpirun -f hostfile -n $npu_all_node ./bin/all_reduce_test -p $npu_per_node -b 8K -e 4096M -f 2 -d fp32 -o sum"
	mpirun -f hostfile -n $npu_all_node ./bin/all_reduce_test -p $npu_per_node -b 8K -e 4096M -f 2 -d fp32 -o sum
}

install_torch() {
    # 获取传入的torch版本
    local version="$1"
	# 获取系统的架构
    local arch=$(uname -m)

    # 检查版本是否为支持的版本
    if [[ "$version" != "2.1.0" && "$version" != "2.6.0" ]]; then
        echo "当前只支持torch 2.1.0和torch 2.6.0"
        return 1  # 返回非0状态码表示出错
    fi

    # 获取当前环境Python版本号
    PYTHON_VERSION=$(python -V 2>&1 | awk -F '[ .]' '{print $2$3}')
    
    # 安装对应版本的torch
    if [[ "$version" == "2.1.0" ]]; then
		# 安装torch 2.1.0
        pip install torch==2.1.0 torchvision==0.18.0 torchaudio==2.1.0 --extra-index https://download.pytorch.org/whl/cpu/

        # 安装对应版本的torch_npu
        if [[ "$arch" == x86* ]]; then
            torch_npu_url="https://gitee.com/ascend/pytorch/releases/download/v7.1.0.2-pytorch2.1.0/torch_npu-2.1.0.post16-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
            
        elif [[ "$arch" == arrch* ]]; then
            torch_npu_url="https://gitee.com/ascend/pytorch/releases/download/v7.1.0.2-pytorch2.1.0/torch_npu-2.1.0.post16-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
        else    
            # 可以添加对未知架构的处理
            echo "不支持的架构: $arch"
            return 1
        fi 

		# 开始安装torch_npu: $torch_npu_url
		pip install $torch_npu_url
    elif [[ "$verddsion" == "2.6.0" ]]; then
		# 安装torch
        pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index https://download.pytorch.org/whl/cpu/
        
        # 安装对应版本的torch_npu
        if [[ "$arch" == x86* ]]; then
            torch_npu_url="https://gitee.com/ascend/pytorch/releases/download/v7.1.0.2-pytorch2.6.0/torch_npu-2.6.0.post2-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
            
        elif [[ "$arch" == arrch* ]]; then
            torch_npu_url="https://gitee.com/ascend/pytorch/releases/download/v7.1.0.2-pytorch2.6.0/torch_npu-2.6.0.post2-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux_2_28_aarch64.whl"
        else    
            # 可以添加对未知架构的处理
            echo "不支持的架构: $arch"
            return 1
        fi   

		# 开始安装torch_npu: $torch_npu_url
		pip install $torch_npu_url
    fi
}

install_conda() {
	# 设置中文字符支持
	export LANG=C.UTF-8

	# 配置选项
	INSTALL_DIR="$HOME/miniconda3"
	CONDA_VERSION="latest"
	OS_ARCH="Linux-x86_64"

	echo "检测操作系统和架构..."
	OS_NAME=$(uname -s)
	ARCH_NAME=$(uname -m)

	if [[ "$ARCH_NAME" == "x86_64" ]]; then
		OS_ARCH="Linux-x86_64"
	elif [[ "$ARCH_NAME" == "aarch64" ]]; then
		OS_ARCH="Linux-aarch64"
	else
		echo "错误：不支持的Linux架构: $ARCH_NAME"
		return 1
	fi

	DOWNLOAD_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-${OS_ARCH}.sh"
	INSTALL_SCRIPT="Miniconda3-${CONDA_VERSION}-${OS_ARCH}.sh"
	echo "检测完成：操作系统=$OS_NAME, 架构=$ARCH_NAME, 安装包=$INSTALL_SCRIPT"

	# 检查是否有wget或curl
	wget -c --no-check-certificate "$DOWNLOAD_URL" -O "$INSTALL_SCRIPT"

	echo "开始静默安装Miniconda..."
	echo "安装目录: $INSTALL_DIR"

	# 给安装脚本添加执行权限
	chmod +x "$INSTALL_SCRIPT"

	# 静默安装Miniconda
	./"$INSTALL_SCRIPT" -b -p "$INSTALL_DIR"

	if [[ $? -ne 0 ]]; then
		echo "错误：Miniconda安装失败！"
		return 1
	fi

	echo "Miniconda安装成功！"

	# 初始化conda环境
	source "$INSTALL_DIR/etc/profile.d/conda.sh"

	# 备份原有配置
	CONDA_CONFIG="$HOME/.condarc"

	# 写入清华源配置
	cat > "$CONDA_CONFIG" << 'EOF'
channels:
- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
- https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
- defaults
channel_priority: flexible
show_channel_urls: true
EOF
		
	echo "conda源配置完成："
	cat "$CONDA_CONFIG"
}

install_aisbench() {
	echo 安装及文档：https://github.com/AISBench/benchmark
	git clone https://github.com/AISBench/benchmark.git
	cd benchmark/
	pip3 install -e ./ --use-pep517
	pip3 install -r requirements/api.txt 
	pip3 install -r requirements/extra.txt
	# 测试工具调用
	# pip3 install -r requirements/bfcl_dependencies.txt --no-deps
	# Berkeley Function Calling Leaderboard (BFCL) 测评支持
	pip3 install -r requirements/datasets/bfcl_dependencies.txt --no-deps
	# OCRBench_v2数据集测评支持（可选）
	# pip3 install -r requirements/datasets/ocrbench_v2.txt
	
	echo 修改配置
	echo sed -i 's/WORKERS_NUM = 0/WORKERS_NUM = 1/g' ais_bench/benchmark/global_consts.py

	echo 修改配置
	echo vim ais_bench/datasets/synthetic/synthetic_config.py

	echo 修改请求文件
	echo vim ./ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py

	echo 执行推理测试
	echo ais_bench --models vllm_api_general_stream --datasets synthetic_gen --mode perf --debug

	echo 卸载aisbench
	echo pip3 uninstall ais_bench_benchmark
}

install_bishengir() {
	echo 安装指导：https://gitcode.com/Ascend/AscendNPU-IR/blob/master/docs/source/zh_cn/introduction/quick_start/installing_guide.md
	git clone https://gitcode.com/Ascend/ascendnpu-ir.git
	cd ascendnpu-ir
	# 递归地拉取所有子模块
	git submodule update --init --recursive
	# 加载cann环境变量
	source /usr/local/Ascend/ascend-toolkit/set_env.sh
	# 在项目根目录下编译安装
	./build-tools/build.sh -o ./build --build-type Release --apply-patches
	# 重新构建，一般不用
	# ./build-tools/build.sh -o ./build --build-type Release -r
}


igetlocal() {
	# 1. 通过python在个人主机上启动服务，python -m http.server 5170
	# 2. 在Moba上建立“远程端口转发”类型的隧道，将远端服务器的5170映射到本机的5170
	# 3. 在Moba上使用wget或curl下载文件，例如：wget http://127.0.0.1:5170/your_file.txt
	
	wget http://127.0.0.1:5170/$1
}
__init_env
