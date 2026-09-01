ROOT_PATH=/workspace
MindSpeed_MM_PATH=$ROOT_PATH/MindSpeed-MM

cd $ROOT_PATH

echo ======= 开始安装MindSpeed-MM =========
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd $MindSpeed_MM_PATH

bash scripts/install.sh --msid eb10b92 && bash examples/qwen3_5/install_extensions.sh
pip install -U mistral_common==1.11.0

