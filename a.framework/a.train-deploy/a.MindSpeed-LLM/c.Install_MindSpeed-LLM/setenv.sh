CONDA_HOME=/root/miniforge3  # 改成你的conda路径
TARGET_ENV=mindspeed-llm

# 官方标准hook初始化，兼容交互/非交互shell
__conda_setup="$(${CONDA_HOME}/bin/conda 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_HOME}/etc/profile.d/conda.sh"
    else
        export PATH="${CONDA_HOME}/bin:$PATH"
    fi
fi
unset __conda_setup

# 现在再激活环境就不会报错
conda activate $TARGET_ENV

MindSpeed_LLM_PATH=/workspace/MindSpeed-LLM
Model_Name=Qwen/Qwen3-0.6B
Prompt_Type=qwen3
