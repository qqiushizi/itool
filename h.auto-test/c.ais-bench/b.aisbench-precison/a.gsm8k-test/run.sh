# 运行测试时会出现以下日志，第一行中的py文件是服务请求需要配置的文件，第二行是请求模板文件
#[2026-07-30 11:33:30,068] [ais_bench] [INFO] Loading vllm_api_general_chat: /root/miniforge3/envs/ais-bench/lib/python3.11/site-packages/ais_bench/benchmark/configs/./models/vllm_api/vllm_api_general_chat.py
#[2026-07-30 11:33:30,073] [ais_bench] [INFO] Loading demo_gsm8k_gen_4_shot_cot_chat_prompt: /root/miniforge3/envs/ais-bench/lib/python3.11/site-packages/ais_bench/benchmark/configs/./datasets/demo/demo_gsm8k_gen_4_shot_cot_chat_prompt.py

export AIS_BENCH_DATASETS_CACHE=/workspace
mkdir -p $AIS_BENCH_DATASETS_CACHE/ais_bench/datasets
cd $AIS_BENCH_DATASETS_CACHE/ais_bench/datasets

if [ ! -f "./gsm8k.zip" ]; then
	wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip
	unzip gsm8k.zip
fi;
#rm gsm8k.zip

ls -al $AIS_BENCH_DATASETS_CACHE/ais_bench/datasets/gsm8k

ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug
