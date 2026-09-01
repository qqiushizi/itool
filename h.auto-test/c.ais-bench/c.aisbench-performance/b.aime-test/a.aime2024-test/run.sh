# 运行测试时会出现以下日志，第一行中的py文件是服务请求需要配置的文件，第二行是请求模板文件
#[2026-07-30 11:33:30,068] [ais_bench] [INFO] Loading vllm_api_general_chat: /root/miniforge3/envs/ais-bench/lib/python3.11/site-packages/ais_bench/benchmark/configs/./models/vllm_api/vllm_api_general_chat.py
#[2026-07-30 11:33:30,073] [ais_bench] [INFO] Loading demo_gsm8k_gen_4_shot_cot_chat_prompt: /root/miniforge3/envs/ais-bench/lib/python3.11/site-packages/ais_bench/benchmark/configs/./datasets/demo/demo_gsm8k_gen_4_shot_cot_chat_prompt.py

export AIS_BENCH_DATASETS_CACHE=/workspace
mkdir -p $AIS_BENCH_DATASETS_CACHE/ais_bench/datasets
cd $AIS_BENCH_DATASETS_CACHE/ais_bench/datasets

if [ ! -f "/workspace/ais_bench/datasets/aime/aime.jsonl" ]; then
	modelscope download --dataset vllm-ascend/aime2024 --local_dir=/workspace/ais_bench/datasets/aime
	ls -al /workspace/ais_bench/datasets/aime
fi;

ais_bench --models vllm_api_general_chat --datasets aime2024_gen_0_shot_chat_prompt.py --summarizer default_perf --mode perf
