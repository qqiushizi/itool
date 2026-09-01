# 运行测试时会出现以下日志，第一行中的py文件是服务请求需要配置的文件，第二行是请求模板文件
#[2026-07-30 11:33:30,068] [ais_bench] [INFO] Loading vllm_api_general_chat: /root/miniforge3/envs/ais-bench/lib/python3.11/site-packages/ais_bench/benchmark/configs/./models/vllm_api/vllm_api_general_chat.py
#[2026-07-30 11:33:30,073] [ais_bench] [INFO] Loading demo_gsm8k_gen_4_shot_cot_chat_prompt: /root/miniforge3/envs/ais-bench/lib/python3.11/site-packages/ais_bench/benchmark/configs/./datasets/demo/demo_gsm8k_gen_4_shot_cot_chat_prompt.py

export AIS_BENCH_DATASETS_CACHE=/workspace

# 切到脚本所在目录，保证 ais_bench 的输出目录(outputs)落在可写位置
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_DIR="$AIS_BENCH_DATASETS_CACHE/ais_bench/datasets"
mkdir -p "$DATA_DIR"

if [ ! -d "$DATA_DIR/textvqa" ]; then
	git lfs install
	(
		cd "$DATA_DIR"
		git clone https://huggingface.co/datasets/maoxx241/textvqa_subset
		mv textvqa_subset/ textvqa/
		mkdir textvqa/textvqa_json/
		mv textvqa/*.json textvqa/textvqa_json/
		mv textvqa/*.jsonl textvqa/textvqa_json/

		cd textvqa/textvqa_json
		sed -i 's#data/textvqa/train_images/#/workspace/ais_bench/datasets/textvqa/train_images/#g' textvqa_val.jsonl
	)
	ls -al "$DATA_DIR/textvqa"
	ls -al "$DATA_DIR/textvqa/textvqa_json"
fi

ais_bench --models vllm_api_stream_chat --datasets textvqa_gen_base64 --summarizer default_perf --mode perf
