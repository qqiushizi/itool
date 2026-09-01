# 运行测试时会出现以下日志，第一行中的py文件是服务请求需要配置的文件，第二行是请求模板文件
#[2026-07-30 11:33:30,068] [ais_bench] [INFO] Loading vllm_api_general_chat: /root/miniforge3/envs/ais-bench/lib/python3.11/site-packages/ais_bench/benchmark/configs/./models/vllm_api/vllm_api_general_chat.py
#[2026-07-30 11:33:30,073] [ais_bench] [INFO] Loading demo_gsm8k_gen_4_shot_cot_chat_prompt: /root/miniforge3/envs/ais-bench/lib/python3.11/site-packages/ais_bench/benchmark/configs/./datasets/demo/demo_gsm8k_gen_4_shot_cot_chat_prompt.py

export AIS_BENCH_DATASETS_CACHE=/workspace
mkdir -p $AIS_BENCH_DATASETS_CACHE/ais_bench/datasets
cd $AIS_BENCH_DATASETS_CACHE/ais_bench/datasets

if [ ! -d "$AIS_BENCH_DATASETS_CACHE/ais_bench/datasets/ceval/formal_ceval/" ]; then
	mkdir ceval/
	mkdir ceval/formal_ceval
	cd ceval/formal_ceval
	wget https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip
	unzip ceval-exam.zip
	rm ceval-exam.zip
fi;

ais_bench --models vllm_api_general_chat --datasets ceval_gen_0_shot_cot_chat_prompt.py --mode all --dump-eval-details --merge-ds --debug

#测试结果精度为0问题 分析完成。根因已完全确认。
#
#  ## 根因：max_out_len=512 过小，导致所有输出在写出 \boxed{} 答案前被截断
#
#  完整证据链如下：
#
#  配置问题
#  模型配置文件 vllm_api_general_chat.py:25 设置了 max_out_len=512，而该参数类的默认值是 4096。这个值经 vllm_custom_api_chat.py:121 直接映射为 API 请求的 max_tokens=512，即 vLLM 服务端（glm-
#  5 @ 30.249.16.61:5172）最多只生成 512 个 token。
#
#  所有 30 条预测均被截断
#  检查 predictions/vllm-api-general-chat/aime2026.jsonl，30 条预测全部在推理中途戛然而止：
#
#  - id=0 结尾：$D = v \cdot T = \（公式写到一半）
#  - id=1 结尾：sequences of length $k$ of（句子未完）
#  - id=29 结尾：3: 3, 4, 6\n（列表中断）
#  - 没有任何一条预测正常结束
#
#  0/30 预测包含 \boxed{}
#  prompt 明确要求"Put your answer inside \boxed{}"，但因 512 token 上限，模型来不及推导完就断了。30 条预测中含 \boxed{} 的数量为 0。
#
#  评测器无法提取答案 → 精度 0
#  math_postprocess_v2 首先调用 extract_boxed_answer(text)——无 \boxed{} 时返回 None；然后回退查找"final answer / answer is"（截断的推理中也没有）；最终回退到取第一句话片段（如"The problem
#  asks for the distance..."），这与整型金标答案（277、62、156…）永远无法匹配。MATHEvaluator 计算正确数 = 0/30 → accuracy: 0.00。
#
#  模型本身能力没问题（关键反证）
#  id=0 的预测中，模型已正确推导出 v=18/5、T=14/5，正要计算 D = 18/5 × 14/5 = 252/25，而 m+n = 252+25 = 277 正是金标答案——但输出在 $D = v \cdot T = \ 处被 512 token 截断，未能写出
#  \boxed{277}。模型会做，只是被 token 上限掐断了。
#
#  次要观察（非根因）
#  配置中 pred_postprocessor=extract_non_reasoning_content 期望模型输出  visitors... visitors 推理块。但当前 glm-5 的输出是纯 CoT 文本（无这些标签），该后处理器实际是空操作，不影响当前结果。
#  增大 token 后若模型启用推理模式，该后处理器会正确提取  visitors 后的最终答案，需留意 \boxed{} 应在推理块之外。
#
#  修复方法
#  将 max_out_len 从 512 调大。AIME 是竞赛级数学题，推理链长，建议设为 8192（glm-5 若为推理模型，思考阶段也消耗 token，需更大余量）。需修改的文件：
#
#  /root/miniforge3/envs/ais-bench/lib/python3.11/site-packages/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py:25（max_out_len=512 → max_out_len=8192）
