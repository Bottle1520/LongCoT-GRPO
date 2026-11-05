Reasoning-GRPO

基于 Qwen2.5-1.5b 的轻量级长思维链数学推理微调项目。项目采用“冷启动监督微调（SFT）+ 群体相对策略优化（GRPO）”两阶段训练流程，并使用规则验证器对模型生成的答案进行评分。

一、项目流程

1. 数据准备：将数学题目与参考解答整理成统一的 JSONL 格式。
2. 冷启动 SFT：通过自定义 LoRA，将目标线性层替换为 LoRA 层，只训练低秩适配参数。
3. GRPO 强化学习：使用 vLLM 采样多条回答，根据组内奖励计算优势，并更新策略模型。
4. 规则验证：解析回答中的 <think> 与 <answer> 标签，并根据参考答案判断正确性，作为 GRPO 奖励。

二、目录结构

Reasoning-GRPO/

├── README.md

├── requirements.txt

├── setup.py

├── configs/

│   ├── sft_config.yaml

│   └── grpo_config.yaml

├── data/

│   ├── prepare_sft.py

│   └── dataset_loaders.py

├── verifier/

│   ├── math_parser.py

│   └── math_evaluator.py

├── sft/

│   ├── lora.py

│   └── train_sft.py

├── rl/

│   ├── rollout_engine.py

│   ├── grpo_trainer.py

│   └── train_rl.py

└── utils/

|    └── logger.py

三、环境依赖

建议使用支持项目所需 PyTorch、Transformers 和 vLLM 的 Python 环境。安装项目依赖：

pip install -r requirements.txt

主要依赖包括 PyTorch、Transformers、vLLM（用于加速生成）和 SymPy（用于数学表达式等价性判断）。具体版本请以 requirements.txt 为准。使用 vLLM 时，还需确保 CUDA、PyTorch 与 vLLM 版本兼容。

四、数据格式

SFT 数据

prepare_sft.py 接受 JSON 或 JSONL 数据。题目字段支持 question、prompt 或 problem；答案字段支持 answer、solution 或 response。

整理后的 JSONL 每行包含：

{"prompt":"计算 12 × 13。","response":"<think>将 12 乘以 13，得到 156。</think><answer>156</answer>","source_id":"example-1"}

运行数据整理脚本：

python data/prepare_sft.py input.jsonl data/processed/sft.jsonl

若输入答案没有 <answer> 标签，脚本会按当前实现补充 <think> 和 <answer> 标签。请检查整理后的样本格式及答案内容，再用于训练。

GRPO 数据

GRPO 训练数据为 JSONL，每行至少需要包含 prompt 和参考答案字段 reference_answer 或 answer：

{"prompt":"计算 12 × 13。","reference_answer":"156"}

五、冷启动 SFT

示例命令：

python -m sft.train_sft --model Qwen/Qwen2.5-1.5B-Instruct --data data/processed/sft.jsonl --output checkpoints/sft-lora

默认会尝试将 LoRA 注入以下线性层：q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj。可通过 --targets、--rank、--alpha、--dropout 等参数调整 LoRA 配置。训练完成后，输出目录保存 LoRA adapter 权重与配置；使用 adapter 时需要加载对应的基座模型并恢复 LoRA 权重。

六、GRPO 强化学习

示例命令：

python -m rl.train_rl --model checkpoints/sft-lora --data data/processed/grpo.jsonl --output checkpoints/grpo --batch-size 4 --generations 4 --steps 100

训练入口使用 rollout engine 为每个题目生成多条回答，调用规则验证器计算奖励，并进行 GRPO 更新。--no-vllm 可切换到 Transformers 生成路径：

python -m rl.train_rl --model MODEL_PATH --data data/processed/grpo.jsonl --output checkpoints/grpo --no-vllm

七、回答格式与奖励

规则验证器识别以下结构：

<think>
推理过程
</think>
<answer>
最终答案
</answer>

当前实现主要根据最终答案是否与参考答案等价评分，并对包含 <answer> 标签的回答提供格式奖励。表达式比较会优先尝试使用 SymPy 化简，解析失败时回退到去除空白后的字符串比较。

