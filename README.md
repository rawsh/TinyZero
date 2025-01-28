- Built on veRL: https://github.com/volcengine/verl
- Forked from TinyZero: https://github.com/Jiayi-Pan/TinyZero
- Added code + math verifiers from PRIME https://github.com/PRIME-RL/PRIME
- Applied [liger kernels](https://github.com/linkedin/Liger-Kernel) for Qwen2


See `workers/fsdp_workers.py`:

```python
# Apply Liger kernel optimizations to Qwen2 model
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
apply_liger_kernel_to_qwen2(
    rope=False,
    cross_entropy=False,
    fused_linear_cross_entropy=True,
    rms_norm=True,
    swiglu=True
)
```

Key fix for OOMs with long sequences is [veRL dynamic batch size](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html#dynamic-batch-size-tuning-tips)

See `scripts/train.sh`:

```
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18000 \
++actor_rollout_ref.rollout.use_dynamic_bsz=True \
++actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=18000 \
++actor_rollout_ref.ref.use_dynamic_bsz=True \
++actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=18000 \
++critic.use_dynamic_bsz=True \
++critic.ppo_max_token_len_per_gpu=18000 \
```

veRL also [supports deepspeed ulysses](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html#ulysses-sequence-parallel-for-long-context-training)

> To utilize this technique, users can set `ulysses_sequence_parallel_size>1` in actor, ref, critic and reward models.

Settings to reduce memory

```
critic.model.enable_gradient_checkpointing=True \
critic.model.use_remove_padding=True \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.grad_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.ref.fsdp_config.param_offload=False \
```

With larger models you can use `ROLLOUT_TP_SIZE > 1`

For tuning batch size, note that `*_micro_batch_size` must be >= world size

To train:

1. Follow Installation
2. Clean and load dataset

This script filters examples with prompts longer than 1k tokens and removes code questions with empty test cases.

dataset: https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data

```
python ./examples/data_preprocess/eurus_dataset.py --local_dir {path_to_your_dataset}
```

3. Run training

to train deepseek-r1-distill-1.5B with 4xh100:

```
export N_GPUS=4
export BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
export CRITIC_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
export DATA_DIR=/home/ubuntu/data/filtered/
export EXPERIMENT_NAME=eurus-deepseek
export VLLM_ATTENTION_BACKEND=XFORMERS
export ROLLOUT_TP_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Note that we use transformers attention backend due to flashattn bug training qwen2 models: https://github.com/volcengine/verl/issues/12

Should be fixed once vLLM == 0.6.4 is supported [(comment)](https://github.com/volcengine/verl/issues/12#issuecomment-2543870798)

Original readme:

---

## Installation

```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## Countdown task

**Data Preparation**
```
conda activate zero
python ./examples/data_preprocess/countdown.py --local_dir {path_to_your_dataset}
```

### Run Training
```
conda activate zero
```

For the following code, if you see Out-of-vram, try add `critic.model.enable_gradient_checkpointing=True` to the script

**Single GPU**


Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning.

```
export N_GPUS=1
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

**3B+ model**
In this case, the base model is able to develop sophisticated reasoning skills.
```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

### Instruct Ablation
We experiment with QWen-2.5-3B Instruct too.
**Data Preparation**
To follow chat template, we need to reprocess the data:
```
conda activate zero
python examples/data_preprocess/countdown.py --template_type=qwen-instruct --local_dir={path_to_your_dataset}
```

**Training**
```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation
```
@misc{tinyzero,
author       = {Jiayi Pan and Junjie Zhang and Xingyao Wang and Lifan Yuan},
title        = {TinyZero},
howpublished = {https://github.com/Jiayi-Pan/TinyZero},
note         = {Accessed: 2025-01-24},
year         = {2025}
}
```
