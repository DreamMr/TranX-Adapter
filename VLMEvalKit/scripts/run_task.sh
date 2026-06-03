#!/bin/bash
export LMUData=YOUR_LMU_ROOT
export OPENAI_API_KEY=sk-testmyllm
export OPENAI_API_BASE=http://127.0.0.1:22003/v1/chat/completions
export GPU=$(nvidia-smi --list-gpus | wc -l)

WORKSPACE=../
cd $WORKSPACE
# Evaluation with LLaVA-Next on Chameleon, trained on GenImage All
EXPERIMENT_NAME="TranXAdapter-LLaVA-next-mistral7B-ChameleonAll"
MODEL_PATH="DreamMr/TranXAdapter-LLaVA-next-mistral7B-ChameleonAll"
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data Chamelon --model LLaVA-Next_expert_projector_sft_step --work-dir ./experiment_result/${EXPERIMENT_NAME} --model_path $MODEL_PATH

# Evaluation with LLaVA-Next on Chameleon, trained on GenImage SDv1.4
EXPERIMENT_NAME="TranXAdapter-LLaVA-next-mistral7B-ChameleonSd1d4"
MODEL_PATH="DreamMr/TranXAdapter-LLaVA-next-mistral7B-ChameleonSd1d4"
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data Chamelon --model LLaVA-Next_expert_projector_sft_step --work-dir ./experiment_result/${EXPERIMENT_NAME} --model_path $MODEL_PATH

# Evaluation with LLaVA-Next on RRDataset, trained on RRDataset_train
EXPERIMENT_NAME="TranXAdapter-LLaVA-next-mistral7B-RRDataset"
MODEL_PATH="DreamMr/TranXAdapter-LLaVA-next-mistral7B-RRDataset"
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data RR --model LLaVA-Next_expert_projector_sft_step --work-dir ./experiment_result/${EXPERIMENT_NAME} --model_path $MODEL_PATH

# Evaluation with LLaVA-Next on GenImage, trained on GenImage SDv1.4+BFP
EXPERIMENT_NAME="TranXAdapter-LLaVA-next-mistral7B-GenImage"
MODEL_PATH="DreamMr/TranXAdapter-LLaVA-next-mistral7B-GenImage"
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data GenImage --model LLaVA-Next_expert_projector_sft_step --work-dir ./experiment_result/${EXPERIMENT_NAME} --model_path $MODEL_PATH


# Evaluation with Qwen3VL-2B on Chameleon, trained on GenImage All
EXPERIMENT_NAME="TranXAdapter-Qwen3VL2B-ChameleonAll"
MODEL_PATH="DreamMr/TranXAdapter-Qwen3VL2B-ChameleonAll"
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data Chamelon --model Qwen3VL_NPR --work-dir ./experiment_result/${EXPERIMENT_NAME} --model_path $MODEL_PATH --judge chatgpt-0125 --api-nproc 4

# Evaluation with Qwen3VL-2B on Chameleon, trained on GenImage SDv1.4
EXPERIMENT_NAME="TranXAdapter-Qwen3VL2B-ChameleonSd1d4"
MODEL_PATH="DreamMr/TranXAdapter-Qwen3VL2B-ChameleonSd1d4"
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data Chamelon --model Qwen3VL_NPR --work-dir ./experiment_result/${EXPERIMENT_NAME} --model_path $MODEL_PATH --judge chatgpt-0125 --api-nproc 4

# Evaluation with Qwen3VL-2B on RRDataset, trained on RRDataset_train
EXPERIMENT_NAME="TranXAdapter-Qwen3VL2B-RRDataset"
MODEL_PATH="DreamMr/TranXAdapter-Qwen3VL2B-RRDataset"
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data RR --model Qwen3VL_NPR --work-dir ./experiment_result/${EXPERIMENT_NAME} --model_path $MODEL_PATH --judge chatgpt-0125 --api-nproc 4

# Evaluation with Qwen3VL-2B on GenImage, trained on GenImage SDv1.4+BFP
EXPERIMENT_NAME="TranXAdapter-Qwen3VL2B-GenImage"
MODEL_PATH="DreamMr/TranXAdapter-Qwen3VL2B-GenImage"
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data GenImage --model Qwen3VL_NPR --work-dir ./experiment_result/${EXPERIMENT_NAME} --model_path $MODEL_PATH --judge chatgpt-0125 --api-nproc 4


# Evaluation with Qwen3VL-4B on Chameleon, trained on GenImage All
EXPERIMENT_NAME="TranXAdapter-Qwen3VL4B-ChameleonAll"
MODEL_PATH="DreamMr/TranXAdapter-Qwen3VL4B-ChameleonAll"
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data Chamelon --model Qwen3VL_NPR --work-dir ./experiment_result/${EXPERIMENT_NAME} --model_path $MODEL_PATH --judge chatgpt-0125 --api-nproc 4

# Evaluation with Qwen3VL-4B on Chameleon, trained on GenImage SDv1.4
EXPERIMENT_NAME="TranXAdapter-Qwen3VL4B-ChameleonSd1d4"
MODEL_PATH="DreamMr/TranXAdapter-Qwen3VL4B-ChameleonSd1d4"
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data Chamelon --model Qwen3VL_NPR --work-dir ./experiment_result/${EXPERIMENT_NAME} --model_path $MODEL_PATH --judge chatgpt-0125 --api-nproc 4

# Evaluation with Qwen3VL-4B on RRDataset, trained on RRDataset_train
EXPERIMENT_NAME="TranXAdapter-Qwen3VL4B-RRDataset"
MODEL_PATH="DreamMr/TranXAdapter-Qwen3VL2B-ChameleonAll"
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data RR --model Qwen3VL_NPR --work-dir ./experiment_result/${EXPERIMENT_NAME} --model_path $MODEL_PATH --judge chatgpt-0125 --api-nproc 4

# Evaluation with Qwen3VL-4B on GenImage, trained on GenImage SDv1.4+BFP
EXPERIMENT_NAME="TranXAdapter-Qwen3VL4B-GenImage"
MODEL_PATH="DreamMr/TranXAdapter-Qwen3VL4B-GenImage"
torchrun --nproc-per-node=$GPU --master_port 29501 run.py --data GenImage --model Qwen3VL_NPR --work-dir ./experiment_result/${EXPERIMENT_NAME} --model_path $MODEL_PATH --judge chatgpt-0125 --api-nproc 4

