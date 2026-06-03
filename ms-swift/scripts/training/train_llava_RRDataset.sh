export OMP_NUM_THREADS=20
output_dir=./checkpoints
NPROC_PER_NODE=8 \

# Noted: modify the resolution to (512x512)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model DreamMr/TranXAdapter-LLaVA-next-mistral7B-ChameleonAll \
    --model_type llava1_6_mistral_hf_npr_expert_projector \
    --num_train_epochs 15 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --train_type full \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --freeze_vit_expert true \
    --freeze_aligner_expert false \
    --freeze_llm true \
    --freeze_aligner true \
    --max_length 8192 \
    --deepspeed zero2 \
    --dataset ./Dataset/TranXAdapter-Dataset/training/RRDataset_processed.jsonl \
    --output_dir ${output_dir}/TranXAdapter-LLaVA-next-mistral7B-RRDataset \
    --save_total_limit 5 \
    --seed 0 \
    --eval_strategy no \
    --save_steps 50 \
    --dataset_num_proc 8