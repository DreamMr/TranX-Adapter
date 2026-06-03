export OMP_NUM_THREADS=32

output_dir=./checkpoints

# Noted: modify the resolution to (512x512)
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model DreamMr/TranXAdapter-Qwen3VL2B-ChameleonAll \
    --model_type qwen3_vl_aigc_detection_npr \
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
    --output_dir ${output_dir}/TranXAdapter-Qwen3VL2B-RRDataset \
    --save_total_limit 10 \
    --seed 0 \
    --eval_strategy no \
    --save_steps 50 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 2 \
    --lazy_tokenize true