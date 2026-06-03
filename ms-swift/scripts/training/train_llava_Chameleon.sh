export OMP_NUM_THREADS=20
output_dir=./checkpoints
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model DreamMr/TranXAdapter-LLaVA-next-mistral7B-v0 \
    --model_type llava1_6_mistral_hf_npr_expert_projector \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.03 \
    --per_device_train_batch_size 32 \
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
    --dataset ./Dataset/TranXAdapter-Dataset/training/GenImage_Sdv1d4_processed.jsonl \
    --output_dir ${output_dir}/TranXAdapter-LLaVA-next-mistral7B-ChameleonSd1d4 \
    --save_total_limit 5 \
    --seed 0 \
    --eval_strategy no \
    --save_steps 1000 \
    --dataset_num_proc 8

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model DreamMr/TranXAdapter-LLaVA-next-mistral7B-v0 \
    --model_type llava1_6_mistral_hf_npr_expert_projector \
    --num_train_epochs 2 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.03 \
    --per_device_train_batch_size 32 \
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
    --dataset ./Dataset/TranXAdapter-Dataset/training/GenImageAll_processed.jsonl \
    --output_dir ${output_dir}/TranXAdapter-LLaVA-next-mistral7B-ChameleonAll \
    --save_total_limit 20 \
    --seed 0 \
    --eval_strategy no \
    --save_steps 500 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 12 \
    --dataloader_prefetch_factor 2 \
    --lazy_tokenize true


