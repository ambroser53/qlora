export WANDB_PROJECT=SysReview
export WANDB_ENTITY=lab-ai

python qlora.py \
    --model_name_or_path microsoft/BioGPT-Large \
    --output_dir ./outputs/SysRev1_3b \
    --dataset ./data/instruct_cochrane_deduped.json \
    --do_train True \
    --source_max_len 1024 \
    --target_max_len 384 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --logging_steps 10 \
    --max_steps 6250 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 1000 \
    --save_total_limit 40 \
    --optim paged_adamw_32bit \
    --without_trainer_checkpoint True \
    --force_lora_training True \
    --lora_bias lora_only \
    --report_to wandb \
    --run_name SysRev1_3b \
    --accelerate_method "cuda:0" \
    --max_memory_MB=24000
