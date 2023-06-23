export WANDB_PROJECT=SysReview
export WANDB_ENTITY=lab-ai

python qlora.py \
    --model_name_or_path chaoyi-wu/PMC_LLAMA_7B \
    --output_dir ./outputs/PMC_test \
    --dataset ./data/instruct_cochrane_deduped.json \
    --do_train True \
    --source_max_len 1024 \
    --target_max_len 384 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --max_steps 100 \
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
