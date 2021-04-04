
python3.7 run_language_modeling.py \
    --output_dir "./tswana_models/output" \
    --model_type "roberta" \
    --mlm \
    --config_name "./tswana_models" \
    --tokenizer_name "./tswana_models" \
    --train_data_file "./tswana_data/train.txt" \
    --eval_data_file "./tswana_data/dev.txt" \
    --do_train \
    --overwrite_output_dir \
    --block_size 512 \
    --max_step 25 \
    --warmup_steps 10 \
    --learning_rate 5e-5 \
    --per_gpu_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 100.0 \
    --save_total_limit 10 \
    --save_steps 10 \
    --logging_steps 2 \
    --seed 42

