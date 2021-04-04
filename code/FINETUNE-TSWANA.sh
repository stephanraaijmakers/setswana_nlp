python3.7 run_ner.py \
	  --data_dir "./ner_tagged/preprocessed" \
	  --model_type bert \
	  --labels "./ner_tagged/preprocessed/labels.txt" \
	  --model_name_or_path "./tswana_models/output" \
	  --output_dir "./tswana_models/output/finetuned" \
	  --num_train_epochs 3 \
	  --per_gpu_train_batch_size 32 \
	  --save_steps 100 \
	  --logging_steps 100 \
	  --seed 42 \
	  --do_train \
	  --do_eval \
	  --do_predict \
	  --overwrite_output_dir



