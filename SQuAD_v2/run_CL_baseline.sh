CUDA_VISIBLE_DEVICES=1 python3.7 -u /home/fwx/project/dd_2020/SQuAD_v2/run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad_v2 \
  --do_train \
  --do_eval  \
  --version_2_with_negative \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir /home/fwx/tmp/squadv2_final/exp_final_1    \
  --per_device_train_batch_size 16 \
  --save_steps 5000 \
  --overwrite_output_dir \

