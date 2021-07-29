/home/fwx/anaconda3/envs/py37/bin/python3.7 -u /home/fwx/project/dd_2020/SQuAD_v2/run_qa_CL_final_quantum.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad_v2 \
  --do_train \
  --do_eval  \
  --version_2_with_negative \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /home/fwx/tmp/squadv2/exp_final_qu_1    \
  --per_device_train_batch_size 16 \
  --save_steps 5000 \
  --overwrite_output_dir \
  --DE_model /home/fwx/tmp/squadv2/exp_org/ \
  --div_subset 3

