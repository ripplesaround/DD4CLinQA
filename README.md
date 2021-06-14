## Info

题目：基于数据蒸馏的课程学习算法及其在问答中的应用

北京理工大学2021年毕业设计，严禁抄袭

---

使用方法：

运行命令

```
$python_path $file_path --model_name_or_path bert-base-uncased --dataset_name squad_v2 --do_train --do_eval --version_2_with_negative --learning_rate 3e-5 --num_train_epochs 4 --max_seq_length 384 --doc_stride 128 --output_dir /home/fwx/tmp/squadv2/exp_final_4/ --per_device_train_batch_size 16 --save_steps 5000 --overwrite_output_dir --DE_model /home/fwx/tmp/squadv2/exp_org/ --div_subset 3 
```

其中\$python_path为python的路径， \$file_path为file的路径

