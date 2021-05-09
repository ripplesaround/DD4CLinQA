#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import copy
from Trianer_CL.Trainer_CL import Trainer_CL

# notice 制定GPU

import logging
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from geomloss import SamplesLoss

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
# from transformers.trainer_utils import  is_main_process
from transformers.trainer_utils import get_last_checkpoint, is_main_process
# from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.6.0.dev0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    # notice 添加的参数
    div_subset: Optional[int] = field(
        default=3,
        metadata={"help": "划分成几个subset"},
    )

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    DE_model: str = field(
        metadata={"help": "用于难度划分的模型"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

def DE_random(data_args,training_args, train_dataset):
    logger.info("***随机划分***")
    # subset_quantity = data_args.div_subset
    # n_train = len(train_dataset)
    # split = n_train // subset_quantity
    # indices = list(range(n_train))
    # random.shuffle(indices)
    # train_sampler = []
    # # 1,12,123
    # temp = []
    # for i in range(subset_quantity - 1):
    #     temp = []
    #     for j in range(i + 1):
    #         temp += indices[j * split: j * split + int(1 / 3 * split)]
    #     train_sampler.append(torch.utils.data.sampler.SubsetRandomSampler(temp))
    # train_sampler.append(torch.utils.data.sampler.SubsetRandomSampler(
    #     temp + indices[(subset_quantity - 1) * split: (subset_quantity - 1) * split + int(1 / 3 * split)])
    # )
    #
    # # 1 2 3
    # # for i in range(subset_quantity):
    # #     train_sampler.append(indices[i * split: i * split + int(1 / 3 * split)])
    #
    # result = []
    # for i in range(subset_quantity):
    #     result.append(DataLoader(train_dataset, sampler=train_sampler[i], batch_size=training_args.per_device_train_batch_size))

    subset_quantity = data_args.div_subset
    n_train = len(train_dataset)
    split = n_train // subset_quantity
    indices = list(range(n_train))
    random.shuffle(indices)
    result = []
    for i in range(data_args.div_subset):
        result.append(train_dataset.select(indices[i * split: i * split + int(1 / 3 * split)]))
        logger.info("第 %s 个subset的长度： %s",i,len((result[i])))

    # data_len = int(len(train_dataset) / data_args.div_subset)
    # result = random_split(train_dataset, [data_len,data_len, (len(train_dataset)-2*data_len) ], generator=torch.Generator().manual_seed(training_args.seed))
    # result[2] =  result[2].select(range(data_len))

    logger.info("***随机划分完成***")
    return result

def cal_diff(x, y, norm="org", criterion ="wd" ):
    if norm == "softmax":
        x = F.softmax(x)
        y = F.softmax(y)
    elif norm == "logsoftmax":
        x = F.log_softmax(x)
        y = F.log_softmax(y)
    elif norm == "line":
        # logger.info("使用线性归一化")
        x = linear_normalization(x)
        y = linear_normalization(y)
    elif norm == "Gaussian":
        z = 1
        # 实现高斯分布
        # transform_BZ = transforms.Normalize(
        #     mean=[0.5, 0.5, 0.5],  # 取决于数据集
        #     std=[0.5, 0.5, 0.5]
        # )
        # logger.info("使用高斯分布归一化")

    # 每个batch一起算
    # KLloss = criterion(x, y)
    # return KLloss.item()

    # 每个batch 内单独算，最后算一个和
    dim0 = x.shape[0]
    result = 0.0
    blur = .05
    OT_solver = SamplesLoss("sinkhorn", p=2, blur=blur,
                            scaling=.9, debias=False, potentials=True)
    for i in range(dim0):
        if criterion == "kl":
            criterion_kl = nn.KLDivLoss()
            # notice 考虑了KL的不对称性
            KLloss = (criterion_kl(x[i], y[i])+criterion_kl(y[i], x[i]))/2
            result += KLloss.item()
        else:
            # change wgan
            F_i, G_j = OT_solver(x[i], y[i])
            # # print("F_i ",torch.sum(F_i).item())
            result += (torch.sum(F_i).item())
            # print("hi")
    return result / dim0

def linear_normalization(x):
    temp_min = torch.min(x)
    temp_max = torch.max(x)
    x = (x-temp_min)/(temp_max-temp_min)
    return x

def DE(trainer,train_dataset,training_args,data_args):
    logger.info("***难度评估开始***")
    total_train_dataloader = trainer.get_train_dataloader()
    for i,item in enumerate(total_train_dataloader):
        print(item)
        if i>0:
            break
    print("--------")
    for i,item in enumerate(total_train_dataloader.sampler):
        print(item)
        if i>0:
            break

    difficult_result = []
    # notice 划分方法
    method = "org"
    criterion = "wd"
    logger.info("划分方法 " + method + "   " + criterion)
    for inputs in tqdm(total_train_dataloader):
        # cuda error 证明算法写错了
        # inputs = inputs.to(training_args.device)
        # print(inputs)
        # print(inputs["idx"].tolist())
        idx = inputs.pop("idx")
        # sentences = inputs.pop("sentence")
        for key in inputs:
            inputs[key] = inputs[key].to(training_args.device)
        output = trainer.compute_loss(
            model=trainer.model,
            inputs=inputs,
            return_outputs=True
        )
        # output 的第一项是loss 第二项是SequenceClassifierOutput
        # SequenceClassifierOutput 是 这个（https://huggingface.co/transformers/model_doc/bert.html）
        # if return_dict=True is passed or when config.return_dict=True) or a tuple of torch.FloatTensor comprising various elements depending on the configuration (BertConfig) and inputs.
        # 默认返回的是一个tuple 分别是：loss，logit，hidden-states, attentions
        # hidden state的第一项是embedding的输出，后面是每一层的输出
        # hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) – Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
        # 所以这里选用 output[1][2][0] 来访问hidden-state
        difficult_result.append( cal_diff(output[1][2][0],output[1][2][-1],norm = method,criterion=criterion))

        # if i>15:
        #     break


    difficult_result = np.array(difficult_result)
    logger.info("dic len {len1}".format(len1=len(difficult_result)))

    difficult_result_max = max(difficult_result)
    difficult_result_min = min(difficult_result)
    gap = difficult_result_max - difficult_result_min

    subset = []
    total_len = 0
    for i in range(data_args.div_subset):
        subset.append([])
    for i, batch in enumerate(total_train_dataloader):
        if difficult_result[i] == difficult_result_max:
            subset[-1] += batch["idx"].tolist()
            total_len += len(batch["idx"].tolist())
            continue
        level = int(data_args.div_subset * (difficult_result[i] - difficult_result_min) / gap)
        subset[level] += batch["idx"].tolist()
        total_len += len(batch["idx"].tolist())
        # print("batch")
        # print(batch["idx"].tolist())
        # if i>15:
        #     break
    # 不能直接用squad中的方法，因为这里需要返回的不是一个dataloader的合集，而是一个dataset的合集
    # 所以必须要用 idx
    print("total_len {e}".format(e = total_len))
    # for i in range(data_args.div_subset):
    #     print(subset[i])

    # notice 进行采样
    # 模拟退火
    dd = []
    for i in range(data_args.div_subset):
        sample_num = (len(subset[i])) // data_args.div_subset
        dd += random.sample(subset[i],sample_num)
        subset[i] = train_dataset.select(dd)
        print(len(subset[i]))

    # 顺序
    # dd = []
    # for i in range(data_args.div_subset):
    #     sample_num = (len(subset[i])) // data_args.div_subset
    #     dd += random.sample(subset[i], sample_num)
    #     subset[i] = train_dataset.select(dd)
    #     print(len(subset[i]))

    logger.info("***难度评估结束***")

    # notice 释放缓存
    torch.cuda.empty_cache()
    return subset

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    # notice 这里默认的seed是42 ，需要我们自己重新设置seed
    set_seed(training_args.seed)
    # set_seed(int(time.time()))

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    # if training_args.do_train:
    #     for index in random.sample(range(len(train_dataset)), 3):
    #         logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None




    total_num_train_epochs = training_args.num_train_epochs
    # training_args._n_gpu = 1
    training_args.per_device_eval_batch_size = 1

    # notice Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 用于难度划分的trainer
    DE_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.DE_model,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    trainer_DE = Trainer_CL(
        model=DE_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # notice
    # 测试DE
    # curr_subset = DE(trainer_DE, train_dataset, training_args, data_args)

    # 进行随机处理
    curr_subset = DE_random(data_args, training_args, train_dataset)

    # training_args_curr = training_args
    training_args_curr = copy.deepcopy(training_args)
    training_args_curr.num_train_epochs = 1
    # training_args_curr.per_device_eval_batch_size
    trainer_curr = []
    for i in range(data_args.div_subset):
        trainer_curr.append(
            Trainer(
                model=model,
                args=training_args_curr,
                train_dataset=curr_subset[i] if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
        )

    # Training
    if training_args.do_train:


        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        # notice 这里直接讲train打包了，需要继承train方法

        # for i in range(data_args.div_subset):
        #     logger.info("******* 开始课程训练 *******")
        #     train_result = trainer_curr[i].train(resume_from_checkpoint=checkpoint)
        #     checkpoint = training_args.output_dir
        #     trainer_curr[i].save_model()
        #     logger.info("curr   save model at " + training_args.output_dir)

        # 还要修改一下epoch，搞成很多个train的方式
        # for i in range(int(total_num_train_epochs)):
        #     logger.info("num_train_epochs --- " + str(training_args.num_train_epochs))
        #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
        #     checkpoint = training_args.output_dir
        #     trainer.save_model()
        #     logger.info("save model at "+ training_args.output_dir)

        logger.info("num_train_epochs --- " + str(training_args.num_train_epochs))
        logger.info("curr num_train_epochs --- " + str(training_args_curr.num_train_epochs))
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        checkpoint = training_args.output_dir
        trainer.save_model()
        logger.info("save model at " + training_args.output_dir)
        logger.info("trainer.state.global_step = %s",trainer.state.global_step)
        # 归0
        trainer.state.global_step = 0
        trainer.save_state()

        torch.cuda.empty_cache()

        for i in range(data_args.div_subset):
            logger.info("******* 开始课程训练 *******")
            train_result = trainer_curr[i].train(resume_from_checkpoint=checkpoint)
            checkpoint = training_args.output_dir
            trainer_curr[i].save_model()
            logger.info("curr   save model at " + training_args.output_dir)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer_curr[data_args.div_subset - 1].log_metrics("train", metrics)
        trainer_curr[data_args.div_subset - 1].save_metrics("train", metrics)
        trainer_curr[data_args.div_subset - 1].save_state()
        # for i in range(data_args.div_subset):

        torch.cuda.empty_cache()


    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            torch.cuda.empty_cache()

            metrics = trainer_curr[data_args.div_subset - 1].evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            torch.cuda.empty_cache()
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer_curr[data_args.div_subset - 1].log_metrics("eval", metrics)
            trainer_curr[data_args.div_subset - 1].save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()