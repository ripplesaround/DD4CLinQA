#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric
import random
import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pyarrow as pa
from geomloss import SamplesLoss
import torch
from datasets.arrow_dataset import update_metadata_with_features
import copy
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.7.0.dev0")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()

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

def change_dataset(temp_dataset, add_column="idx"):
    """
    为数据集增加列
    :param dataset: 一般为训练数据集，用作划分
    :param add_column: 增加的列的名称
    :return: 增加idx后的数据集
    """
    logger.info("开始为数据集添加 "+ add_column)
    add_dataset = copy.deepcopy(temp_dataset)
    logger.info("添加前的列： " + " ".join(add_dataset.column_names))
    print(len(add_dataset))
    print(add_dataset._info.features)
    # print(add_dataset[20])
    # add_info = np.array(range(len(add_dataset)))
    add_info = pa.array(range(len(add_dataset)))
    # add_info = "test"
    add_dataset._data = add_dataset._data.append_column("idx",add_info)
    # dataset._data = update_metadata_with_features(dataset._data, self.features)
    # print(add_dataset._info.features)
    add_dataset._data = update_metadata_with_features(add_dataset._data, add_dataset._info.features)
    logger.info("添加后的列： " + " ".join(add_dataset.column_names))
    print(len(add_dataset))
    # print(add_dataset[10])
    # print("hee")
    return add_dataset

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
    cnt=0
    for inputs in tqdm(total_train_dataloader):
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



        # output 为 QuestionAnsweringModelOutput
        # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.QuestionAnsweringModelOutput
        # 和之前的不一样
        # print(type(output[1]))
        # print(len(output[1]))
        # print(len(output[1].hidden_states))
        # print(output[1].hidden_states[0].shape)
        # print(output[1].__dir__())
        # print(output[1])
        # print(output[1].encoder_hidden_states)
        # print(output[1].encoder_hidden_states.shape)
        # print(output.encoder_hidden_states[0].shape)
        # print("output[1].shape: ",output[1].shape )
        # print("output[1][2].shape: ", output[1][2].shape)
        # print("output[1][2][-1].shape: ", output[8][0].shape)
        difficult_result.append(cal_diff(output[1].hidden_states[0],output[1].hidden_states[-1],norm = method,criterion=criterion))
        # cnt +=1
        # if cnt>1:
        #     sys.exit(100)

    difficult_result = np.array(difficult_result)
    logger.info("dic len {len1}".format(len1=len(difficult_result)))

    difficult_result_max = max(difficult_result)
    difficult_result_min = min(difficult_result)
    gap = difficult_result_max - difficult_result_min

    subset = []
    total_len = 0
    for i in range(data_args.div_subset):
        subset.append([])
    # 拿到idx为了后续的划分
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
    threshold = ((total_len) * data_args.div_subset) // training_args.per_device_train_batch_size
    for i in range(data_args.div_subset):
        #数据蒸馏部分
        if len(subset[i]) > threshold:
            sample_num = ((len(subset[i])) // data_args.div_subset)
        else:
            sample_num = (len(subset[i]))
        #课程安排
        dd += random.sample(subset[i],sample_num)
        subset[i] = train_dataset.select(dd)
        print(len(subset[i]))

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
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
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
    # logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)
    logger.setLevel(logging.INFO)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    # if training_args.should_log:
    #     transformers.utils.logging.set_verbosity_info()
    #     transformers.utils.logging.enable_default_handler()
    #     transformers.utils.logging.enable_explicit_format()
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label={i: l for l, i in label_to_id.items()},
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    DE_config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label={i: l for l, i in label_to_id.items()},
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Metrics
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        # if training_args.resume_from_checkpoint is not None:
        #     checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tags": "token-classification"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
