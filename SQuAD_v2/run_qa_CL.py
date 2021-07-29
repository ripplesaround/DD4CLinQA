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
Fine-tuning the library models for question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import os
# notice 制定GPU
from datasets.arrow_dataset import update_metadata_with_features

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pyarrow as pa
import torch
from geomloss import SamplesLoss

import copy
import random
import logging
import time
import sys
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import transformers
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from utils_qa import postprocess_qa_predictions
from Trainer_CL import QATrainer_CL

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0.dev0")

logger = logging.getLogger(__name__)


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
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
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
    # notice 添加的参数
    div_subset: Optional[int] = field(
        default=3,
        metadata={"help": "划分成几个subset"},
    )

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
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
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


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

        # output 的第一项是loss 第二项是SequenceClassifierOutput
        # SequenceClassifierOutput 是 这个（https://huggingface.co/transformers/model_doc/bert.html）
        # if return_dict=True is passed or when config.return_dict=True) or a tuple of torch.FloatTensor comprising various elements depending on the configuration (BertConfig) and inputs.
        # 默认返回的是一个tuple 分别是：loss，logit，hidden-states, attentions
        # hidden state的第一项是embedding的输出，后面是每一层的输出
        # hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) – Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
        # 所以这里选用 output[1][2][0] 来访问hidden-state

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
    dd_whole = []
    for i in range(data_args.div_subset):
        sample_num = (len(subset[i])) // data_args.div_subset
        dd += random.sample(subset[i],sample_num)
        dd_whole += subset[i]
        subset[i] = train_dataset.select(dd)
        print(len(subset[i]))

    # 构建一个按照顺序排序的集合
    train_dataset_ordered = train_dataset.select(dd_whole)

    logger.info("***难度评估结束***")

    # notice 释放缓存
    torch.cuda.empty_cache()
    return subset, train_dataset_ordered

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
    set_seed(int(time.time()))
    # set_seed(training_args.seed)

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
            extension = data_args.train_file.split(".")[-1]

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, field="data", cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # output_hidden_states = True,
    )
    DE_config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
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
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    else:
        column_names = datasets["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # Create train feature from dataset
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        # Validation Feature Creation
        eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        predict_dataset = predict_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            is_world_process_zero=trainer.is_world_process_zero(),
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Initialize our Trainer
    # 这里与其他的runqa的不一样，要重新重构
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # 用于难度划分
    DE_model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.DE_model,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=DE_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    trainer_DE = QATrainer_CL(
        model=DE_model,
        args=training_args,
        train_dataset= change_dataset(train_dataset),
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    curr_subset, train_dataset_ordered = DE(trainer_DE, train_dataset, training_args, data_args)
    curr_subset.insert(0,train_dataset_ordered)

    training_args_curr = copy.deepcopy(training_args)
    training_args_curr.num_train_epochs = 1
    trainer_curr = []
    for i in range(data_args.div_subset):
        trainer_curr.append(
            QuestionAnsweringTrainer(
                model=model,
                args=training_args_curr,
                train_dataset=curr_subset[i] if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                eval_examples=eval_examples if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                post_process_function=post_processing_function,
                compute_metrics=compute_metrics,
            )
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        # if training_args.resume_from_checkpoint is not None:
        #     checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        if last_checkpoint is not None:
            checkpoint = last_checkpoint

        logger.info("num_train_epochs --- " + str(training_args.num_train_epochs))
        logger.info("curr num_train_epochs --- " + str(training_args_curr.num_train_epochs))
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        checkpoint = training_args.output_dir
        trainer.save_model()  # Saves the tokenizer too for easy upload
        logger.info("save model at " + training_args.output_dir)
        logger.info("trainer.state.global_step = %s", trainer.state.global_step)
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

        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        # trainer.save_state()
        trainer_curr[data_args.div_subset - 1].log_metrics("train", metrics)
        trainer_curr[data_args.div_subset - 1].save_metrics("train", metrics)
        trainer_curr[data_args.div_subset - 1].save_state()
        torch.cuda.empty_cache()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # metrics = trainer.evaluate()
        metrics = trainer_curr[data_args.div_subset - 1].evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer_curr[data_args.div_subset - 1].log_metrics("eval", metrics)
        trainer_curr[data_args.div_subset - 1].save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    if training_args.push_to_hub:
        trainer.push_to_hub()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()