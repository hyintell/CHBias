# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    # LineByLineTextDatasetLabels,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)
import ray
from ray import tune
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import CLIReporter
# if is_wandb_available():
#   import wandb

ray.shutdown()
ray.init(log_to_driver=True, ignore_reinit_error=True)

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    force_pad_token: bool = field(
        default=False,
        metadata={
            "help": "Whether to force the addition of a padding token to tokenizer that does not already have one."
        },
    )
    debiasing_head: Optional[str] = field(
        default=None, metadata={"help": "The type of de-biasing head to be used"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
        # return LineByLineTextDatasetLabels(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
            cache_dir=cache_dir,
        )


class TuneTransformerTrainer(Trainer):

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
        return self.current_optimizer, self.current_scheduler

    def evaluate(self,
                 eval_dataset= None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(
            eval_dataloader, description="Evaluation")
        self.log(output.metrics)

        self.save_state()

        tune.report(**output.metrics)

        return output.metrics

    def save_state(self):
        with tune.checkpoint_dir(step=self.global_step) as checkpoint_dir:
            self.args.output_dir = checkpoint_dir
            # This is the directory name that Huggingface requires.
            output_dir = os.path.join(
                self.args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
            self.save_model(output_dir)
            self.current_optimizer, self.current_scheduler = self.create_optimizer_and_scheduler(360)
            if self.is_world_master():
                torch.save(self.current_optimizer.state_dict(),
                           os.path.join(output_dir, "optimizer.pt"))
                torch.save(self.current_scheduler.state_dict(),
                           os.path.join(output_dir, "scheduler.pt"))


def recover_checkpoint(tune_checkpoint_dir, model_name=None):
    if tune_checkpoint_dir is None or len(tune_checkpoint_dir) == 0:
        return model_name
    # Get subdirectory used for Huggingface.
    subdirs = [
        os.path.join(tune_checkpoint_dir, name)
        for name in os.listdir(tune_checkpoint_dir)
        if os.path.isdir(os.path.join(tune_checkpoint_dir, name))
    ]
    # There should only be 1 subdir.
    assert len(subdirs) == 1, subdirs
    return subdirs[0]


# def train_transformer(config, checkpoint_dir=None):
#     train_dataset, eval_dataset = get_datasets(config)
#
#     training_args = TrainingArguments(
#         output_dir=tune.get_trial_dir(),
#         learning_rate=config["learning_rate"],
#         do_train=True,
#         do_eval=True,
#         evaluate_during_training=True,
#         # Run eval after every epoch.
#         eval_steps=(len(train_dataset) // config["per_gpu_train_batch_size"]) +
#                    1,
#         # We explicitly set save to 0, and do checkpointing in evaluate instead
#         save_steps=0,
#         num_train_epochs=config["num_epochs"],
#         max_steps=config["max_steps"],
#         per_device_train_batch_size=config["per_gpu_train_batch_size"],
#         per_device_eval_batch_size=config["per_gpu_val_batch_size"],
#         warmup_steps=0,
#         weight_decay=config["weight_decay"],
#         logging_dir="./logs",
#     )
#
#     model_name_or_path = recover_checkpoint(checkpoint_dir, config["model_name"])
#     # num_labels = glue_tasks_num_labels[config["task_name"]]
#
#     config = AutoConfig.from_pretrained(
#         model_name_or_path,
#         num_labels=num_labels,
#         finetuning_task=task_name,
#     )
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_name_or_path,
#         config=config,
#     )
#
#     # Use our modified TuneTransformerTrainer
#     tune_trainer = TuneTransformerTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         compute_metrics=utils.build_compute_metrics_fn(task_name),
#     )
#     tune_trainer.train(model_name_or_path)


def train_transformer(config, checkpoint_dir=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config_in = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config_in = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config_in = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    if tokenizer.pad_token_id is None:
        if model_args.force_pad_token:
            # See PR 3388. Some tokenizers don't had pad tokens which causes errors at the encoding step in the collate_fn.
            # We give here the option to force the addition of a pad token. The attention mask is used to ignore this token
            # when feeding to the model.x
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        else:
            logger.warning(
                "Attempting to train a model whose tokenizer has no padding token. This may result in errors in the encoding step. Set the --force_pad_token flag to fix this."
            )

    model_name_or_path = recover_checkpoint(checkpoint_dir, config["model_name"])

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config_in,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config_in)

    special_tokens_dict = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    model.resize_token_embeddings(len(tokenizer))

    if config_in.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    # print('train_dataset {}'.format(train_dataset.examples[0]))

    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    if config_in.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=data_args.plm_probability,
            max_span_length=data_args.max_span_length,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
        )

    training_args = TrainingArguments(
        output_dir=tune.get_trial_dir(),
        learning_rate=config["learning_rate"],
        do_train=True,
        do_eval=True,
        evaluate_during_training=True,
        # Run eval after every epoch.
        eval_steps=(len(train_dataset) // config["per_gpu_train_batch_size"]) + 1,
        # We explicitly set save to 0, and do checkpointing in evaluate instead
        save_steps=0,
        num_train_epochs=config["num_epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_gpu_train_batch_size"],
        per_device_eval_batch_size=config["per_gpu_val_batch_size"],
        warmup_steps=0,
        weight_decay=config["weight_decay"],
        logging_dir="./logs")

    # Initialize our Trainer
    tune_trainer = TuneTransformerTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
        # compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        tune_trainer.train(model_path=model_path)


if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = {
      # These 3 configs below were defined earlier
      "model_name": model_args.model_name_or_path,
      "task_name": "CLM",
      "data_dir": "",
      "per_gpu_val_batch_size": 32,
      "per_gpu_train_batch_size": tune.choice([16, 32, 64]),
      "learning_rate": tune.uniform(1e-5, 5e-5),
      "weight_decay": tune.uniform(0.0, 0.3),
      "num_epochs": tune.choice([2, 3, 4, 5]),
      "max_steps": -1,  # We use num_epochs instead.
      "wandb": {
          "project": "pbt_transformers",
          "reinit": True,
          "allow_val_change": True
      }
    }

    logger.info(config)
    scheduler = PopulationBasedTraining(
          time_attr="training_iteration",
          metric="eval_loss",
          mode="min",
          perturbation_interval=2,
          hyperparam_mutations={
              "weight_decay": lambda: tune.uniform(0.0, 0.3).func(None),
              "learning_rate": lambda: tune.uniform(1e-5, 5e-5).func(None),
              "per_gpu_train_batch_size": [16, 32, 64],
          })

    reporter = CLIReporter(
          parameter_columns={
              "weight_decay": "w_decay",
              "learning_rate": "lr",
              "per_gpu_train_batch_size": "train_bs/gpu",
              "num_epochs": "num_epochs"
          },
          metric_columns=[
              "eval_acc", "eval_loss", "epoch", "training_iteration"
          ])

    analysis = tune.run(
          train_transformer,
          resources_per_trial={
              "cpu": 1,
              "gpu": 1
          },
          config=config,
          num_samples=3,
          scheduler=scheduler,
          keep_checkpoints_num=3,
          checkpoint_score_attr="training_iteration",
          progress_reporter=reporter,
          local_dir="./ray_results/",
          name="tune_trans")

    best_config = analysis.get_best_config(metric="eval_loss", mode="min")
    print(best_config)
