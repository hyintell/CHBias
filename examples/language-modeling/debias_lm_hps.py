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
from scipy import stats
import torch
import warnings

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMAndDebiasHead,
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

from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
    distributed_broadcast_scalars,
    distributed_concat,
    nested_concat,
    nested_numpify,
    # nested_xla_mesh_reduce,
    set_seed,
)

from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm.auto import tqdm, trange
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch

# from transformers.file_utils import WEIGHTS_NAME, is_datasets_available, is_torch_tpu_available


# if is_torch_tpu_available():
#     import torch_xla.core.xla_model as xm
#     import torch_xla.debug.metrics as met
#     import torch_xla.distributed.parallel_loader as pl

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
pad_token_id = None


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
    eval_data_file_1: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    eval_data_file_2: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
    eval_data_file: Optional[str] = None,
):
    if eval_data_file:
        file_path = eval_data_file if evaluate else args.train_data_file
    else:
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


class TrainerHps(Trainer):

    # do not override init method of parent, hence not calling super init here
    def prediction_loop(
            self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        assert not getattr(
            self.model.config, "output_attentions", False
        ), "The prediction loop does not work with `output_attentions=True`."
        assert not getattr(
            self.model.config, "output_hidden_states", False
        ), "The prediction loop does not work with `output_hidden_states=True`."

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        # loss_all_list: List[float] = []
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        loss_all_list = []
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            # print('inputs: {}'.format(inputs.shape))
            loss, loss_all, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)
            # print('all losses {}'.format(loss_all))
            # print('labels {}'.format(labels))
            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            if loss is not None:
                eval_losses.extend([loss] * batch_size)
            if logits is not None:
                preds = logits if preds is None else nested_concat(preds, logits, dim=0)
            if labels is not None:
                label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)
            if loss_all is not None:
                loss_all_list.append(loss_all)
        # print('loss all list {}'.format(loss_all_list))
        loss_all_list = torch.flatten(torch.stack(loss_all_list))
        # print('all losses after for loop {}'.format(loss_all_list))

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
            if loss_all_list is not None:
                loss_all_list = distributed_concat(loss_all_list, num_total_examples=self.num_examples(dataloader))

        print('local rank {}'.format(self.args.local_rank))
             # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = nested_numpify(preds)
        if label_ids is not None:
            label_ids = nested_numpify(label_ids)

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            if self.args.local_rank != -1:
                metrics["eval_loss"] = (
                    distributed_broadcast_scalars(eval_losses, num_total_examples=self.num_examples(dataloader))
                        .mean()
                        .item()
                )
                metrics["all_losses"] = loss_all_list
            else:
                metrics["eval_loss"] = np.mean(eval_losses)
                metrics["all_losses"] = loss_all_list

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[Any], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        global pad_token_id

        has_labels = all(inputs.get(k) is not None for k in self.args.label_names)
        inputs = self._prepare_inputs(inputs)
        # pad_token_id = self.tokenizer.pad_token_id
        # print('input_ids {}'.format(inputs))
        # print('pad token {}'.format(pad_token_id))

        # for i in inputs["input_ids"]:
        inputs["input_ids"] = inputs["input_ids"][inputs["input_ids"] != pad_token_id]

        with torch.no_grad():
            model.eval()
            # print('input ids shape {}'.format(inputs["input_ids"].shape))
            # print('input ids labels {}'.format(inputs["labels"]))
            outputs = model(**inputs)
            # print('output0 tuple: {}'.format(outputs[0]))
            # print('output1 tuple: {}'.format(outputs[1].shape))
            # print('output2 tuple: {}'.format(outputs[2].shape))
            # print('output3 tuple: {}'.format(outputs[3].shape))

            loss_all = outputs[0]
            if has_labels:
                # The .mean() is to reduce in case of distributed training
                loss = outputs[0].mean().item()
                logits = outputs[1:]
            else:
                loss = None
                # Slicing so we get a tuple even if `outputs` is a `ModelOutput`.
                logits = outputs[:]
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, loss_all, None, None)

        logits = tuple(logit.detach() for logit in logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = tuple(inputs.get(name).detach() for name in self.args.label_names)
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, loss_all, logits, labels)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
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

    pad_token_id = tokenizer.pad_token_id
    print('PAD token id is {}'.format(pad_token_id))

    # if model_args.model_name_or_path:
    #     model = AutoModelWithLMAndDebiasHead.from_pretrained(
    #         model_args.model_name_or_path,
    #         from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #         config=config,
    #         cache_dir=model_args.cache_dir,
    #         debiasing_head=model_args.debiasing_head
    #     )
    # else:
    #     logger.info("Training new model from scratch")
    #     model = AutoModelWithLMAndDebiasHead.from_config(config)
    #
    # special_tokens_dict = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    #
    # model.resize_token_embeddings(len(tokenizer))

    def model_init():
        model1 = AutoModelWithLMAndDebiasHead.from_pretrained(
            model_args.model_name_or_path,
            debiasing_head=model_args.debiasing_head,
            #    from_tf=bool(".ckpt" in model_args.model_name_or_path),
            #    config=config,
            #    cache_dir=model_args.cache_dir,
        )
        model1.resize_token_embeddings(len(tokenizer))
        return model1

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
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
    eval_dataset_1 = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir,
                    eval_data_file=data_args.eval_data_file_1)
        if training_args.do_eval
        else None
    )
    eval_dataset_2 = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir,
                    eval_data_file=data_args.eval_data_file_2)
        if training_args.do_eval
        else None
    )
    if config.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=data_args.plm_probability,
            max_span_length=data_args.max_span_length,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
        )

    def my_objective(metrics):
        # Your elaborate computation here
        eval_op_1 = trainer.evaluate(eval_dataset_1)
        eval_op_2 = trainer.evaluate(eval_dataset_2)
        # print('output loss 1 {}'.format(eval_op_1))
        # print('output loss 2 {}'.format(eval_op_2))
        eval_perp_1 = torch.exp(eval_op_1["eval_all_losses"])
        eval_perp_2 = torch.exp(eval_op_2["eval_all_losses"])
        t_paired, p_paired = stats.ttest_rel(eval_perp_1.cpu().detach().numpy(),
                                             eval_perp_2.cpu().detach().numpy())
        print('t value {}, p value {}'.format(t_paired, p_paired))
        return p_paired

    # def my_hp_space(trial):
    #     return {
    #         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 2]),
    #         "lm_hyp": trial.suggest_float("lm_hyp", 0, 1, 0.01),
    #         "debias_hyp": trial.suggest_float("debias_hyp", 1, 20, 100),
    #     }

    def my_hp_space(trial):
        from ray import tune
        return {
            # "per_device_train_batch_size": tune.choice([4]),
            "lm_hyp": tune.choice([0, 1, 0.01]),
            "debias_hyp": tune.choice([1, 50, 100]),
        }

    # Initialize our Trainer
    trainer = TrainerHps(
        model_init=model_init,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )
    #
    # algo = BayesOptSearch(utility_kwargs={
    #     "kind": "ucb",
    #     "kappa": 2.5,
    #     "xi": 0.0
    # })
    # algo = ConcurrencyLimiter(algo, max_concurrent=2)
    # scheduler = AsyncHyperBandScheduler()

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )

        best_hyp = trainer.hyperparameter_search(direction="maximize", backend="ray", n_trials=1,
                                                 compute_objective=my_objective, hp_space=my_hp_space,
                                                 resources_per_trial={
                                                     "cpu": 1,
                                                     "gpu": 1}
                                                 )  # number of trials
        #  n_jobs=2  # number of parallel jobs, if multiple GPUs n_jobs=1 search_alg=algo, scheduler=scheduler,

        print(best_hyp)

        for n, v in best_hyp.hyperparameters.items():
            setattr(trainer.args, n, v)

        print('training best model...')
        trainer.train(model_path=model_path)
        print('saving best model...')
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    # results = {}
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #
    #     eval_output = trainer.evaluate()
    #
    #     perplexity = math.exp(eval_output["eval_loss"])
    #     result = {"perplexity": perplexity}
    #
    #     output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
    #     if trainer.is_world_master():
    #         with open(output_eval_file, "w") as writer:
    #             logger.info("***** Eval results *****")
    #             for key in sorted(result.keys()):
    #                 logger.info("  %s = %s", key, str(result[key]))
    #                 writer.write("%s = %s\n" % (key, str(result[key])))
    #
    #     results.update(result)

    # return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
