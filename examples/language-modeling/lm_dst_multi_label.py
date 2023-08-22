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
from typing import Optional, Dict, List, Union, NewType, Any, Callable, Tuple
import json
import torch
import pandas as pd
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
import numpy as np

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    GPT2PreTrainedModel,
    GPT2Model,
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
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils_base import BatchEncoding
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
REPLACE_MAP_GLOBAL = {}
MULTI_BIN = MultiLabelBinarizer()
ALL_LABELS = []

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
        metadata={"help": "Whether to force the addition of a padding token to tokenizer that does not already have one."},
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


class DialogTurnDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, data_type: str):
        global MULTI_BIN, ALL_LABELS
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        data = open(file_path, 'r')
        dial_data = json.load(data)

        self.examples = []
        all_labels = []
        for dia_idx in dial_data:
            for dt_i, dia_turn in enumerate(dia_idx["dialogue"]):
                labels = []
                current_turn = dia_turn["system_transcript"] + ' [SEP] ' + dia_turn["transcript"]
                # print('current_turn {}'.format(current_turn))
                if dt_i > 0:
                    last_turn = dia_idx["dialogue"][dt_i]["system_transcript"] + ' [SEP] ' + dia_idx["dialogue"][dt_i]["transcript"]
                else:
                    last_turn = ''
                # print('last turn {}'.format(last_turn))
                turn_input = last_turn + current_turn #ToDO recheck last turn code
                input_ids = tokenizer(turn_input, add_special_tokens=True, truncation=True, max_length=block_size)
                labels_list = [d["slots"] for d in dia_turn["belief_state"]]
                for lab in labels_list:
                    for l in lab:
                        l_str = '-'.join(l)
                        if data_type == 'train':
                            ALL_LABELS.append(l_str)
                            labels.append(l_str)
                        else:
                            if l_str in ALL_LABELS:
                                labels.append(l_str)
                all_labels.append(labels)

                example = {"input_ids": torch.tensor(input_ids["input_ids"]), "dst_labels": labels}
                self.examples.append(example)

        print('total example turns: {}'.format(len(self.examples)))
        # if data_type == 'train':
        #     df = pd.DataFrame({'all_labels': all_labels})
        #     print('df shape {}'.format(df.shape))
        #     # print('df labels {}'.format(df.head()))
        #     labels_unique = df['all_labels'].astype('category').cat.categories.tolist()
        #     REPLACE_MAP_GLOBAL = {k: v for k, v in zip(labels_unique, list(range(1, len(labels_unique) + 1)))}
        #     print('unique labels {}'.format(len(labels_unique)))
        # else:
        #     print('Getting already generated replace map')
        #
        # for exp in self.examples:
        #     try:
        #         exp["labels"] = torch.tensor([REPLACE_MAP_GLOBAL[l] for l in exp["labels"]])
        #     except KeyError:
        #         exp["labels"] = torch.tensor(3000)

        # print('all labels length {}, {}'.format(len(all_labels), all_labels[:5]))
        # print('replace map {}'.format(REPLACE_MAP_GLOBAL))

        if data_type == 'train':
            MULTI_BIN = MULTI_BIN.fit(all_labels)
            transformed_labels = MULTI_BIN.transform(all_labels)
        else:
            transformed_labels = MULTI_BIN.transform(all_labels)
        print('labels shape {}'.format(transformed_labels.shape))

        for i, exp in enumerate(self.examples):
            exp["dst_labels"] = torch.tensor(transformed_labels[i])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset
and collate them into a batch, as a dictionary of Tensors.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])


@dataclass
class DataCollatorDST:
    tokenizer: PreTrainedTokenizer
    # examples: List[Dict[str, torch.tensor]]

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            example_ids = [e["input_ids"] for e in examples]
        # print('examples {}'.format(example_ids))
        batch = self._tensorize_batch(example_ids)
        labels = [e["dst_labels"] for e in examples]
        labels = torch.stack(labels, dim=0)
        # print('batch ids {}'.format(batch))
        # print('labels {}'.format(labels))
        # if self.tokenizer.pad_token_id is not None:
        #     labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": batch, "dst_labels": labels}

    def _tensorize_batch(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)


class TrainerDST(Trainer):

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
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        all_logits = []
        all_labels = []
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)
            # print('logits shape defore concat {}'.format(logits.shape))
            # print('logits defore concat {}'.format(logits))
            # print('labels shape defore concat {}'.format(labels.shape))
            # print('labels defore concat {}'.format(labels))

            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            if loss is not None:
                eval_losses.extend([loss] * batch_size)
            # if logits is not None:
            #     preds = logits if preds is None else nested_concat(preds, logits, dim=0)
            # if labels is not None:
            #     label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)
            if logits is not None:
                for logit in logits:
                    all_logits.extend(torch.sigmoid(logit).cpu().detach().numpy().tolist())
            if labels is not None:
                for label in labels:
                    all_labels.extend(label.cpu().detach().numpy().tolist())

        # print('logits aftr nested concat {}'.format(all_logits))
        print('logits aftr concat {}'.format(len(all_logits)))
        # print('labels shape aftr concat {}'.format(all_labels))
        print('labels after concat {}'.format(len(all_labels)))

        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            print('in distributed....')
            if preds is not None:
                preds = distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))

        # preds = torch.cat(preds)
        # print('preds after cat {}'.format(preds))
        # print('preds shape after cat {}'.format(preds.shape))
        #
        # preds = torch.sigmoid(preds)
        # label_ids = label_ids.reshape(-1)
        #
        # print('preds shape {}'.format(preds.shape))
        # print('labels shape {}'.format(label_ids.shape))

        # preds = [1 if p >= 0.5 else 0 for logi in all_logits for p in logi]
        # label_ids = [l for lab in all_labels for l in lab]

        preds = all_logits >= 0.5
        label_ids = all_labels
        metrics1 = dict()

        metrics1["accuracy"] = metrics.accuracy_score(label_ids, preds)
        print('accuracy {}'.format(metrics1["accuracy"]))
        metrics1["f1_score_micro"] = metrics.f1_score(label_ids, preds, average='micro')
        print('f1_score_micro {}'.format(metrics1["f1_score_micro"]))
        metrics1["f1_score_macro"] = metrics.f1_score(label_ids, preds, average='macro')
        print('f1_score_macro {}'.format(metrics1["f1_score_macro"]))

        if len(eval_losses) > 0:
            if self.args.local_rank != -1:
                metrics1["eval_loss"] = (
                    distributed_broadcast_scalars(eval_losses, num_total_examples=self.num_examples(dataloader))
                        .mean()
                        .item()
                )
            else:
                metrics1["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics1.keys()):
            if not key.startswith("eval_"):
                metrics1[f"eval_{key}"] = metrics1.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics1)

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
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
        has_labels = all(inputs.get(k) is not None for k in self.args.label_names)
        # print('label_names {}'.format(self.args.label_names))
        # print('input labels 1 {}'.format(inputs.get(k) for k in self.args.label_names))
        inputs = self._prepare_inputs(inputs)
        # print('inputs are {}'.format(inputs["input_ids"].shape))
        # print('inputs labels are {}'.format(inputs["dst_labels"].shape))
        # print('labels are {}'.format(has_labels))

        with torch.no_grad():
            outputs = model(**inputs)
            if has_labels:
                # The .mean() is to reduce in case of distributed training
                loss = outputs[0].mean().item()
                logits = outputs[1]
                # print('dst logits {}'.format(logits))
                # print('dst logits shape {}'.format(logits.shape))
            else:
                loss = None
                # Slicing so we get a tuple even if `outputs` is a `ModelOutput`.
                logits = outputs[:]
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = tuple(logit.detach() for logit in logits)
        # print('logits after detach {}'.format(logits))

        if len(logits) == 1:
            logits = logits[0]

        # print('final logits {}'.format(logits.shape))
        # print('labels {}'.format(labels))
        if has_labels:
            labels = tuple(inputs.get(name).detach() for name in self.args.label_names)
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        # labels_new = tuple(label for label in labels)

        # print('final labels {}'.format(len(labels_new)))

        return (loss, logits, labels)


class GPT2LMandDST(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multi_label_classifier = nn.Linear(config.n_embd, 1896)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        dst_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`)
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`)
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        """
        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("lm_labels")
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        # print('transformer output 0 {}'.format(transformer_outputs[0].shape))
        # print('transformer output 1 {}'.format(transformer_outputs[1]))

        lm_logits = self.lm_head(hidden_states)
        dst_logits = self.multi_label_classifier(hidden_states[:, -1, :])
        # print('lm_logits {}'.format(lm_logits.size()))
        # print('dst_logits {}'.format(dst_logits.shape))
        # print('dst_labels {}'.format(dst_labels.shape))

        dst_labels = dst_labels.type_as(dst_logits)
        dst_loss = None
        if dst_labels is not None:
            loss_fct = BCEWithLogitsLoss()
            dst_loss = loss_fct(dst_logits.view(-1, dst_logits.size(-1)), dst_labels.view(-1, dst_labels.size(-1)))
        # lm_loss = None
        # if labels is not None:
        #     shift_logits = lm_logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     loss_fct = CrossEntropyLoss()
        #     lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # total_loss = lm_loss + dst_loss
        print('dst_loss {}'.format(dst_loss))
        # print('total_loss {}'.format(total_loss))

        output = (dst_logits,) + transformer_outputs[1:]

        return (dst_loss,) + output


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
    data_type: Optional[str] = None,
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return DialogTurnDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, data_type=data_type)
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
            cache_dir=cache_dir,
        )


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

    if model_args.model_name_or_path:
        model = GPT2LMandDST.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = GPT2LMandDST.from_config(config)

    special_tokens_dict = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    model.resize_token_embeddings(len(tokenizer))

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
        get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, data_type='train') if training_args.do_train else None
    )
    print('train_dataset {}'.format(train_dataset[0]))
    print('train_dataset {}'.format(train_dataset[1]))
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    print('eval_dataset {}'.format(eval_dataset[0]))
    print('eval_dataset {}'.format(eval_dataset[1]))

    data_collator = DataCollatorDST(
        tokenizer=tokenizer
    )

    # Initialize our Trainer
    trainer = TrainerDST(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=False,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()
        print('eval_output is {}'.format(eval_output))

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

# logits aftr concat 13981104
# labels after concat 13981104
# accuracy 0.4924892197354372
# f1_score_micro 0.4924892197354372