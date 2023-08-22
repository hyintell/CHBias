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
    GPT2LMHeadModel,
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
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
END_OF_TEXT_TOKEN = '<|endoftext|>'


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
    output_resp_file: Optional[str] = field(
        default=None, metadata={"help": "Output file with mode generated responses"}
    )


class ConvoDataset(Dataset):
    """
    This class creates Dataset with input as context and response sequences and labels as responses.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, data_type: str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            convos = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]
        # max_seq_length = 40, 128
        max_seq_length = 128

        self.examples = []
        for line in convos:
            all_fields = line.split("\t")
            key = all_fields[0]
            conv_id = all_fields[2]
            context = all_fields[5]
            if all_fields[4] != 1:
                response = all_fields[-1]
            else:
                response = ''

            context_id = tokenizer.encode(context)
            response_id = tokenizer.encode(response)

            input_ids_len = len(context_id) + len(response_id) + 2
            if input_ids_len > max_seq_length:
                if len(context_id) > input_ids_len - max_seq_length:
                    # cut context from beginning if length of context + response is too long
                    # and len of context is long enough to cut
                    context_id = context_id[input_ids_len - max_seq_length:]
                else:
                    # cut response from end if length of context + response is too long
                    # and len of response is long enough to cut
                    # if no response is available, discard the data
                    if max_seq_length - len(context_id) - 2 < 0:
                        context_id = None
                        response_id = None
                    else:
                        response_id = response_id[:max_seq_length - len(context_id) - 2]

            # print('context_id {}'.format(context_id))
            # print('response_id {}'.format(response_id))
            if context_id is not None:
                input_ids = context_id + [end_of_text_id] + response_id + [end_of_text_id]
                lm_labels = [-1] * len(context_id) + response_id + [end_of_text_id] + [-1]
                position_ids = list(range(len(input_ids)))
                token_type_ids = [0] * len(input_ids)

                example = {"input_ids": input_ids, "labels": lm_labels, "position_ids": position_ids,
                           "token_type_ids": token_type_ids, "conv_id": conv_id, "key": key}
                self.examples.append(example)

        print('total example convos: {}'.format(len(self.examples)))

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
class DataCollatorDSTC7:
    tokenizer: PreTrainedTokenizer
    # examples: List[Dict[str, torch.tensor]]

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_ids_ex, pos_ids, tok_type_ids, label_ids, conv_ids, keys = [], [], [], [], [], []

        if isinstance(examples[0], (dict, BatchEncoding)):
            input_ids_ex = [e["input_ids"] for e in examples]
            pos_ids = [e["position_ids"] for e in examples]
            tok_type_ids = [e["token_type_ids"] for e in examples]
            label_ids = [e["labels"] for e in examples]
            conv_ids = [e["conv_id"] for e in examples]
            keys = [e["key"] for e in examples]

        batch_input_ids = self._tensorize_batch(input_ids_ex)
        batch_pos_ids = self._tensorize_batch(pos_ids)
        batch_tok_type_ids = self._tensorize_batch(tok_type_ids)
        batch_label_ids = self._tensorize_batch(label_ids)

        if self.tokenizer.pad_token_id is not None:
            batch_pos_ids[batch_pos_ids == self.tokenizer.pad_token_id] = 0
            batch_tok_type_ids[batch_tok_type_ids == self.tokenizer.pad_token_id] = 0
            batch_label_ids[batch_label_ids == self.tokenizer.pad_token_id] = -1

        return {"input_ids": batch_input_ids, "position_ids": batch_pos_ids, "token_type_ids": batch_tok_type_ids,
                "labels": batch_label_ids, "conv_id": conv_ids, "key": keys}

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


class GPT2LMHeadModelCustom(GPT2LMHeadModel):
    # authorized_missing_keys = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
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
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # shift_logits = lm_logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss = loss.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels != -1, dim=1).type(loss.type())
            loss = torch.sum(loss) / torch.sum(label_size)
        # print(loss)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class Dstc7Trainer(Trainer):
    """
    This class has some original functions of class Trainer overridden, making them compatible with DSTC7 task
    Model takes position_ids and token_type_ids as inputs. A new response generation function is added.
    """

    def compute_loss(self, model, inputs):
        outputs = model(input_ids=inputs["input_ids"], position_ids=inputs["position_ids"],
                        token_type_ids=inputs["token_type_ids"], labels=inputs["labels"])
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        return outputs[0]

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
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], position_ids=inputs["position_ids"],
                            token_type_ids=inputs["token_type_ids"], labels=inputs["labels"])
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
            return (loss, None, None)

        logits = tuple(logit.detach() for logit in logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = tuple(inputs.get(name).detach() for name in self.args.label_names)
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, logits, labels)

    def generate_i(self, eval_dataset: Optional[Dataset] = None, tokenizer=None, output_resp_file = None):
        model = self.model
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        dec_res_list = []
        for inputs in tqdm(eval_dataloader, desc="generation", disable=False):
            inputs = self._prepare_inputs(inputs)
            # print('prepared inputs are {}'.format(inputs))
            responses = model.generate(input_ids=inputs["input_ids"][:, :64], max_length=128, do_sample=True, top_k=50,
                                       top_p=0.95, num_return_sequences=1, early_stopping=True,
                                       pad_token_id=tokenizer.pad_token_id)
            # responses = model.generate(input_ids=inputs["input_ids"], do_sample=True, top_k=50, top_p=0.95,
            #                            num_return_sequences=1, early_stopping=True,
            #                            pad_token_id=tokenizer.pad_token_id)

            # print('responses are {}'.format(responses))
            for idx, res in enumerate(responses):
                gen_data_dict = {}
                gen_data_dict['key'] = inputs["key"][idx]
                gen_data_dict['input'] = tokenizer.decode(inputs["input_ids"][idx], skip_special_tokens=True)
                gen_data_dict['response'] = tokenizer.decode(res, skip_special_tokens=True)
                dec_res_list.append(gen_data_dict)

        with open(output_resp_file, 'w') as file:
            file.write(json.dumps(dec_res_list))


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
    data_type: Optional[str] = None,
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return ConvoDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, data_type=data_type)
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
        model = GPT2LMHeadModelCustom.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = GPT2LMHeadModelCustom.from_config(config)

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

    if training_args.do_eval:
        print('eval_dataset {}'.format(eval_dataset[0]))
        print('eval_dataset {}'.format(eval_dataset[1]))

    data_collator = DataCollatorDSTC7(
        tokenizer=tokenizer
    )

    # Initialize our Trainer
    trainer = Dstc7Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
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

    # Generate responses from fine-tuned model
    trainer.generate_i(eval_dataset=eval_dataset, tokenizer=tokenizer, output_resp_file=data_args.output_resp_file)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
