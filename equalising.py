class EVAModel_EQ(nn.Module):

    def __init__(
            self,
            config: EVAConfig,
            parallel_output=False,
            checkpoint_activations=False,
            checkpoint_num_layers=1):

        super(EVAModel_EQ, self).__init__()
        if config.vocab_size is None:
            raise RuntimeError("Should set vocab size")
        self.enc_config = copy.deepcopy(config)
        self.dec_config = copy.deepcopy(config)

        self.parallel_output = parallel_output

        self.word_embeds = mpu.VocabParallelEmbedding(config.vocab_size, config.d_model,
                                                      init_method=init_method_normal(std=0.02))

        self.role_embeds = nn.Embedding(2, config.d_model)

        self.lm_head = mpu.VocabParallelEmbedding(config.vocab_size, config.d_model,
                                                  init_method=init_method_normal(std=config.init_method_std))

        self.encoder = mpu.ParallelTransformer(self.enc_config, word_embeds=self.word_embeds,
                                               role_embeds=self.role_embeds, is_decoder=False,
                                               checkpoint_activations=checkpoint_activations,
                                               checkpoint_num_layers=checkpoint_num_layers)
        self.decoder = mpu.ParallelTransformer(self.dec_config, word_embeds=self.word_embeds,
                                               role_embeds=self.role_embeds, is_decoder=True,
                                               checkpoint_activations=checkpoint_activations,
                                               checkpoint_num_layers=checkpoint_num_layers)
        self.debias_head = nn.functional.linear

        # for model_batch, no_model_batch in train_dataloader:
        #     for k in model_batch:
        #         model_batch[k] = model_batch[k].to(device)
        #     for k in no_model_batch:
        #         no_model_batch[k] = no_model_batch[k].to(device)

    def forward(
            self,
            enc_input_ids=None,
            enc_role_ids=None,
            enc_attention_mask=None,
            dec_input_ids=None,
            dec_role_ids=None,
            dec_attention_mask=None,
            cross_attention_mask=None,
            enc_hidden_states=None,
            past_key_values=None,
            only_encoder=False,
            demographic=None,
            target_pair_type=None,
            lm_hyp=None,
            debias_hyp=None,
            norm_debias_loss=None,
            labels=None,
            loss_mask=None
    ):
        labels = enc_input_ids
        if enc_hidden_states is None:
            enc_outputs = self.encoder(
                input_ids=enc_input_ids,
                attention_mask=enc_attention_mask,
                role_ids=enc_role_ids,
            )

            enc_hidden_states = enc_outputs["last_hidden_state"]

        if only_encoder:
            outputs = {
                "encoder_last_hidden_state": enc_hidden_states,
            }

            return outputs

        dec_outputs = self.decoder(
            input_ids=dec_input_ids,
            role_ids=dec_role_ids,
            attention_mask=dec_attention_mask,
            cross_attention_mask=cross_attention_mask,
            enc_hidden_states=enc_hidden_states,
            past_key_values=past_key_values,
        )
        ############gai##
        # outputs = {
        #     "last_hidden_state": hidden_states,
        #     "past_key_values": present_key_value_states,
        #     "hidden_states": None,
        #     "attentions": all_self_attention_probs,
        #     "cross_attentions": all_cross_attention_probs
        # }
        # print('hidden_states shape: ', dec_outputs)
        hidden_states = dec_outputs['last_hidden_state']

        # print('hidden_states: {}'.format(hidden_states))

        # lm_logits = self.lm_head(hidden_states)
        debias_loss_total = torch.tensor(0)
        if demographic:
            # print('demographic', demographic)
            if embedding_type == 'input':
                pass
            elif embedding_type == 'output':
                all_output_embeds = self.lm_head.weight.data
                all_output_embeds = all_output_embeds.to(self.device)
            else:
                raise ValueError('Please specify valid embedding type - input or output')
            if demographic == 'appearance':
                target_ids_list = [[[2531, 91], [2951, 91]], [[4130, 10506], [2951, 91]],
                                   [[4130, 4218], [2951, 91]], [[5121, 91], [148, 40]],
                                   [[5121, 5], [148, 5]], [[2531, 5], [2951, 5]], [[4130, 5]], [[2951, 5]]]

            elif demographic == 'orientation':
                target_ids_list = [[[331, 102, 2444], [2047, 102, 2444]], [[691, 102, 2444], [2047, 102, 2444]],
                                   [[1884, 852, 1679], [2047, 102, 2444]], [[331, 102, 5], [2047, 102, 5]],
                                   [[995, 10506, 5], [955, 270, 5]], [[993, 1007, 847], [955, 270, 5]]]

            elif demographic == 'age':
                target_ids_list = [[[359, 38, 5], [44, 1282, 38]], [[61, 359, 44], [1566, 44, 5]],
                                   [[359, 74, 2597], [104, 9828, 5]], [[74, 2597, 5], [104, 9828, 5]],
                                   [[74, 564, 5], [104, 6486, 2473]], [[359, 255, 5], [104, 9828, 5]],
                                   [[359, 239, 4218], [104, 6486, 2473]], [[359, 239, 239], [104, 6486, 2473]],
                                   [[359, 239, 5], [104, 6486, 2473]]]

            elif demographic == 'gender':
                target_ids_list = [[[293, 418], [418, 91]], [[3271, 293], [270, 38]], [[293, 5], [270, 5]],
                                   [[88, 5], [35, 5]], [[1214, 1214], [1975, 1975]], [[1214, 5], [1975, 5]],
                                   [[1286, 1286], [1436, 1436]], [[1286, 5], [1436, 5]], [[564, 564], [1890, 1890]],
                                   [[564, 5], [1890, 5]], [[6486, 2473], [104, 9828]], [[695, 3681], [2864, 2864]]]

            else:
                raise ValueError('Please specify valid demographic - religion1, religion2, orientation, gender or race')

            target_ids_list = torch.LongTensor(target_ids_list)
            target_ids_list = target_ids_list.to(self.device)

            make_mask = labels.clone()
            # print('make_mask {}'.format(make_mask))
            make_mask[make_mask > 0] = 1
            make_mask[make_mask < 0] = 0
            # print('make_mask {}'.format(make_mask))

            remove_pad_mask = make_mask.unsqueeze(-1).expand(hidden_states.size())
            # print('remove_pad_mask {}'.format(remove_pad_mask))

            hidden_states_no_pad_token = hidden_states * remove_pad_mask
            # print('hidden_states_no_pad_token {}'.format(hidden_states_no_pad_token))

            if target_pair_type == 'per_sent_targets':
                for i, input_id in enumerate(dec_input_ids):
                    # print('input_id:', input_id)
                    for t_i, target in enumerate(target_ids_list[:, 0]):
                        # print('target', target)
                        # 去掉补齐target的[pad]:
                        # if target[-1] == 1:
                        #     target = target[:-1]
                        n_5 = 0
                        for i in target[::-1]:
                            if k== 5:
                                n_5 += 1
                            else:
                                break
                        target = target[:-n_5]
                        # input_id_split = list(zip(*[iter(input_id)*len(target)]))
                        target_len = len(target)
                        input_id_split = []
                        for start in range(0, len(input_id)):
                            end = start + target_len
                            input_id_split.append(input_id[start:end])
                        # if target[0] in input_id:
                        # print('target {}'.format(target))
                        # print('target_ids_list:', target_ids_list[t_i])
                        # print('input_id_split', input_id_split)
                        # if target in input_id_split:
                        target_in_input_id = False
                        for input_id_s in input_id_split:
                            # print('input_id_s', input_id_s)
                            # print('target', target)
                            if input_id_s.equal(target):
                                target_in_input_id = True
                                break
                        # print('target_in_input_id', target_in_input_id)
                        if target_in_input_id:
                            if embedding_type == 'input':
                                target_embeds = all_input_embeds(target_ids_list[t_i])
                                # print('target_embeds {}'.format(target_embeds))
                            elif embedding_type == 'output':
                                target_embeds = all_output_embeds[target_ids_list[t_i]]
                                # print('target_embeds {}'.format(target_embeds.size()))
                            else:
                                raise ValueError('Please specify valid embedding type - input or output')
                            # 先判断target 的len(黑人=2，美国人=3)，再使用debias_head len次（乘以target的每个字），\
                            # 其中将target_embeds替换成debias_logits, 当target1 != target2时，用全1向量补全短的target。
                            # print('hidden_states_no_pad_token[i]', hidden_states_no_pad_token[i].size())
                            len_target = target_embeds.size()[1]
                            # print('target_embeds {}'.format(target_embeds.size()))
                            # print('len_target {}'.format(len_target))

                            if len_target > 1:
                                target_embeds = torch.mean(target_embeds, 1)
                            debias_logits = self.debias_head(hidden_states_no_pad_token[i], weight=target_embeds)

                            softmax_layer = nn.Softmax(dim=1)
                            debias_softmax = softmax_layer(debias_logits)
                            debias_softmax = torch.squeeze(debias_softmax)

                            debias_softmax_1 = torch.flatten(debias_softmax[:, 0])
                            debias_softmax_2 = torch.flatten(debias_softmax[:, 1])

                            debias_loss = torch.abs(torch.log(debias_softmax_1 / debias_softmax_2))
                            if norm_debias_loss:
                                debias_loss = torch.sum(debias_loss) / torch.sum(input_id != 30000)
                            else:
                                debias_loss = torch.sum(debias_loss)

                            debias_loss_total = debias_loss_total + debias_loss

                debias_loss_total = torch.true_divide(debias_loss_total, target_ids_list.shape[0])

        # total_loss = None
        # if debias_loss_total:
        #     total_loss = lm_hyp * lm_loss + debias_hyp * debias_loss_total
        # else:
        #     # This loss is used for evaluation
        #     total_loss = lm_loss
        # # print('total loss {}'.format(total_loss))
        # # if not return_dict:
        # output = (lm_logits,) + transformer_outputs[1:]
        #
        # return (total_loss,) + output

        last_hidden_state_parallel = mpu.copy_to_model_parallel_region(dec_outputs["last_hidden_state"])
        logits_parallel = F.linear(last_hidden_state_parallel, self.lm_head.weight)

        total_loss = None
        if self.parallel_output:
            lm_logits = logits_parallel
        else:
            lm_logits = mpu.gather_from_model_parallel_region(logits_parallel)

        # print('eva_modeling lm_logits {}'.format(lm_logits.size()))

        # losses = mpu.vocab_parallel_cross_entropy(lm_logits.contiguous().float(), no_model_batch["labels"])
        #
        # loss_mask = no_model_batch["loss_mask"]
        # losses = (losses * loss_mask).sum(-1) / loss_mask.sum(-1)
        # lm_loss = losses.mean()
        # labels = inputs["labels"]

        # print('eva_modeling labels{}'.format(labels))
        lm_loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # print('evalmodeling341 debias_loss_total', debias_loss_total.size())
        if debias_loss_total:
            total_loss = lm_hyp * lm_loss + debias_hyp * debias_loss_total
        else:
            # This loss is used for evaluation
            # total_loss = lm_loss
            total_loss = lm_loss

        outputs = {
            "total_loss": total_loss,
            "lm_logits": lm_logits,
            "enc_last_hidden_state": enc_hidden_states,
            "dec_last_hidden_state": dec_outputs["last_hidden_state"],
            "past_key_values": dec_outputs["past_key_values"],
        }

        return outputs

class GPT2DoubleHeadsModelHardDebiasing(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
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

    # @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=GPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
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
            mc_labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_type=None,
            handle_broken_token=None,
            demographic=None,
            target_pair_type=None,
            lm_hyp=None,
            debias_hyp=None,
            norm_debias_loss=None,
            **kwargs,
    ):
        r"""
            mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
                Index of the classification token in each input sequence.
                Selected in the range ``[0, input_ids.size(-1) - 1[``.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`)
                Labels for language modeling.
                Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
                Indices are selected in ``[-1, 0, ..., config.vocab_size]``
                All labels set to ``-100`` are ignored (masked), the loss is only
                computed for labels in ``[0, ..., config.vocab_size]``
            mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`, defaults to :obj:`None`)
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
        lm_logits = self.lm_head(hidden_states)
        # print(input_ids[0])

        hd_loss_total = torch.tensor(0)
###########gai#HD
        if demographic:
            if demographic == 'appearance':
                target_ids_list = [[[1477, 31], [1541, 31]], [[1687, 3079], [1541, 31]],
                                   [[1687, 773], [1541, 31]], [[1937, 31], [155, 18]],
                                   [[1937, 1], [155, 1]], [[1477, 1], [1541, 1]], [[1687, 1], [1541, 1]]]
                attribute_list = [[1696, 2599, 1], [1696, 1, 1], [745, 69, 1], [2072, 127, 485], [1729, 3451, 1],
                                  [1729, 1, 1], [909, 2504, 1]]
            else:
                raise ValueError('Please specify valid demographic - religion1, religion2, orientation, gender')

            target_ids_list = torch.tensor(target_ids_list)
            target_ids_list = target_ids_list.to(self.device)

            attribute_list = torch.tensor(attribute_list)
            attribute_list = attribute_list.to(self.device)

            lm_head_weights = self.lm_head.weight.data

            diff_matrix = torch.zeros((target_ids_list.size(0), 768))####check if it's 768 ?
            diff_matrix = diff_matrix.to(self.device)

            # create matrix over union of target pair difference in embeddings
            for i, target_pair in enumerate(target_ids_list):

                num_pad_list = target_pair == 1
                num_pad = 0
                for i in num_pad_list[0]:
                    if i:
                        num_pad = num_pad + 1
                # print('target_pair',target_pair)
                # print('num_pad', num_pad)
                target_pair = target_pair[:, :3 - num_pad]
                # print('target_pair[:, :3 - num_pad]', target_pair)
                target_len = len(target_pair[0])
                # print('target_len', target_len)
                j_w = lm_head_weights[target_pair[0]]
                # print('j_w', j_w.size())
                c_w = lm_head_weights[target_pair[1]]
                if target_len > 1:
                    j_w = torch.mean(j_w, 0)
                    c_w = torch.mean(c_w, 0)
                diff_w = (c_w - j_w) / 2
                diff_matrix[i] = diff_w

            u, s, v = torch.svd(diff_matrix)
            s_2_sum = torch.sum(torch.square(s))
            s_total = torch.tensor(0.0)

            # Keep columns of V that most represent bias space
            for k_i, s_i in enumerate(s):
                s_total = s_total + torch.square(s_i)
                if s_total > 0.5 * s_2_sum:
                    k = k_i + 1
                    break

            v_k = v[:, :k]

            # neutralise hidden states of attribute words
            word_projection_total = torch.zeros([768])
            word_projection_total = word_projection_total.to(self.device)
            # hidden_state_final = torch.zeros([768])
            # hidden_state_final = hidden_state_final.to(self.device)

            for i_sent, sent in enumerate(input_ids):
                for i, input_id in enumerate(sent):
                    for x, attr in enumerate(attribute_list):
                        if input_id in attr:
                            if torch.sum(attr != 1) > 1:
                                if all(i1 in attr for i1 in sent[i:i + torch.sum(attr != 1)]):
                                    hidden_state = hidden_states[i_sent, i:i + torch.sum(attr != 1)]
                                    hidden_state_final = torch.mean(hidden_state, 0)
                                else:
                                    hidden_state_final = torch.zeros([768])
                                    hidden_state_final = hidden_state_final.to(self.device)
                            else:
                                hidden_state_final = torch.squeeze(hidden_states[i_sent, i:i+1])

                            for b in range(v_k.shape[1]):
                                word_projection = torch.dot(hidden_state_final, v_k[:, b]) * v_k[:, b]
                                word_projection_total = word_projection_total + word_projection
                            hd_loss = torch.sum(torch.abs(word_projection_total))
                            hd_loss_total = hd_loss_total + hd_loss

        lm_loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # print('lm loss {}'.format(lm_loss))
        if hd_loss_total:
            # print('Hd loss {}'.format(hd_loss_total))
            lm_loss_total = lm_hyp * lm_loss + debias_hyp * hd_loss_total
        else:
            lm_loss_total = lm_loss
        # print('lm loss in hard debias {}'.format(lm_loss_total))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((lm_loss_total,) + output) if lm_loss is not None else output

        mc_loss = None
        mc_logits = None

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss_total,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
