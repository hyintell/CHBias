#!/bin/bash

source activate redditbias

group=(religion1 religion2 race gender orientation)
demo1=(jews muslims black female lgbtq)
demo2=(christians christians white male straight)
debias=(eq_loss_grid cos_loss_grid hard_de_grid)
debias_s=(EqualisingLoss Cosine HardDe)
for i in 0 1 2 3 4
do
    for db in 0 1 2
	do
	python ../examples/language-modeling/debias_lm_grid.py \
            --output_dir=../debiasmodels/${group[i]}/${debias[db]}/ \
            --debiasing_head=${debias_s[db]}   \
            --debias_method=${debias_s[db]} \
            --demographic=${group[i]} \
            --demo1_valid=../data/text_files/${group[i]}/${group[i]}_${demo1[i]}_biased_valid_reduced.txt \
            --demo2_valid=../data/text_files/${group[i]}/${group[i]}_${demo2[i]}_biased_valid_reduced.txt \
            --demo1_test=../data/text_files/${group[i]}/${group[i]}_${demo1[i]}_biased_test_reduced.txt \
            --demo2_test=../data/text_files/${group[i]}/${group[i]}_${demo2[i]}_biased_test_reduced.txt \
            --train_data_file=../data/text_files/${group[i]}/${group[i]}_bias_manual_train.txt \
            --model_type=gpt2 \
            --model_name_or_path=../microsoft/DialoGPT_small \
            --config_name=../microsoft/DialoGPT_small  \
            --tokenizer_name=../microsoft/DialoGPT_small \
            --save_total_limit=2 \
            --num_train_epochs=2.0  \
            --do_train \
            --evaluate_during_training \
            --logging_steps=2000 \
            --save_steps=2000 \
            --do_eval  \
            --eval_data_file=../data/text_files/humanref6k.txt \
            --block_size=36 \
	    --line_by_line  \
            --force_pad_token \
            --overwrite_output_dir  \
            --embedding_type=output  \
            --target_pair_type=per_sent_targets \
            --norm_debias_loss 
	done
done
conda deactivate



