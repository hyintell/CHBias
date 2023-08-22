import pandas as pd
import numpy as np
from scipy import stats
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelWithLMAndDebiasHead
import time
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import torch
import math


def get_perplexity_list(df, m, t):
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            perplexity = perplexity_score(row['comments_processed'], m, t)
        except Exception as ex:
            print(ex.__repr__())
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


def get_perplexity_list_test(df, m, t, dem):
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            if dem == 'black':
                perplexity = perplexity_score(row['comments_1'], m, t)
            else:
                perplexity = perplexity_score(row['comments_2'], m, t)
        except Exception as ex:
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list

def perplexity_score(sentence, model, tokenizer):
    with torch.no_grad():
        model.eval()
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss = model(tensor_input, labels=tensor_input)
        # print('loss is {}'.format(loss[0]))
        return math.exp(loss[0])


def model_perplexity(sentences, model, tokenizer):
    total_loss = 0
    for sent in sentences:
        with torch.no_grad():
            model.eval()
            tokenize_input = tokenizer.tokenize(sent)
            tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
            loss = model(tensor_input, labels=tensor_input)
            # print('loss is {}'.format(loss[0]))
            total_loss += loss[0]
    return math.exp(total_loss/len(sentences))

def get_model_perplexity(df, m, t):
    model_perp = model_perplexity(df['comments_processed'], m, t)
    return model_perp


start = time.time()
data_path = '/work-ceph/sbariker/data/'
exp_path = '/work-ceph/sbariker/logs/'

ON_SET = True
GET_PERPLEXITY = True

demo = 'religion2' # 'religion1' # 'orientation' # 'religion2' # 'race' # 'gender' # 'race'  #
demo_1 = 'muslims' # 'jews' # 'lgbtq' # 'muslims' # 'black' # 'female' # 'black_pos' # 'muslims' #
demo_2 = 'christians' # 'straight' # 'white'  # 'male' # 'white_pos'  # 'white'
input_file_biased = '_processed_phrase_biased_testset' # '_processed_phrase_biased' # '_processed_phrase_biased_testset' # '_processed_sent_biased' # '_processed'
input_file_unbiased = '_processed_phrase_unbiased_testset_pos_attr'

debiasing_head = 'EqualisingLoss'

if ON_SET:
    logging.basicConfig(filename=exp_path+'measure_bias_attr_swap_'+demo+'.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
else:
    logging.basicConfig(filename=exp_path+'measure_bias_attr_swap_'+demo+'_test.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')

pd.set_option('max_colwidth', 600)
pd.options.display.max_columns = 10

if ON_SET:
    if GET_PERPLEXITY:

        logging.info('Calculating perplexity')
        race_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + input_file_biased + '.csv')
        race_df_2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + input_file_unbiased + '.csv')

        # race_df = race_df.dropna()
        # race_df_2 = race_df_2.dropna()
        pretrained_model = '/work-ceph/sbariker/models/religion2/lm_loss_swapped_attr/' # 'microsoft/DialoGPT-small' # 'gpt2' # 'roberta-base' # 'bert-base-uncased' #'ctrl'
        # "microsoft/DialoGPT-small" # 'ctrl' # 'openai-gpt' # 'minimaxir/reddit' # 'xlnet-large-cased'
        # pretrained_model = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/colab_outputs/religion1/normal_biased_data_allt/'
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        model = AutoModelWithLMHead.from_pretrained(pretrained_model)
        # model = AutoModelWithLMAndDebiasHead.from_pretrained(pretrained_model, debiasing_head=debiasing_head)
        # model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
        # model = AutoModelForCausalLM.from_pretrained(pretrained_model)

        race_1_perplexity = get_perplexity_list(race_df, model, tokenizer)
        print('Done with demo1 perplexity in {} on set'.format((time.time() - start)/60))
        race_2_perplexity = get_perplexity_list(race_df_2, model, tokenizer)

        model_perp = get_model_perplexity(race_df, model, tokenizer)
        print('Model perplexity {}'.format(model_perp))

        logging.info('Time to get perplexity scores {}'.format((time.time() - start)/60))
        race_df['perplexity'] = race_1_perplexity
        race_df_2['perplexity'] = race_2_perplexity

        # race_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + output_file_suffix + '.csv')
        # race_df_2.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + output_file_suffix +'.csv')
    else:
        logging.info('Getting saved perplexity')
        print('Getting saved perplexity')
        race_df = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + output_file_suffix +'.csv')
        race_df_2 = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + output_file_suffix +'.csv')
        race_1_perplexity = race_df['perplexity']
        race_2_perplexity = race_df_2['perplexity']

logging.debug('Instances in demo 1 and 2: {}, {}'.format(len(race_1_perplexity), len(race_2_perplexity)))
logging.debug('Mean and variance of unfiltered perplexities demo1 - Mean {}, Variance {}'.format(np.mean(race_1_perplexity), np.var(race_1_perplexity)))
logging.debug('Mean and variance of unfiltered perplexities demo2 - Mean {}, Variance {}'.format(np.mean(race_2_perplexity), np.var(race_2_perplexity)))

print('Mean and variance of unfiltered perplexities demo1 - Mean {}, Variance {}'.format(np.mean(race_1_perplexity), np.var(race_1_perplexity)))
print('Mean and variance of unfiltered perplexities demo2 - Mean {}, Variance {}'.format(np.mean(race_2_perplexity), np.var(race_2_perplexity)))

assert len(race_1_perplexity) == len(race_2_perplexity)
print(len(race_1_perplexity))

race_1_p = []
race_2_p = []

logging.info('Filtering out perplexities more than 5000')


for i, (p1, p2) in enumerate(zip(race_1_perplexity, race_2_perplexity)):
    if p1 < 50000 and p2 < 50000:
        race_1_p.append(p1)
        race_2_p.append(p2)
    else:
        print('extreme perplexity d1 {}, d2 {}'.format(p1, p2))
        print(race_df.iloc[i].values)
        print(race_df_2.iloc[i].values)

# reduced_race_df = race_df[(race_df['perplexity'] < 50000) & (race_df_2['perplexity'] < 50000)]
# reduced_race_df_2 = race_df_2[(race_df['perplexity'] < 50000) & (race_df_2['perplexity'] < 50000)]
#
# print('DF shape after reducing {}'.format(reduced_race_df.shape))
# print('DF 2 shape after reducing {}'.format(reduced_race_df_2.shape))
#
# reduced_race_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + input_file_suffix + '_reduced.csv', index=False)
# reduced_race_df_2.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_2 + input_file_suffix + '_reduced.csv', index=False)

logging.info('Saving perplexity difference for each pair of sentence')

dif = np.array(race_1_perplexity) - np.array(race_2_perplexity)

logging.debug('Mean and variance of filtered perplexities demo1 - Mean {}, Variance {}'.format(np.mean(race_1_p), np.var(race_1_p)))
logging.debug('Mean and variance of filtered perplexities demo2 - Mean {}, Variance {}'.format(np.mean(race_2_p), np.var(race_2_p)))
logging.debug('Instances in filtered demo 1 and 2: {}, {}'.format(len(race_1_p), len(race_2_p)))
print('mean of difference {}'.format(np.mean(dif)))
print('Var of difference {}'.format(np.var(dif)))

t_value, p_value = stats.ttest_ind(race_1_perplexity, race_2_perplexity, equal_var=False)

logging.info('Unfiltered perplexities - T value {} and P value {}'.format(t_value, p_value))
print(t_value, p_value)
print(len(race_1_p), len(race_2_p))

# print(race_1_p)
# print(race_2_p)
dif2 = np.array(race_1_p) - np.array(race_2_p)

print('mean of difference {}'.format(np.mean(dif2)))
print('Var of difference {}'.format(np.var(dif2)))

t_vt, p_vt = stats.ttest_ind(race_1_p, race_2_p)
logging.info('Filtered perplexities - T value {} and P value {}'.format(t_vt, p_vt))
print(t_vt, p_vt)

t_vf, p_vf = stats.ttest_ind(race_1_p, race_2_p, equal_var=False)
print(t_vf, p_vf)

logging.info('Total time taken {}'.format((time.time() - start)/60))
