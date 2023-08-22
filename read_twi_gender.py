import json
import pandas as pd

def twi2reddit_shape(input_file, train_manual_file, m_valid_file, f_valid_file, m_test_file, f_test_file,
                     csvm_valid_file, csvf_valid_file, csvm_test_file, csvf_test_file):
    male_data = []
    female_data = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        pairs = json.load(fin)
        for pair in pairs:
            male_data.append(pair[0])
            female_data.append(pair[1])

    # n_total = len(male_data)
    # train_end = int(n_total * 0.6)
    # valid_start = int(n_total * 0.8)


    train_end = int(3000)
    valid_end = int(3300)
    test_end = int(3600)

    train_f = female_data[:train_end]

    valid_m = male_data[train_end:valid_end]
    valid_f = female_data[train_end:valid_end]

    test_m = male_data[valid_end:test_end]
    test_f = female_data[valid_end:test_end]


    with open(train_manual_file, 'w', encoding='utf-8') as tm:
        for m_sent in train_f:
            tm.write(m_sent + '\n')

    with open(m_valid_file, 'w', encoding='utf-8') as mv:
        for m_sent in valid_m:
            mv.write(m_sent + '\n')

    with open(f_valid_file, 'w', encoding='utf-8') as fv:
        for m_sent in valid_f:
            fv.write(m_sent + '\n')

    with open(m_test_file, 'w', encoding='utf-8') as mt:
        for m_sent in test_m:
            mt.write(m_sent + '\n')

    with open(f_test_file, 'w', encoding='utf-8') as ft:
        for m_sent in test_f:
            ft.write(m_sent + '\n')

    #csv
    with open(csvm_valid_file, 'w', encoding='utf-8') as cmv:
        dict_cmv = {'comments_processed': valid_m}
        dfcmv = pd.DataFrame(dict_cmv)
        dfcmv.to_csv(csvm_valid_file)

    with open(csvf_valid_file, 'w', encoding='utf-8') as cfv:
        dict_cfv = {'comments_processed': valid_f}
        dfcfv = pd.DataFrame(dict_cfv)
        dfcfv.to_csv(csvf_valid_file)

    with open(csvm_test_file, 'w', encoding='utf-8') as cmt:
        dict_cmt = {'comments_processed': test_m}
        dfcmt = pd.DataFrame(dict_cmt)
        dfcmt.to_csv(csvm_test_file)

    with open(csvf_test_file, 'w', encoding='utf-8') as cft:
        dict_cft = {'comments_processed': test_f}
        dfcft = pd.DataFrame(dict_cft)
        dfcft.to_csv(csvf_test_file)

def twi2targetswap(input_file, target_pair_file, swap_file):
    male_data = []
    female_data = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        pairs = json.load(fin)
        for pair in pairs:
            male_data.append(pair[0])
            female_data.append(pair[1])



input_file = r"C:\Users\20214573\OneDrive - TU Eindhoven\文档\gender_corpus_twitter.json"
train_manual_file = r"C:\Users\20214573\OneDrive - TU Eindhoven\文档\gender_corpus_twitter_train_manual.txt"
m_valid_file = r"C:\Users\20214573\OneDrive - TU Eindhoven\文档\gender_corpus_twitter_mv.txt"
f_valid_file = r"C:\Users\20214573\OneDrive - TU Eindhoven\文档\gender_corpus_twitter_fv.txt"
m_test_file = r"C:\Users\20214573\OneDrive - TU Eindhoven\文档\gender_corpus_twitter_mt.txt"
f_test_file = r"C:\Users\20214573\OneDrive - TU Eindhoven\文档\gender_corpus_twitter_ft.txt"

csvm_valid_file = r"C:\Users\20214573\OneDrive - TU Eindhoven\文档\reddit_comments_gender_male_biased_valid_reduced.csv"
csvf_valid_file = r"C:\Users\20214573\OneDrive - TU Eindhoven\文档\reddit_comments_gender_female_biased_valid_reduced.csv"
csvm_test_file = r"C:\Users\20214573\OneDrive - TU Eindhoven\文档\reddit_comments_gender_male_biased_test_reduced.csv"
csvf_test_file = r"C:\Users\20214573\OneDrive - TU Eindhoven\文档\reddit_comments_gender_female_biased_test_reduced.csv"

x = twi2reddit_shape(input_file, train_manual_file, m_valid_file, f_valid_file, m_test_file, f_test_file, csvm_valid_file, csvf_valid_file, csvm_test_file, csvf_test_file)