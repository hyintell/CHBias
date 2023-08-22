import json

def get_anony(in_file, out_file, bias, in_demographic, out_demographic):

    if bias == 'appearance':
        if in_demographic == 'fat':
            pairs = (('胖子', '瘦子'), ('胖', '瘦'), ('肥佬', '瘦子'), ('肥婆', '瘦子'), ('肥', '瘦'),
                     ('矮子', '高个'), ('矮', '高'))

    elif bias == 'age':
        # '老人', '中老年', '老年', '大爷', '大妈', '老头'
        if in_demographic == 'old':
            pairs = (('老人', '年轻人'), ('中老年', '年轻人'), ('中老年', '青年'), ('老大爷', '小伙'), ('大爷', '小伙'),
                     ('大妈', '小姑娘'), ('老头', '小伙'), ('老太婆', '小姑娘'), ('老太太', '小姑娘'), ('老太', '小姑娘'))

    elif bias == 'orientation':
        if in_demographic == 'lgbt':
            pairs = (('同性恋', '异性恋'), ('双性恋', '异性恋'), ('txl', '异性恋'), ('同性', '异性'),
                     ('基佬', '直男'), ('gay', '直男'))
    elif bias == 'gender':
        if in_demographic == 'female':
            # '女', '妇', '她', '姐', '妹', '妈', '母亲', '妻', '姑娘', '太太', '夫人', '阿姨', '奶', '姥', '婆'
            pairs = (('女儿', '儿子'), ('妇女', '男人'), ('孕妇', '男人'), ('妇', '夫'), ('女', '男'), ('她', '他'),
                     ('姐姐', '哥哥'), ('姐', '哥'), ('妹妹', '弟弟'), ('妹', '弟'), ('妈妈', '爸爸'), ('妈', '爸'),
                     ('母亲', '父亲'), ('妻子', '丈夫'), ('姑娘', '小伙'), ('太太', '大爷'), ('夫人', '丈夫'), ('阿姨', '叔叔'),
                    ('奶奶', '爷爷'), ('姥姥', '姥爷'), ('老婆', '老公'))


    with open(in_file, 'r', encoding='utf-8') as fine:
        replace_lines = []
        lines = fine.readlines()
        for line in lines:
            line = line.strip()
            for p in pairs:
                line = line.replace(*p)
                # line = line.replace(p[0], '%temp%').replace(*reversed(p)).replace('%temp%', p[1])

            replace_lines.append(line)
    with open(out_file, 'a', encoding='utf-8') as fone:
        for line in replace_lines:
            fone.write(line + '\n')


bias = 'age'
in_demographic = 'old'
out_demographic = 'young'
# in_file = 'C:/Users/20214573/OneDrive - TU Eindhoven/桌面/data/' + bias + '/' + bias + '_' + in_demographic + '_valid.txt'
# out_file = 'C:/Users/20214573/OneDrive - TU Eindhoven/桌面/data/' + bias + '/' + bias + '_' + out_demographic + '_total_valid.txt'

# in_file = 'C:/Users/20214573/OneDrive - TU Eindhoven/桌面/data/' + bias + '/' + in_demographic + '_400.txt'
# out_file = 'C:/Users/20214573/OneDrive - TU Eindhoven/桌面/data/' + bias + '/' + bias + '_bias_manual_swapped_targets_train.txt'

# in_file = 'C:/Users/20214573/OneDrive - TU Eindhoven/桌面/data/' + bias + '/' + bias + '_' + in_demographic + '_total_test.txt'
# out_file = 'C:/Users/20214573/OneDrive - TU Eindhoven/桌面/data/' + bias + '/' + bias + '_' + out_demographic + '_total_test.txt'

# in_file = './data/' + bias + '/' + bias + '_' + in_demographic + '_' + 'valid.txt'
# out_file = './data/' + bias + '/' + bias + '_' + out_demographic + '_' + 'valid.txt'
in_file = './data/' + bias + '/' + bias + '_' + 'train.txt'
out_file = './data/' + bias + '/' + bias + '_' + 'train_ctda.txt'
# x = get_anony(in_file, out_file, bias, in_demographic, out_demographic)

import pandas as pd
def txt2csv(txtfile, csvfile):
    with open(txtfile, 'r', encoding='utf-8') as ft:
        txt_list = []
        for line in ft.readlines():
            line = line.strip()
            txt_list.append(line)
###########gai##### comments_processed
    save = pd.DataFrame(columns=['replaced_sentence'], index=None, data=txt_list)
    with open(csvfile, 'w', encoding='utf-8', newline='') as fc:
        save.to_csv(fc)

bias = 'age'
demo = 'young'
txtfile = './data/' + bias + '/' + bias + '_' + demo + '_' + 'test' + '.txt'
csvfile = './data/' + bias + '/' + bias + '_' + demo + '_' + 'test' + '.csv'
# txtfile = 'C:/Users/20214573/OneDrive - TU Eindhoven/桌面/data/' + bias + '/' + bias + '_' + demo + '_total_test_example.txt'
# csvfile = 'C:/Users/20214573/OneDrive - TU Eindhoven/桌面/data/' + bias + '/' + bias + '_' + demo + '_total_test_example.csv'
# x = txt2csv(txtfile, csvfile)

import random

def shuffle_txt(input_file, out_file):
    lines = []
    out = open(out_file, 'w', encoding='utf-8')
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            lines.append(line)
        random.shuffle(lines)
        for line in lines:
            out.write(line)

bias = 'age'
in_demographic = 'old'
# in_file = './data/' + bias + '/' + 'weibo_orientation_train.txt'
# out_file = './data/' + bias + '/' + 'weibo_orientation_train_f.txt'
in_file = './data/' + bias + '/' + bias + '_' + 'train_ctda.txt'
out_file = './data/' + bias + '/' + bias + '_' + 'train_ctda-f.txt'
# x = shuffle_txt(in_file, out_file)

def list_minus_1(list):
    one_l = []
    for one in list:
        two_l = []
        for two in one:
            index_l = []
            for index in two:
                if index != 1:
                    index -= 1

                index_l.append(index)
            two_l.append(index_l)
        one_l.append(two_l)
    print(one_l)

def list_minus_1_c(list):
    two_l = []
    for two in list:
        index_l = []
        for index in two:
            if index != 1:
                index -= 1
            index_l.append(index)
        two_l.append(index_l)
    print(two_l)

def vocabjson_index(vocab, words):
    with open(words, 'r', encoding='utf-8') as wf:
        lines = wf.readlines()
        with open(vocab, 'r', encoding='utf-8') as vf:
            dict = json.load(vf)
        whole_list = []
        for line in lines:
            for words in line.strip().split(', '):
                words_list = []
                for word in words:
                    index = dict[word]
                    words_list.append(index)
                while len(words_list) < 4:
                    words_list.append(5)

                whole_list.append(words_list)

    print(whole_list)

list = [[278, 659, 1],  [338, 64, 1], [12, 1271, 98], [955, 1, 1], [1235, 1, 1], [51, 2077, 1],
      [2915, 476, 1]]

# x = list_minus_1_c(list)
# vocab = 'D:/Model/debias_CDialGPT/CDial-GPT2_LCCC-base/vocab.json'
# words = 'D:/Model/debias_CDialGPT/data/age/age_attr.txt'
vocab = 'D:/Model/debias_EVA/eva2.0_base/vocab.json'
words = 'D:/Model/debias_CDialGPT/data/age/age_attr.txt'
t = vocabjson_index(vocab, words)