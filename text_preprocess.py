# import pandas as pd
import re

# def delete_commenter(in_file, chunksize):
#     df_chunk = pd.read_csv(in_file, chunksize=chunksize)
#     res_chunk = []
#     for index, row in df_chunk.iterrows():
#         comments = row[1]
#
#     # res_df = pd.concat(res_chunk)

def preprocess_douban(in_file, out_file):
    with open(in_file, 'r', encoding='utf-8') as inf:
        while True:
            line = inf.readline()
            if line:
                line_splits = line.strip().split('\t')
                if line_splits[0] == str(1):
                    with open(out_file, 'a', encoding='utf-8') as outf:
                        for utterance in line_splits[1:]:
                            utterance = re.sub(' ', '', utterance)
                            outf.write(utterance + '\n')
            else:
                break

def preprocess_weibo(in_file, out_file):
    with open(in_file, 'r', encoding='utf-8') as inf:
        while True:
            line = inf.readline()
            if line:
                line_splits = line.strip().split('	')
                with open(out_file, 'a', encoding='utf-8') as outf:
                    for utterance in line_splits:
                        outf.write(utterance + '\n')
            else:
                break

def preprocess_set(in_file):
    out_file = in_file[:-4] + '_set.txt'
    with open(in_file, 'r', encoding='utf-8') as inf:
        lines = inf.readlines()
        lines_list = []
        for line in lines:
            line = line.strip()
            if line not in lines_list:
                lines_list.append(line)
        with open(out_file, 'w', encoding='utf-8') as outf:
            for line in lines_list:
                outf.write(line + '\n')


def extract_demographic_groups(demographic, file):
    if demographic == 'gender':
        query = ['女', '妇', '她', '姐', '妹', '妈', '母亲', '妻', '姑娘', '太太', '夫人', '阿姨', '奶', '姥', '婆']
        attribute = ['护士', '秘书', '服务员', '舞', '拜金', '打扫', '做饭', '家务', '艺术', '温柔',
                     '整容', '修图', '矫情', '心机', '小气', '做作', '嫉妒', '女司机', '钱',
                     '潜规则', '吃醋', '天真', '单纯', '虚荣', '软弱', '清纯', '八卦', '娱乐', '偶像', '追星',
                     '拍照', '自拍', '消费', '网购', '包养', '花', '打扮', '出轨', '婚', '年龄']

    elif demographic == 'age':
        query = ['老人', '老年', '大爷', '大妈', '老头', '老太']
        attribute = ['古板', '傲慢', '高傲', '邋遢', '蛮横', '霸道', '素质', '讨厌', '坏', '讲理', '骂', '脏', '吵', '骗',
                     '不要脸', '脾气', '文化', '教育', '指责', '责任', '劣根']

    elif demographic == 'orientation':
        # query = ['男同', '女同', '同性恋', '同性', '双性恋', '基佬', '蕾丝', '拉拉']
        query = ['gay']
        attribute = ['变态', '病态', '病', '心理', '精神', '酒', '虐待', '淫', '失败', '伤风败俗', '恶心', '吐', '灭绝', '乱', '毒',
                     '道德', '伦理', '艾滋', '反人类', '恐', '死', '畸形', '消灭', '反对', '不赞成', '悲', '禁', '可耻', '歧视']

    elif demographic == 'disability':
        query = ['残疾', '残疾人', '心理疾病', '精神病', '精神疾病', '抑郁症', '痴呆', '自闭症']
        attribute = ['变态', '病态', '病', '心理', '精神', '酒', '虐待', '淫', '失败', '伤风败俗', '恶心', '吐', '灭绝', '乱', '毒',
                     '道德', '伦理', '艾滋', '反人类', '恐', '死', '畸形', '消灭', '反对', '不赞成', '悲', '禁', '可耻', '歧视']

    elif demographic == 'region':
        query = ['外地人']
        attribute = ['变态', '病态', '病', '心理', '精神', '酒', '虐待', '淫', '失败', '伤风败俗', '恶心', '吐', '灭绝', '乱', '毒',
                     '道德', '伦理', '艾滋', '反人类', '恐', '死', '畸形', '消灭', '反对', '不赞成', '悲', '禁', '可耻', '歧视']

    out_file = file[:-4] + '_' + demographic + '.txt'
    with open(file, 'r', encoding='utf-8') as inf:
        while True:
            line = inf.readline()
            if line:
                line = line.strip()
                for term in query:
                    if term in line:
                        # for attr_term in attribute:
                        #     if attr_term in line:
                                with open(out_file, 'a', encoding='utf-8') as outf:
                                    outf.write(line + '\n')

                    # break
            else:
                break


file = 'D:/datasets/weibo_u.tsv'
demographic = 'orientation'
c = extract_demographic_groups(demographic, file)

# in_file = 'D:/datasets/weibo.tsv'
# out_file = 'D:/datasets/weibo_u.tsv'

in_file = 'D:/datasets/weibo_u_orientation.txt'

# x = preprocess_douban(in_file, out_file)
# x = preprocess_weibo(in_file, out_file)
x = preprocess_set(in_file)







