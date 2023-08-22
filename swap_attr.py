# import pandas as pd
# import re
# from utils import reddit_helpers as rh


# data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
# demo = 'race' # 'gender'  # 'orientation' # 'religion2' # 'religion1' # 'race' #'gender'
# demo_1 = 'black' # 'female'  # 'lgbtq' # 'muslims' # 'jews' # 'black' #'female' # 'jews'
# demo_2 = 'white' # 'male'  # 'straight' # 'christians' # 'white_pos'
# in_file_suffix = '_processed_phrase_biased_testset' # '_processed_phrase_biased_trainset'
# out_file_suffix = '_processed_phrase_unbiased_testset_pos_attr' # '_processed_phrase_unbiased_trainset_pos_attr'
#
# demo1_df_processed = pd.read_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + in_file_suffix + '.csv', encoding='Latin-1')
#
# print(demo1_df_processed.head())
# print(demo1_df_processed.shape)
#
#
# demo2_df = pd.DataFrame(columns=['initial_attr', 'replaced_attr', 'comments', 'comments_processed'])
demo = 'age'
bias = 'age'# 'gender'  # 'orientation' # 'religion2' # 'religion1' # 'race' #'gender'
# original_train_file = 'C:/Users/20214573/OneDrive - TU Eindhoven/桌面/data/' + demo + '/' + demo + '_train.txt'
# swapped_train_file = 'C:/Users/20214573/OneDrive - TU Eindhoven/桌面/data/' + demo + '/' + demo + '_bias_manual_swapped_attr_train-.txt'
original_train_file = './data/' + bias + '/' + bias + '_' + 'train.txt'
swapped_train_file = './data/' + bias + '/' + bias + '_' + 'train_cada.txt'

if demo == 'appearance':
    pairs = (('丑陋', '美丽'), ('丑', '美'), ('恶心', '舒心'), ('辣眼睛', '好看'), ('懒惰', '努力'), ('懒', '勤奋'),
             ('讨厌', '喜欢'), ('厌恶', '喜欢'), ('不好看', '漂亮'), ('不喜欢', '喜欢'), ('油腻', '清新'))
elif demo == 'gender':
    pairs = (('丑', '美'), ('胖', '瘦'), ('骚', '文静'), ('傻', '聪明'), ('矮', '高'), ('瘦', '胖'), ('讨厌', '喜欢'),
            ('空姐', '空少'), ('老', '年轻'), ('恶心', '美好'), ('大龄', '年轻'), ('不靠谱', '不离谱'), ('矫情', '耿直'),
             ('服务员', '老板'), ('难看', '好看'), ('麻烦', '简单'), ('撒泼', '文雅'), ('感性', '理性'))
elif demo == 'orientation':
    pairs = (('肮脏', '干净'), ('恶心', '舒心'), ('讨厌', '喜欢'), ('远离', '接近'), ('传染病', '健康'), ('下流', '高尚'), ('辣眼睛', '好看'))
elif demo == 'age':
    pairs = (('贪小便宜', '勤俭节约'), ('讹人', '帮助人'), ('固执', '随和'), ('讨厌', '喜欢'), ('倚老卖老', '老当益壮'), ('恶心', '舒心'))

else:
    raise ValueError("Specify correct demographic")


with open(original_train_file, 'r', encoding='utf-8') as fo:
    replace_lines = []
    lines = fo.readlines()
    for line in lines:
        line = line.strip()
        for p in pairs:
            line = line.replace(*p)
            # line = line.replace(p[0], '%temp%').replace(*reversed(p)).replace('%temp%', p[1])

        replace_lines.append(line)
with open(swapped_train_file, 'w', encoding='utf-8') as fo:
    for line in replace_lines:
        fo.write(line + '\n')

# for idx, row in demo1_df_processed.iterrows():
#     initial_attr = []
#     replaced_attr = []
#     s = row['comments_processed']
#     # print(s)
#     demo2_df.at[idx, 'comments'] = s
#
#     for p in pairs:
#         s = s.replace(*p)
#
#         if p[1] in s and p[0] in row['comments_processed']:
#             initial_attr.append(p[0])
#             replaced_attr.append(p[1])
#
#     demo2_df.at[idx, 'comments_processed'] = s
#     demo2_df.at[idx, 'initial_attr'] = initial_attr
#     demo2_df.at[idx, 'replaced_attr'] = replaced_attr
#
# print('Shape of demo2 data {}'.format(demo2_df.shape))
# demo2_df.to_csv(data_path + demo + '/' + 'reddit_comments_' + demo + '_' + demo_1 + out_file_suffix + '.csv', index=False)