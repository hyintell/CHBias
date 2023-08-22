import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

# def mean_list(valid_ids, valid_values, stride=2, iters=1000):
#     length = len(valid_ids)
#     nums = length // stride
#     used_ids = []
#     mean_values = []
#     std_vallues = []
#     valid_values = np.asarray(valid_values)
#     for k in range(nums):
#         used_ids.append(int(k*stride*iters))
#         temp_list = valid_values[k*stride:(k+1)*stride]
#         mean = np.mean(temp_list)
#         std = np.std(temp_list)
#         mean_values.append(mean)
#         std_vallues.append(std)
#     return used_ids, mean_values, std_vallues
def mean_list(loss1, loss2, loss3):
    with open(loss1, 'r', encoding='utf-8') as l1f:
        loss1_list = []
        for line in l1f.readlines():
            line = line.strip()
            loss1_list.append(float(line))

    with open(loss2, 'r', encoding='utf-8') as l2f:
        loss2_list = []
        for line in l2f.readlines():
            line = line.strip()
            loss2_list.append(float(line))
    with open(loss3, 'r', encoding='utf-8') as l3f:
        loss3_list = []
        for line in l3f.readlines():
            line = line.strip()
            loss3_list.append(float(line))
    # totoal num=100, n=1
    n = 2
    new_loss1_list = []
    for i in range(0, len(loss1_list), n):
        mean = np.mean(loss1_list[i: i + n])
        new_loss1_list.append(mean)

    new_loss2_list = []
    for i in range(0, len(loss2_list), n):
        mean = np.mean(loss2_list[i: i + n])
        new_loss2_list.append(mean)

    new_loss3_list = []
    for i in range(0, len(loss3_list), n):
        mean = np.mean(loss3_list[i: i + n])
        new_loss3_list.append(mean)

    mean_values = []
    std_vallues = []

    for i, loss_tuple in enumerate(zip(new_loss1_list, new_loss2_list, new_loss3_list)):

            loss_list = list(loss_tuple)
            mean = np.mean(loss_list)
            std = np.std(loss_list)
            mean_values.append(mean)
            std_vallues.append(std)
    # print(len(mean_values))
    # print(len(std_vallues))
    return mean_values, std_vallues


# 设置全局格式，包括字体风格和大小等等
# 这里主要用来修改文本字体里面的格式
font_size = 50
config = {
    "font.family":'serif',
    "font.size": font_size,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

# 修改x轴的显示方式，科学计数法
def formatnumx(x, pos):
    return '%d' % (x/1000)
formatterx = FuncFormatter(formatnumx)

fig, ax1 = plt.subplots(figsize=(16, 16), dpi=100)
# ax2 = ax1.twinx()

debais_method = 'CD_CTDA'
bias_list = ['appearance', 'age', 'gender', 'orientation']

bias = bias_list[0]
if bias == 'appearance':
    loss1 = './loss/'+debais_method+'/appearance1.txt'
    loss2 = './loss/'+debais_method+'/appearance2.txt'
    loss3 = './loss/'+debais_method+'/appearance3.txt'
    color = 'C4'
elif bias == 'age':
    loss1 = './loss/'+debais_method+'/age1.txt'
    loss2 = './loss/'+debais_method+'/age2.txt'
    loss3 = './loss/'+debais_method+'/age3.txt'
    color = 'C2'
elif bias == 'gender':
    loss1 = './loss/'+debais_method+'/gender1.txt'
    loss2 = './loss/'+debais_method+'/gender2.txt'
    loss3 = './loss/'+debais_method+'/gender3.txt'
    color = 'C0'
elif bias == 'orientation':
    loss1 = './loss/'+debais_method+'/orientation1.txt'
    loss2 = './loss/'+debais_method+'/orientation2.txt'
    loss3 = './loss/'+debais_method+'/orientation3.txt'
    color = 'C1'

def plot_lines(ap_loss1, ap_loss2, ap_loss3):
    ap_mean_values, ap_std_vallues = mean_list(ap_loss1, ap_loss2, ap_loss3)
    id = list(range(0, len(ap_mean_values)))
    ap_std_down = [ap_mean_values[x]-ap_std_vallues[x] for x in range(len(ap_mean_values))]
    ap_std_up = [ap_mean_values[x]+ap_std_vallues[x] for x in range(len(ap_mean_values))]
    l1 = ax1.plot(id, ap_mean_values, color=color)
    ax1.fill_between(id, ap_std_down, ap_std_up, color=color, alpha=0.3)

    # ag_mean_values, ag_std_vallues = mean_list(ag_loss1, ag_loss2, ag_loss3)
    # id = list(range(0, len(ag_mean_values)))
    # ag_std_down = [ag_mean_values[x] - ag_std_vallues[x] for x in range(len(ag_mean_values))]
    # ag_std_up = [ag_mean_values[x] + ag_std_vallues[x] for x in range(len(ag_mean_values))]
    # l2 = ax1.plot(id, ag_mean_values, color='C1')
    # ax1.fill_between(id, ag_std_down, ag_std_up, color='C1', alpha=0.3)

    # plt.legend(handles=[l1, l2], labels=['appearance', 'age'], loc='best')

ap = plot_lines(loss1, loss2, loss3)

# print(len(mean_values))
# valid_ids 为迭代次数（每一千次为一个单位）
# valid_mse 为每1000次的validation值
# used_ids, mean_values, std_vallues = mean_list(valid_ids, valid_mse, stride=2, iters=1000)
# std_down = [mean_values[x]-std_vallues[x] for x in range(len(mean_values))]
# std_up = [mean_values[x]+std_vallues[x] for x in range(len(mean_values))]
# ax2.plot(used_ids, mean_values, color='C1', label='Ours w/o PT')
# ax2.fill_between(used_ids, std_down, std_up, color='C1', alpha=0.3)

ax1.set_xlabel(r'Steps', fontdict={'family': 'Times New Roman', 'size': font_size})
ax1.set_ylabel('Training loss', fontdict={'family': 'Times New Roman', 'size': font_size})
# ax2.set_ylabel('Validation result (VOI)', color='C1', fontdict={'family': 'Times New Roman', 'size': font_size})

#plt.gca().xaxis.set_major_formatter(formatterx)

ax1.tick_params(labelsize=font_size)

ticks = ax1.set_xticks([0, 25, 50, 75, 100])
# print(list(range(0,400,80)))
labels = ax1.set_xticklabels(['0', '100', '200', '300', '400'], rotation=30, fontsize='small')
# labels = ax1.get_xticklabels() + ax1.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# ax2.tick_params(labelsize=font_size)
# labels = ax2.get_xticklabels() + ax2.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]

# ax2.set_ylim((1, 6.5))

fname_path = './loss_pic/' + debais_method + '_' + bias + '.pdf'
plt.savefig(fname_path)
# plt.show()