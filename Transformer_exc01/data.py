import random
import torch

import numpy as np

# 定义字典
# 1.定义字符串
dictionary_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'
# 2.创建字典x，将每个字符映射到一个唯一的整数（索引）。
dictionary_x = {word: i for i, word in enumerate(dictionary_x.split(','))}
# 3.创建一个列表，dictionary_xr[i]会返回索引i对应的字符
dictionary_xr = [k for k, v in dictionary_x.items()]
# 4.创建字典y，小写字母变成大写
dictionary_y = {k.upper(): v for k, v in dictionary_x.items()}
dictionary_yr = [k for k, v in dictionary_y.items()]

# 生成数据函数
# def get_data():
#     # 定义词集合
#     words = [
#         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'w', 'e', 'r',
#         't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k',
#         'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm'
#     ]
#
#     # 定义每个词被选中的概率
#     # 1.设置权重(假设'm'出现的概率最大，对应权重26；'0'出现的概率最小，对应权重1)
#     p = np.array([
#         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
#         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
#     ])
#     # 2.通过权重计算概率
#     p = p / p.sum()
#
#     # 随机选n个词(定义一句话的长度n在[30,48]之间)
#     n =random.randint(30,48)
#     # 生成采样结果x
#     # words可选的字符集合,size抽取字符数量,replace有放回抽样（同一个字符可能被多次选中）,使用前面定义的概率分布p
#     x = np.random.choice(words, size=n, replace=True, p=p)
#
#     x = x.tolist()
#
#     # 由x生成y的变换函数
#     # 字母大写，数字取9的互补数
#     def f(i):
#         i = i.upper()
#         if not i.isdigit():
#             return i
#         i = 9 - int(i)
#         return str(i)
#
#     # 生成y
#     y = [f(i) for i in x]
#     # 创建非对称模式：输入序列长度n，输出序列长度n+1
#     y = y + [y[-1]]
#     # 对y逆序(打破局部相关性,要求模型理解整个序列的结构)
#     y = y[::-1]
#
#     # 加上首尾符号
#     x = ['<SOS>'] + x + ['<EOS>']
#     y = ['<SOS>'] + y + ['<EOS>']
#
#     # 补pad到固定长度
#     x = x + ['<PAD>'] * 50
#     y = y + ['<PAD>'] * 51
#     x = x[:50]
#     y = y[:51]
#
#     # 编码生成数据
#     x = [dictionary_x[i] for i in x]
#     y = [dictionary_y[i] for i in y]
#
#     # 转成pytorch中的tensor
#     x = torch.LongTensor(x)
#     y = torch.LongTensor(y)
#
#     return x, y

# 两数相加测试,使用这份数据请把main.py中的训练次数改为10
def get_data():
    # 定义词集合
    words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # 定义每个词被选中的概率
    p = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    p = p / p.sum()

    # 随机选n个词
    n = random.randint(10, 20)
    s1 = np.random.choice(words, size=n, replace=True, p=p)

    # 采样的结果就是s1
    s1 = s1.tolist()

    # 同样的方法,再采出s2
    n = random.randint(10, 20)
    s2 = np.random.choice(words, size=n, replace=True, p=p)
    s2 = s2.tolist()

    # y等于s1和s2数值上的相加
    y = int(''.join(s1)) + int(''.join(s2))
    y = list(str(y))

    # x等于s1和s2字符上的相加
    x = s1 + ['a'] + s2

    # 加上首尾符号
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']

    # 补pad到固定长度
    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]

    # 编码成数据
    x = [dictionary_x[i] for i in x]
    y = [dictionary_y[i] for i in y]

    # 转tensor
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)

    return x, y


# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    # 虽然长度定义为100000，但由于数据是动态生成的，实际上可以看作是一个"无限"的数据源
    def __len__(self):
        return 100000

    # 每次调用都生成新数据，不依赖于索引 idx
    def __getitem__(self, idx):
        return get_data()


# 数据加载器
loader = torch.utils.data.DataLoader(
    dataset=Dataset(),      # 数据集实例
    batch_size=8,           # 批次大小
    drop_last=True,         # 是否丢弃最后不完整的批次
    shuffle=True,           # 是否打乱数据顺序
    collate_fn=None         # 批次数据整理函数
)
