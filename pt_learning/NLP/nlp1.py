import torch
import torch.autograd as autograd
import torch.nn as nn

#设置随机数种子，产生随机数池，计算机当中不存在真正的随机数，需要初始化随机数序列
#在其他语言中可能是调用系统的当前时间作为随机数种子
#改变括号内参数，会导致下面生成的随机向量数值改变
torch.manual_seed(2)

#设置字典
word_to_ix = {"How": 0, "are": 1, "you":2}

#词库中有3个单词, 每个单词映射到5个维度的随机向量中
embeds = nn.Embedding(3, 5)
#torch.LongTensor是64位带符号的整型
lookup_tensor = torch.LongTensor([word_to_ix["How"],
                                  word_to_ix["are"],
                                  word_to_ix["you"]])
#生成单词的5维的随机向量，这个是初始化状态，尚未训练。
hello_embed = embeds(autograd.Variable(lookup_tensor))
#输出的都是随机数值
print(hello_embed)