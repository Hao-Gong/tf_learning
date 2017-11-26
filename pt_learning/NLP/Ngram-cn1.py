#基于中文的的N-Gram的文字分析
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#设置Ngram神经网络
class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        #将单词嵌入到embedding_dim维的向量中去
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        #全链接神经网络第一个hidden layer的神经单元数128
        self.linear1 = nn.Linear(context_size * embedding_dim, 500)
        #128个神经单元最终映射到词汇表的大小向量中，毕竟每个单词都要计算概率
        self.linear2 = nn.Linear(500, vocab_size)

    #定义forwar的实现定义
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        #产生概率分布，softmax就是干这个事情的
        log_probs = F.log_softmax(out)
        return log_probs

#产生随机数种子
torch.manual_seed(1)
#采样上下文的大小
CONTEXT_SIZE = 5
#文字映射的向量维度
EMBEDDING_DIM = 10
#我们使用的例文
test_sentence = """如何能够真正的表示出单词间的语义相似度？\
我们可以考虑语义属性。例如我们看到数学家和物理学家都可以跑步，\
是否可以给“可以跑步”这个语义属性很高的分数。考虑到其他的属性，\
并且考虑你怎么给一些常见的单词对这些属性打分。\
假设每个属性是一个维度，我们可以从每一个单词得到一个向量。\
但是这种新的向量会有一个问题，你可以想象出成千上万个属性值，\
但是你怎么给这些属性值打分呢？核心想法就是运用深度学习来学习表达的特征，\
而不是程序员自己人工定义特征。"""

#我们采用包含列表的元组([单词i-2,单词i-1],单词i)的存储方式,
N_grams = [([test_sentence[i+j]for j in range(CONTEXT_SIZE)], test_sentence[i + CONTEXT_SIZE])
            for i in range(len(test_sentence) - CONTEXT_SIZE)]

#词汇表做并运算，得到词库
vocab = set(test_sentence)
#设置词库每个字的序号，生成字典
word_to_ix = {word: i for i, word in enumerate(vocab)}
# print(word_to_ix)

#负对数似然损失函数(Negative Log Likelihood),输入是softmax的时候，其实就是Cross_entropy loss
loss_function = nn.NLLLoss()
#初始化神经网络对象
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
#stochastic gradient descent随机梯度下降办法，这是效率最低的一种
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(300):
    total_loss = torch.Tensor([0])
    for context, target in N_grams:
        #为上文的字查找词库中的序号
        context_idxs = [word_to_ix[w] for w in context]
        #把数据放入pytorch的tensor张量中
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        #清空梯度
        model.zero_grad()
        #将上文N个文字塞入网络中,会生成词库中每个字应该出现的概率
        log_probs = model(context_var)
        #计算损失函数
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))
        #梯度计算，backpropogation
        loss.backward()
        #重新计算网络权值
        optimizer.step()
        #因为是N_gram，所以要把所有loss加起来，是整个文本的loss
        total_loss += loss.data
    if epoch%50==0:
        print("epoch:",epoch,"  loss:", total_loss[0])

#词库list化，便于操作
vocab_list=list(vocab)
#来我们做一点有趣的事情，在上面字库中，我们输入N个字，推算下面一个句子是什么
#这里我就使用了和上面不一样的文字初始化，看看生成的句子有意义吗
init_test="""我们能够从"""
init_test_list=[init_test[j]for j in range(CONTEXT_SIZE)]

init_idx=0
#我们来写完一个句子就结束
while(True):
    init_test_idxs = [word_to_ix[w] for w in init_test_list[init_idx:]]
    context_var = autograd.Variable(torch.LongTensor(init_test_idxs))
    log_probs = model(context_var)
    #view(-1)将1x1的tensor转化成1维向量
    pred_idx = log_probs.data.max(1, keepdim=True)[1].view(-1)[0]
    word_add=vocab_list[pred_idx]
    init_idx+=1
    init_test_list.append(word_add)
    if word_add=='。':
        break

#将list合并成字符串
print(" ".join(init_test_list))