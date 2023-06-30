---
layout:       post
title:        "Embedding"
author:       "Ke"
header-style: text
catalog:      true
mathjax: true
tags:
    - Embedding
---

# Embedding的详细内容
## Embedding
Embedding在数学上表示一个maping, $f: X \to Y$， 也就是一个function，其中该函数是injective（就是我们所说的单射函数，每个Y只有唯一的X对应，反之亦然）和structure-preserving (结构保存，比如在X所属的空间上$X_1 < X_2$,那么映射后在Y所属空间上同理 $Y_1 < Y_2$)。那么对于word embedding，就是将单词word映射到另外一个空间，其中这个映射具有injective和structure-preserving的特点。通俗的翻译可以认为是单词嵌入，就是把X所属空间的单词映射为到Y空间的多维向量，那么该多维向量相当于嵌入到Y所属空间中，一个萝卜一个坑。

## Word Embedding
参考[作者](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/),
"文本表示"解决的问题是：将「不可计算」的非结构化数据转换成「可计算」的结构化数据。
一个Word Embedding ***W:words***$\to \mathbb{R}^n$是一个参数化的函数,Word Embedding解决的问题是：将文本映射到一个「低维空间」里，通过「低维向量」来表达文本。重点是「低维」。我们可能会发现
$$  
    \mathbf{W("cat")}=(0.2,-0.4,0.7,\ldots)\\
    \mathbf{W("mat")}=(0.0,0.6,-0.1,\ldots)
$$
典型地来讲，这个函数就像是一个查找表，参数化为一个矩阵$\theta$,矩阵地每一行对应一个单词$W_\theta(w_n)=\theta_n$  
对于每个单词而言$W$被随机向量初始化。通过执行某些任务之后，它能够学习到有意义的向量。  
例如，我们训练网络的一项任务是预测一个5-grams（五个单词的序列）是否“有效”我们可以很容易地从维基百科上得到很多5-grams（例如“cat sat on the mat”），然后通过用一个随机单词切换一个单词来“打断”其中的一半（例如“cat sat song the mat”），因为这几乎肯定会使我们的5-grams变得毫无意义。  
我们训练的模型将通过***W***运行5-grams中的每个单词得到一个表示它的向量，并将其输入另一个称为***R***的“模块”.它试图预测5-grams是“valid”还是“broken”  

$R(W(``\text{cat}\!"),~ W(``\text{sat}\!"),~ W(``\text{on}\!"),~ W(``\text{the}\!"),~ W(``\text{mat}\!")) = 1\\
R(W(``\text{cat}\!"),~ W(``\text{sat}\!"),~ W(``\text{song}\!"),~ W(``\text{the}\!"),~ W(``\text{mat}\!")) = 0$
为了准确预测这些值，网络需要学习***W和R***的参数.事实上，对我们来说，任务的全部意义在于学习***W***。

![img](/img/in-post/post-embedding/visualization.png)
<center>t-SNE visualizations of word embeddings. </center>
相似的词相近。另一种方法是观察哪些单词在嵌入过程中最接近给定的单词。

## Patch Embedding
标准 Transformer 使用***一维标记嵌入序列 (Sequence of token embeddings)*** 作为输入。而为了去处理2D图像，首先需要将图像展平成一个一维的序列。所以ViT的首要任务是将图转换成词的结构，这里采取的方法是如上图左下角所示，将图片分割成小块，每个小块就相当于句子里的一个词。这里把每个小块称作Patch，而Patch Embedding就是把每个Patch再经过一个全连接网络压缩成一定维度的向量。
## Position Encoding & Embedding

Transformer中无RNN的循环结构，无法感知一个句子中词语出现的先后顺序，而词语的位置是相当重要的一个信息。作者提出了位置编码，即Position Encoding，来解决这个问题。关于位置编码是否可以训练，也有两种方法：
- 通过训练学习位置编码，也即***Postion Embedding***
- 构造公式来计算位置编码,也即***Position Encoding***

至于具体采用哪一种，作者经过试验后发现两种方式的结果是相似的，所以选择了第二种。毕竟要简单一点，减少了训练参数，而且在训练集中没有出现过的句子长度上也能用。实际上第一种方式也就是

***Position Embedding***:
一种约定俗成的理解是，embedding是可以学习的nn.Embedding.在 [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)中定义。大意就是$(x_0, x_1, x_2, \ldots)$ token序列经过embedding矩阵变成$(w_0, w_1, w_2, ...)$的词向量序列，同时(0,1,2,...)的绝对位置序列也经过另一个embedding矩阵变为$(p_0,p_1,p_2,...)$。最终的embedding序列就是 $e =(x_0+p_0, x_1+p_1, x_2+p_2, ...)$。位置embeding矩阵靠训练学习获得。

***Position Encoding***:
根据一定的编码规则计算出来位置表示，比如Attention is all you need中的计算方式：
$$
    PE_{(pos,2i)}=sin(\frac{pos}{10000}^{\frac{2i}{d_{model}}}) \\ 
    PE_{(pos,2i+1)}=cos(\frac{pos}{10000}^{\frac{2i}{d_{model}}})
$$
可以理解成静态的，即对于每个同样的pos和i结果均一致，pos位置的奇数维度和偶数维度采用不同的编码方式计算。***Position Encoding***在推理时能解决训练时未出现的Position如何编码的问题，***Position Embeding***只能处理训练时出现的Position.


## 推荐算法中的Embedding
参考[作者](https://www.zhihu.com/question/32275069/answer/2774399127)所说
推荐系统中的核心就在于Embedding。Embedding可以将一个概念拆解为特征向量，从而可以提升推荐算法的扩展能力，达到挖掘低频、长尾、小众的内容，实现个性化推荐。
***Embedding将推荐算法从「精确匹配」转化为「模糊查找」，从而能够「举一反三」***

比如在使用倒排索引的召回中，是无法给一个喜欢“科学”的用户，推出一篇带“科技”标签的文章的（不考虑近义词扩展），因为“科学”与“科技”是两个完全独立的词。但是经过Embedding，我们发现“科学”与“科技”两个向量，并不是正交的，而是有很小的夹角。设想一个极其简化的场景，用户向量就用“科学”向量来表示，文章的向量只用其标签的向量来表示，那么用“科学”向量在所有标签向量里做Top-K近邻搜索，一篇带“科技”标签的文章就有机会呈现在用户眼前，从而破除之前“只能精确匹配‘科学’标签”带来的“「信息茧房」”


