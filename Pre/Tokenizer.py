import collections
import re
from d2l import torch as d2l
#1拆词函数
def tokenize(lines,token='word'):#按行作为字符串输入
    if token=='word':
        return [line.split() for line in lines]#split函数按词拆分
    if token=='char':
        return [list(line) for line in lines]#list函数按字符拆分
    else:
        print("error")
#2词频统计函数（将文本展为一维列表以使用counter函数）
def count_corpus(tokens):
    if len(tokens)==0 or isinstance(tokens[0],list):
        tokens=[token for line in tokens for token in line]#展平操作
    return collections.Counter(tokens)

#3构建词表（映射到数）
class Vocab:
    def __init__(self,tokens=None,min_freq=0,reserved_tokens=None):
        if tokens is None:
            tokens=[]
        if reserved_tokens is None:
            reserved_tokens=[]#形参赋值及两个if的作用，涉及可变参数
        counter=count_corpus(tokens)
        self._token_freqs=sorted(counter.items(),
                                 key=lambda x:x[1],
                                 reverse=True)#构建元组，按词频从高到低排序
        self.idx_to_token=["<unk>"]+reserved_tokens#构建词元索引列表，第一项是未确认，第二项是保留字
        self.token_to_idx={token:idx for idx,token in enumerate(self.idx_to_token)}
                                             #生成词元到索引的反向查找字典
        for token,freq in self._token_freqs:
            if freq<min_freq:
                break
            if token not in self.token_to_idx:#字典索引方式
                self.idx_to_token.append(token)
                self.token_to_idx[token]=len(self.idx_to_token)-1
            ##从文本里提取的词元添加到两个表中
    def __len__(self):
        return  len(self.idx_to_token)
    def unk(self):
        return 0
    def __getitem__(self, tokens):
        if not isinstance(tokens,(list,tuple)):
            return self.token_to_idx.get(tokens,self.unk)
        return [self.__getitem__(token) for token in tokens]
       ##递归地输入文本获取索引，保持tokens的结构（一维或二维）
    def to_tokens(self,indices):
        if not isinstance(indices,(list,tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
      ##使用列表输入索引返回词元
    def token_freqs(self):
        return  self._token_freqs

###此处标记有一段token展平的操作需要理解
