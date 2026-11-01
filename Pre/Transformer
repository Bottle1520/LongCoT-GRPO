from distutils.command.install import key
from idlelib import query
from itertools import repeat
from nt import stat_result

import torch
import  math
from torch import nn
from d2l import torch as d2l
from torch._C import dtype
#1多头注意力
def transpose_qkv(X,num_heads):#改变矩阵形状,方便并行计算
    X=X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    X=X.permute(0,2,1,3)
    return X.reshape(-1,X.shape[2],X.shape[3])
def transpose_output(X,num_heads):
    X=X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X=X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)
class MultiHeadAttention(nn.Module):
    def __init__(self,key_size,query_size,value_size,
    num_hiddens,num_heads,dropout,bias=False,**kwargs):
        super(MultiHeadAttention).__init__(**kwargs)
        self.num_heads=num_heads
        self.attention=d2l.DotProductAttention(dropout)
        self.W_q=nn.Linear(query_size,num_hiddens,bias=bias)#输入输出维数，并行计算
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    def forward(self,keys,values,queries,valid_lens):
        queries=transpose_qkv(self.W_q(queries),self.num_heads)
        keys=transpose_qkv(self.W_k(keys), self.num_heads)
        values=transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens=repeat_interleave(valid_lens,repeats=self.num_heads,dim=0)#在第一维将数据复制num_heads次
        output=self.attention(queries,keys,values,valid_lens)#valid_lens作为掩码
        output_concat=transpose_output(output,self.num_heads)
        return self.W_o(output_concat)
#2位置编码
class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,dropout,max_len=1000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.P=torch.zeros((1,max_len,num_hiddens))
        X1=torch.arrange(max_len,dtype=torch.float32).reshape(-1,1)
        X2=torch.pow(10000,torch.arrange(0,num_hiddens,2,dtype=torch.float32)/num_hiddens)
        X=X1/X2
        self.P[:,:,0::2]=torch.sin(X)
        self.P[:,:,1::2]=torch.cos(X)
    def forward(self,X):
        X=X+self.P[:, :X.shape[1],:]
        return  self.dropout(X)
#3基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    def __init__(self,num_input,num_hiddens,num_output, **kwargs):
        super(PosionWiseFFN,self).__init__(**kwargs)
        self.dense1=nn.Linear(num_input,num_hiddens)
        self.relu=nn.ReLU()#GELU
        self.dense2=nn.Linear(num_hiddens,num_output)
    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))
#4加法和规范化（残差连接与层规范化）
class AddNorm(nn.Modulle):
    def __init__(self,shape,dropout,**kwargs):
        super(AddNorm,self).__init__(**kwargs)
        self.dropout=nn.Dropout(dropout)
        self.ln=nn.LayerNorm(shape)
    def forward(self,X,Y):
        return  self.ln(self.dropout(Y)+X)
#5编码器
class EncoderBlock(nn.Module):
    def __init__(self,key_size,query_size,value_size,
    num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,
    num_heads,dropout,use_bias=false,**kwargs):
        super(EncoderBlock,self).__init__(**kwargs)
        self.attention=d2l.MultiHeadAttention(key_size,query_size,value_size,
        num_hiddens,num_heads,dropout,use_bias)
        self.addnorm1=AddNorm(norm_shape,dropout)
        self.addnorm2=AddNorm(norm_shape, dropout)
        self.ffn=PositionWiseFFN(ffn_num_input,ffn_num_hiddens,num_hiddens)
    def forward(self,X,valid_lens):
        Y=self.addnorm1(X,self.attention(X,X,X,valid_lens))
        return  self.addnorm2(Y,self.ffn(Y))
class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
    num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding=nn.Embedding(vocab_size,num_hiddens)#对词表进行嵌入编码
        self.pos_encoding=d2l.PositionalEncoding(num_hiddens,dropout)
        self.blks=nn.Sequential()#顺序容器，串起各个神经网络层
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),EncoderBlock(key_size,query_size,
            value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,
             num_heads,dropout,use_bias))#名字和具体模块，输入输出形状相同，可直接传递
    def forward(self,X,valid_lens,*args):
        X=self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self.attention_weights=[None]*len(self.blks)#获取blks的层数并用None占位
        for i,blk in enumerate(self.blks):
            X=blk(X,valid_lens)
            self.attention_weights[i]=blk.attention.attention.attention_weights
            #获取注意力权重矩阵，便于可视化处理
        return  X
#6解码器
class DecoderBlock(nn.Module):#i用来标识第i个块
    def __init__(self, key_size, query_size, value_size,
    num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    num_heads, dropout, i, **kwargs):
        super(DecoderBlock,self).__init__(**kwargs)
        self.i=i
        self.attention1= d2l.MultiHeadAttention(key_size, query_size, value_size,
        num_hiddens, num_heads, dropout)
        self.attention2 = d2l.MultiHeadAttention(key_size, query_size, value_size,
        num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.addnorm3 = AddNorm(norm_shape, dropout)
        self.ffn=PositionWiseFFN(ffn_num_input,ffn_num_hiddens,num_hiddens)#input=num_hiddens,而非ffn_num_hiddens
    def forward(self,X,state):#state实现encoder和decoder之间的串联
        enc_outputs,enc_valid_lens=state[0],state[1]
        if state[2][self.i] is None:
            key_values=X
        else:
            key_values=torch.cat((state[2][self.i],X),axis=1)
        state[2][self.i]=key_values
        if self.training:
            batch_size,num_steps,_=X.shape
            dec_valid_lens=torch.arrange(1,num_steps+1,device=X.device).repeat(batch_size,1)#因果掩码，涉及词之间的交互
        else:
            dec_valid_lens=None
        X2=self.attention1(X,key_values,key_values,dec_valid_lens)#以张量形式传入维度信息
        Y=self.addnorm1(X,X2)
        Y2=self.attention2(Y,enc_outputs,enc_outputs,enc_valid_lens)
        Z=self.addnorm2(Y,Y2)
        return self.addnorm3(Z,ffn(Z)),state
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
    num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding=nn.Embedding(vocab_size,num_hiddens)#对词表进行嵌入编码
        self.pos_encoding=d2l.PositionalEncoding(num_hiddens,dropout)
        self.num_layers=num_layers
        self.blks=nn.Sequential()#顺序容器，串起各个神经网络层
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),DecoderBlock(key_size,query_size,
            value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,
             num_heads,dropout,i))#名字和具体模块，输入输出形状相同，可直接传递
        self.dense=nn.Linear(num_hiddens,vocab_size)#隐藏层映射到词表
    def init_state(self,enc_outputs,enc_valid_lens,*args):
        return[enc_outputs,enc_valid_lens,[None]*self.num_layers]
    def forward(self,X,valid_lens,*args):
        X=self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i,blk in enumerate(self.blks):
            X,state=blk(X,state)
            self._attention_weights[0][i]=blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return  self.dense(X),state
    def attention_weights(self):
        return self._attention_weights
