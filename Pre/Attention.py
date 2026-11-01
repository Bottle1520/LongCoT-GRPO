import torch
from torch import nn
from torch._C import dtype
class ScaledDA(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout=dropout
    def forward(self,Q,K,V,valid_lens):
        d_k = Q.size(-1)
        col_indices=torch.arrange(K.shape).unsqueeze(0).unsqueeze(0)
        mask=col_indices>=valid_lens
        scores = torch.bmm(Q, K.transpose(1, 2)) / torch.sqrt(d_k)
        scores=scores.masked_fill(mask,-1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.bmm(attn_weights, V)
        return output, attn_weights
#1多头注意力
def transpose_qkv(X,num_heads):#改变矩阵形状,方便并行计算
    X=X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    X=X.permute(0,2,1,3)
    return X.reshape(-1,X.shape[2],X.shape[3])#适配bmm
def transpose_output(X,num_heads):
    X=X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X=X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)
class MultiHeadAttention(nn.Module):
    def __init__(self,key_size,query_size,value_size,
    num_hiddens,num_heads,dropout,bias=False,**kwargs):
        super(MultiHeadAttention).__init__(**kwargs)
        self.num_heads=num_heads
        self.attention=ScaledDA(dropout)
        self.W_q=nn.Linear(query_size,num_hiddens,bias=bias)#输入输出维数，并行计算
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    def forward(self,keys,values,queries,valid_lens,layer_past):#映射--拆分--运算--拼接
        queries=transpose_qkv(self.W_q(queries),self.num_heads)
        keys=transpose_qkv(self.W_k(keys), self.num_heads)
        values=transpose_qkv(self.W_v(values), self.num_heads)
        if layer_past is not None:
            k_past,v_past=layer_past
            keys=torch.cat([k_past,keys],dim=-2)
            values=torch.cat([v_past, values], dim=-2)#取出元组中元素并拼接
        layer_present=(keys,values)
        output=self.attention(queries,keys,values,valid_lens)#valid_lens作为掩码
        output_concat=transpose_output(output,self.num_heads)
        return self.W_o(output_concat),layer_present
class PositionalEncoding(nn.Module):

    def __init__(self,num_hiddens,dropout,max_len):
        super().__init__()
        self.d=num_hiddens
        self.dropout=nn.Dropout(dropout)
        P=torch.zeros(1,max_len,num_hiddens)#注意三维，可利用广播机制
        X1=torch.arange(max_len).reshape(-1,1)
        X2=torch.pow(1000,torch.range(0,num_hiddens,2,dtype="float32")/self.d)
        Y=X1/X2
        self.P[:,:,0::2]=torch.sin(Y)
        self.P[:,:,1::2] = torch.cos(Y)
    def forward(self,X):
        X+=self.P[:,X.shape[1],:]
        return self.dropout(X)
class RoPE(nn.Module):#旋转位置编码，2维子空间分划
    def __init__(self,num_hiddens,dropout,max_len):
        super().__init__()
        theta=torch.pow(10000,torch.range(0,end=num_hiddens/2,step=2,dtype='float32')/num_hiddens)
        P=torch.range(max_len).reshape(-1,1)
        pos1=(P*theta).reshape(1,max_len,-1)
        self.cos=torch.cos(pos1)
        self.sin=torch.sin(pos1)
        self.dropout=nn.Dropout(dropout)
        self.num_hiddens=num_hiddens
    def trans(self,X):
        Y=torch.zeros(X.shape)
        Y[:,:,:self.num_hiddens/2]=-X[:,:,self.num_hiddens/2:]
        Y[:, :,self.num_hiddens / 2:] = X[:, :, :self.num_hiddens/2]
        return Y
    def forward(self,X):
        Y=X*self.cos[:,:X.shape[1],:]+self.trans(X)*self.sin[:,:X.shape[1],:]
        return self.dropout(Y)
