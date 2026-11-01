from cmath import sqrt
import torch
from torch import nn
class MLP(nn.Module):
    def __init__(self, out_dim,in_dim):
        super().__init__()
        self.out_dim=out_dim
        self.in_dim=in_dim
        self.W=torch.randn(in_dim,out_dim,requires_grad=True)
        self.b=torch.zeros(out_dim,requires_grad=True)
    def forward(self,X):
        Y=torch.matmul(X,self.W)+self.b
        Z=torch.relu(X)
        return Z
def softmax(X):
    Y=torch.exp(X)
    part=Y.sum(2,keepdim=True)
    return Y/part
def loss(X,Y):
    X=X.reshape(-1,X.shape[2])
    Y=Y.reshape(-1)
    loss=-torch.log(X[range(len(Y)),Y])
    return loss.mean()
###优化器编写
class MyOptim():
    def __init__(self,params,lr):
        self.lr=lr
        self.params=list(params)
    def step(self):
        with torch.no_grad():
            for param in self.params:
                param-=self.lr*param.grad
    def zero_grad(self):
        for p in self.params:
            p.grad.zero()
#LayerNorm
class LayerNorm(nn.Module):
    def __init__(self,dim,eps=1e-5):
        super().__init__()
        self.dim=dim
        self.gama=nn.Parameter(torch.ones(dim))
        self.beta=nn.Parameter(torch.zeros(dim))
        self.eps=eps
    def forward(self,X):
        X_mean=X.mean(2,keepdims=True)
        X_var=X.var(-1,unbiasd=False,keepdims=True)
        X_norm=(X-X.mean)/sqrt(X.var)+self.eps
        return  self.gama*X_norm+self.beta
##Dropout
def dropout(X,drate):
    assert 0<=drate<=1
    if drate==1:
        X=torch.zeros(X.shape)
    if drate==0:
        return X
    mask=(torch.rand(X.shape)>drate).float()
    return mask*X/1-drate
#rnn反向传播手撕(BPTT)
class rnn(nn.Module):
    def __init__(self,input,output,hiddens):
        self.W_xh=torch.randn(input,hiddens)
        self.W_hh=torch.randn(hiddens,hiddens)
        self.W_hq=torch.randn(hiddens,output)
    def forward(self,X,t):
        y=[]
        h_list=[torch.zeros(X.shape).reshape(-1,self.hiddens)]
        for t in range(t):
            Y=torch.matmul(X,self.W_xh)
            H=h_list[-1]@self.W_hh+Y
            h_list.append(H)
            y.append(H@self.W_hq)
        return y,h_list
    def backward(self,t,X,y_true):
        y,h=self.forward(self,X,t)
        dW_xh = torch.zeros_like(self.W_xh)
        dW_hh = torch.zeros_like(self.W_hh)
        dW_hq = torch.zeros_like(self.W_hq)
        dh_next=torch.zeros(X.shape).reshape(-1,self.hiddens)
        for t in range(t):
            dy=y-y_true
            dW_hq=dy*h[t+1]
            dh=dy*self.W_hq+dh_next
            dW_xh=dh@X[t]
            dW_hh=dh*h[t]
            dh_next=dh*self.W_hh#从后一个时间步向前累加而得
        params=[dW_xh,dW_hh,dW_hq]
        return params
