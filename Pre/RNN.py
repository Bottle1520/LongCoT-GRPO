import  math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import  torch as d2l
batch_size,num_steps=32,35
train_iter,vocab=d2l.load_data_fashion_mnist(batch_size,num_steps)
#1采样数据转为独热编码,3个维度分别是时间步长，批量大小，词表大小

#2初始化参数
def get_params(vocab_size,num_hiddens,device):
    num_inputs=num_outputs=vocab_size
    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01
    W_xh=normal((num_inputs,num_hiddens))
    W_hh=normal((num_hiddens,num_hiddens))
    b_h=torch.zeros(num_hiddens,device=device)
    W_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros(num_outputs,device=device)
    params=[W_xh,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params
#3构建循环神经网络模型
def init_rnn_state(batch_size,num_hiddens,device):#隐状态初始化
    return (torch.zeros((batch_size,num_hiddens),device=device),)
def rnn(inputs,state,params):
    W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state
    outputs=[]
    for X in inputs:  #按第一个维度分拆数据（即时间步长）
        H=torch.tanh(torch.mm(X,W_xh)+torch.mm(H,W_hh)+b_h)
        Y=torch.mm(H,W_hq)+b_q
        outputs.append(Y)#拼接成2维张量，便于计算交叉熵损失
    return torch.cat(outputs,dim=0),(H,)#输出H方便在下一个batch中使用
class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device
                 ,get_params,init_state,forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
    def __call__(self, X, state):
        X=F.one_hot(X.T,self.vocab_size).type(torch.float32)#转化独热编码，并使时间步长为第一个维度
        return self.forward_fn(X,state,self.params)#实现主体的rnn前向传播计算
    def begin_state(self,batch_size,device):
        return  self.init_state(batch_size,self.num_hiddens,device)
#4初步预测函数
def predict_ch8(prefix,numpreds,net,vocab,device):
    state=net.begin_state(batch_size=1,device=device)
    outputs=[vocab[prefix[0]]]
    get_input=lambda : torch.tensor([outputs[-1]]).reshape((1,1))#匿名函数
    for y in prefix[1:]:#预热期，不进行预测，会自行更新状态
        _,state=net(get_input(),state)#即之前都call函数
        outputs.append(vocab[y])
    for _ in range(numpreds):#预测
        y,state=net(get_input(),state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join(vocab.idx_to_token[i] for i in outputs)#join用前面的字符把参数内容链接
#5梯度裁剪函数防止梯度爆炸
#6训练网络
def train_epoch_ch8(net,train_iter,loss,updater,device,use_random_iter):#一个迭代周期
    state,timer=None,d2l.Timer()
    metric=d2l.Accumulator(2)
    for X,Y in train_iter:
        y=Y.T.reshape(-1)#拉成一维张量
        X,y=X.to(device),y.to(device)
        y_hat,state=net(X,state)
        l=loss(y_hat,y.long()).mean()
        updater.zero_grad()
        l.backward()
        updater.step()
        metric.add(l*y.numel(),y.numel())
    return  math.exp(metric[0]/metric[1]),metric[1]/timer.stop()
#7多组迭代训练
#8简洁实现
class RNNModel(nn.Module):
    def __init__(self, rnn_layer,vocab_size, **kwargs):
        super(RNNModel,self).__init__(**kwargs)
        self.rnn=rnn_layer
        self.vocab_size=vocab_size
        self.num_hiddens=self.rnn.hidden_size
    def forward(self,inputs,state):
        X=F.one_hot(inputs.T.long(),self.vocab_size)
        X=X.to(torch.float32)
        Y,state=self.rnn(X,state)
        output=self.linear(Y.reshape((-1,Y.shape[-1])))#内层做了展平操作，然后用linear函数输出
        return output,state







