import torch
from torch import nn
import math
#0模型加载与数据准备
class model(nn.Module):
    def __init__(self):
        super().__init__()
#1LoRA层
class LoRALayer(nn.Module):
    def __init__(self,in_dim,out_dim,rank,alpha):
        super().__init__()
        self.A=nn.Parameter(torch.empty(in_dim,rank))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))#初始化
        self.B= nn.Parameter(torch.empty(rank,out_dim))#初始B为0，不会影响原权重
        self.alpha=alpha
    def forward(self,x):
        x=self.alpha*(x@self.A@self.B)
#2LinearWithLoRA层的实现,以替换现有的线性层
class LinearWithLoRA(nn.Module):
    def __init__(self,linear,rank,alpha):
        super().__init__()
        self.linear=linear
        self.lora=LoRALayer(linear.in_features,linear.out_features,
                            rank,alpha)
    def forward(self,x):
        return self.linear(x)+self.lora(x)
def replace(model,rank,alpha):
    for name,module in model.named_children():
        if isinstance(module,nn.Linear):
            setattr(model,name,LinearWithLoRA(module,rank,alpha))#name为指针，可实现底层替换
        else:
            replace(module,rank,alpha)#递归地找出嵌套内层的线性层
#3冻结原模型参数，只更新增量
for param in model.parameters():
    param.requires_grad=False
replace(model,rank=16,alpha=16)#replace以原本的linear层为底座，故可保留原来参数
#4使用LoRA层进行微调
