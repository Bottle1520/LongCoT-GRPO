def get_tokens_and_segments(tokens1,tokens2=None):
    tokens=['<cls>']+tokens1+['<seq>']
    segments=[0]*(len(tokens1)+2)#索引
    if tokens2 is not None:
        tokens+=tokens2+['<seq>']
        segments+=[1]*(len(tokens2)+1)
    return tokens,segments
#2实现encoder
class BERTEncoder(nn.Moddule):
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_input
                 ,ffn_num_hiddens,num_heads,num_layers,dropout
                 ,max_len=1000, key_size=768, query_size=768, value_size=768,**kwargs):
        super().__init__(**kwargs)
        self.token_embedding=nn.Embedding(vocab_size,num_hiddens)
        self.segment_embedding=nn.Embedding(2,num_hiddens)
        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}",2l.EncoderBlock(
                      key_size, query_size, value_size, num_hiddens, norm_shape,
                      ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        self.pos_embedding=nn.Parameter(torch.randn(1,max_len,num_hiddens))#可学习位置嵌入
    def forward(self,tokens,segments,valid_lens):
        X=self.token_embedding(tokens)+self.segment_embedding(segments)#注意不是传入的数据
        X=X+self.pos_embedding.data[:, :X.shape[1],:]
        for blk in self.blks:
            X=blk(X,valid_lens)
        return X
#3掩蔽语言模型，掩蔽后让模型预测被掩蔽的位置
class MaskLM(nn.Module):
    def __init__(self,vocab_size,num_hiddens,num_inputs=768,**kwargs):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(num_inputs,num_hiddens),
                               nn.ReLU(),nn.LayerNorm(num_hiddens),
                               nn.Linear(num_hiddens,vocab_size))
        def forward(self,X,pred_positions):#变量记录被mask的token的位置
            num_pred_positions=pred_positions.shape[1]
            pred_positions=pred_positions.reshape(-1)#展平成一维长向量
            batch_size=X.shape[0]
            batch_idx=torch.arrange(0,batch_size)
            batch_idx=torch.repeat_interleave(batch_idx,num_pred_positions)
            masked_X=X[batch_idx,pred_positions]
            masked_X=masked_X.reshape((batch_size,num_pred_positions,-1))#恢复3维
            mlm_Y_hat=self.mlp(masked_X)
            return mlm_Y_hat
#4下一句预测模型
class NextSentencePred(nn.Module):
    def __init__(self, num_inputs, **kwargs):
        super().__init__()
        self.output=nn.Linear(num_inputs,2)#2元输入，判真假
    def forward(self,X):
        return  self.output(X)
