import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig
from typing import Optional
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel
#####  attention modules

def repeat_kv(x: torch.Tensor, n_rep: int):
    bs, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:,:,:,None,:]
            .expand(bs,seq_len,n_kv_heads,n_rep,head_dim)
            .reshape(bs,seq_len,n_kv_heads*n_rep, head_dim))

# 注意：此处的dim应为 dim//n_head，因为我们是对每个head进行旋转嵌入
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个从0到end的序列，长度为end
    t = torch.arange(end, device=freqs.device)
    # 计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim=x.ndim
    assert 0<=1<=ndim
    assert freqs_cis.shape==(x.shape[1],x.shape[-1])
    shape = [d if i==1 or i==ndim-1 else 1 for i,d in enumerate(x.shape)]
    return freqs_cis.view(shape)



def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
):
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1]+(-1,2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1]+(-1,2)).unbind(-1)
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r*freqs_cos-xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    def __init__(self,dim :int, eps: float):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))


    def norm(self, x):

        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
    
    def forward(self, x):
        output = self.norm(x.float()).type_as(x)
        return output * self.scale




class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_kv_heads = ars.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.dim%self.n_kv_heads == 0

        self.model_parallel_size = args.model_parallel_size 
        self.n_local_heads = args.n_heads//self.model_parallel_size
        self.n_local_kv_heads = args.n_kv_heads//self.model_parallel_size

        # repeating tims  might be for grouped
        self.n_rep = self.n_local_heads//self.n_local_kv_heads
        self.head_dim = args.dim//args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_local_kv_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_local_kv_heads*self.head_dim, bias=False)

        self.wo = nn.Linear(args.n_heads*self.head_dim, args.dim)


        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            mask = torch.full((1,1,args.max_seq_len, args.max_seq_len),float(-inf))
            mask - torch.triu(mask,diagonal=1)
            self.register_buffer("mask", mask)    


    def forward(self, x, freqs_cos, freqs_sin):
        bs, seq_len, dim = x.shape
        xq,xk,xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs,seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bs,seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs,seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1,2)  # [bs, n_heads, seq_len, head_dim]
        xk = xk.transpose(1,2)  # [bs, n_heads, seq_len, head_dim]
        xv = xv.transpose(1,2)  # [bs, n_heads, seq_len, head_dim]

        

        if self.flash:
            output = F.scaled_dot_product_attention(xq,xk,xv,
                    attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(2,3))
            assert hasattr(self, 'mask')
            scores = scores+self.mask[:,:,:seq_len, : seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output=output.transpose(1,2).contiguous().view(bs,seq_len,-1)
        out = self.wo(output)
        out = self.resid_dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, dropout):

        super().__init__()


        
        if hidden_dim is None:
            hidden_dim = 4*dim
            hidden_dim = int(2*hidden_dim/3)
            hidden_dim = multiple_of* ((hidden_dim+multiple_of-1)//multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三层线性变换，从输入维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    


    

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x))*self.w3(x)))


class DecoderLayer(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim =  args.dim//args.n_heads
        self.attention = Attention(args)
        self.feedforward = MLP(dim=args.dim, 
                               hidden_dim=args.hidden_dim, 
                               multiple_of=args.multiple_of, 
                               dropout=args.dropout)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)

        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x+self.attention(self.attention_norm(x),freqs_cos, freqs_sin)
        out = h+self.feedforward(self.ffn_norm(h))
        return out



class Transformer(PreTrainedModel):
    config_class = ModelConfig  # 配置类
    last_loss: Optional[torch.Tensor] # 记录最后一次计算的损失
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.vocab_size= args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.dropout = nn.Dropout(args.dropout)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mlp = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.tok_embeddings.weight = self.mlp.weight

        freqs_cos,freqs_sin = precompute_freqs_cis(self.args.dim//self.args.n_heads, 
                                                   self.args.max_seq_len)
        
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*args.n_layers))

        self.last_loss = None   
        self.OUT = CausalLMOutputWithPast()
        self._nno_split_modules = [name for name,_ in self.named_modules()]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, tokens, targets=None, **kwargs):
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'labels' in kwargs:
            targets = kwargs['labels']

        _bs, seq_len = tokens.shape

        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        h = self.norm(h)

        if targets is not None:
            logits = self.mlp(h)

            self.last_loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else:
            logits = self.mlp(h[:,[-1],:])
            self.last_loss = None

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    @torch.inference_mode()
    def generate(self, idx, stop_Id=None, max_new_tokens=256,
                 temperature=1.0, top_k=None):
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            logits = self(idx_cond).logits
            logits = logits[:,-1,:]

            if temperature == 0.0:
                _, idx = torch.topk(logits, k=1, dim=-1)

            else:
                logits = logits/temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits<v[:,[-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
        
            if idx_next == stop_Id:
                break
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:] 



if __name__ == '__main__':
        # 创建Attention实例
    args = ModelConfig()
    # attention_model = Attention(args)

    # # 模拟输入数据
    # batch_size = 1
    # seq_len = 50  # 假设实际使用的序列长度为50
    # dim = args.dim
    # x = torch.rand(batch_size, seq_len, dim)  # 随机生成输入张量
    # # freqs_cos = torch.rand(seq_len, dim // 2)  # 模拟cos频率，用于RoPE
    # # freqs_sin = torch.rand(seq_len, dim // 2)  # 模拟sin频率，用于RoPE

    # freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)

    # # 运行Attention模型
    # output = attention_model(x, freqs_cos, freqs_sin)

    # # attention出来之后的形状 依然是[batch_size, seq_len, dim]
    # print("Output shape:", output.shape)

    # mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
    decoderlayer = DecoderLayer(0, args)

# # 模拟输入数据
#     dim = args.dim
#     seq_len = 50

#     x = torch.randn(1, seq_len, dim) # [bs, seq_len, dim]

#     freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)

#     out = decoderlayer(x, freqs_cos, freqs_sin)

#     print(out.shape) # 形状和输入的x一样 [batch_size, seq_len, dim]
# LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
    x = torch.randint(0, 6144, (1, 50)) # [bs, seq_len]
    # 实例化LLaMA2Model
    model = Transformer(args=args)
    # 计算model的全部参数
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', num_params)

    out = model(x)
    print(out.logits.shape) # [batch_size, 1, vocab_size]
