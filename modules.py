import torch
import torch.nn as nn
from config import ModelConfig

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




if __name__ == "__main__":
    xq = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim
    xk = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim

    # 使用 precompute_freqs_cis 函数获取 sin和cos
    cos, sin = precompute_freqs_cis(288//6, 50)
    print(cos.shape, sin.shape)
    xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)

    

    print(xq_out.shape, xk_out.shape)

