########################################################################################################
# RWKV.py
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def RWKV_Init(module, vocab_size, n_embd, rwkv_emb_scale):
    """
    对给定 module 中的所有 nn.Linear / nn.Embedding 做一些初始化.
    """
    for m in module.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = '[unknown weight]'
            for name_, parameter_ in module.named_parameters():
                if id(m.weight) == id(parameter_):
                    name = name_
                    break

            shape = m.weight.data.shape
            gain = 1.0
            scale = 1.0

            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                # final projection?
                if shape[0] == vocab_size and shape[1] == n_embd:
                    scale = rwkv_emb_scale

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                # token emb?
                if shape[0] == vocab_size and shape[1] == n_embd:
                    scale = rwkv_emb_scale

            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            gain *= scale
            if gain == 0:
                nn.init.zeros_(m.weight)
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0, std=-gain)


class RWKV_TimeMix(nn.Module):
    """
    RWKV的 time-mix 注意力部分
    """
    def __init__(self, layer_id, n_embd, n_attn, n_head, ctx_len):
        super().__init__()
        assert n_attn % n_head == 0
        self.layer_id = layer_id
        self.ctx_len = ctx_len
        self.n_head = n_head
        self.head_size = n_attn // n_head

        with torch.no_grad():
            ww = torch.ones(n_head, ctx_len)
            curve = torch.tensor([-(ctx_len - 1 - i) for i in range(ctx_len)])
            for h in range(n_head):
                if h < n_head - 1:
                    decay_speed = math.pow(ctx_len, -(h+1)/(n_head-1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(ctx_len, 1))

        self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)
        self.receptance = nn.Linear(n_embd, n_attn)
        self.output = nn.Linear(n_attn, n_embd)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.shape                 # x: [B, T, C = n_embd]
        TT = self.ctx_len

        # ---- 构造时间衰减核 w：每个 head 一套 Toeplitz 下三角权重 ----
        w = F.pad(self.time_w, (0, TT))   # self.time_w: [H, TT] -> pad右侧TT: [H, 2*TT]
        w = torch.tile(w, [TT])           # 沿最后一维平铺 TT 次: [H, 2*TT*TT]
        w = w[:, :-TT].reshape(-1, TT, 2*TT - 1)  
        # 去掉末尾TT后重排: [H, TT, 2*TT-1]
        w = w[:, :, TT-1:]                # 取右半边（含主对角）: [H, TT, TT]  # 下三角Toeplitz核
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]
        # 裁成当前序列长 T，并做行/列缩放:
        # w: [H, T, T]  (H=n_head)

        # ---- time-mix：只对前半通道做“向前移1步”的时移，再与后半通道不移位部分拼接 ----
        x_shift = torch.cat([self.time_shift(x[:, :, :C//2]),  # ZeroPad2d((0,0,1,-1)) -> [B, T, C//2]
                            x[:, :, C//2:]], dim=-1)          # 拼接 -> x_shift: [B, T, C]

        # ---- 生出 k, v, r ----
        k = self.key(x_shift)             # [B, T, n_attn]
        v = self.value(x_shift)           # [B, T, n_attn]
        r = self.receptance(x_shift)      # [B, T, n_attn]

        # ---- k 做正值强度，并做累积和供归一化 ----
        k = torch.clamp(k, max=30, min=-60)  # 数值稳定
        k = torch.exp(k)                      # k >= 0  # [B, T, n_attn]
        sum_k = torch.cumsum(k, dim=1)        # 逐时刻累积: [B, T, n_attn]

        # ---- 计算 wkv：对每个 head 的 (k*v) 按 w 做时序加权汇总 ----
        kv  = (k * v).view(B, T, self.n_head, self.head_size)   # [B, T, H, Hd]
        # einsum: w[h,t,u] * kv[b,u,h,c] -> out[b,t,h,c]
        wkv = torch.einsum('htu, buhc -> bthc', w, kv)          # [B, T, H, Hd]
        wkv = wkv.contiguous().view(B, T, -1)                   # 拼回: [B, T, n_attn]

        # ---- 门控 + 归一化 + 线性回投 ----
        rwkv = torch.sigmoid(r) * wkv / sum_k    # 逐位置归一化: [B, T, n_attn]
        rwkv = self.output(rwkv)                 # -> [B, T, n_embd = C]

        # ---- 按时间步缩放 ----
        return rwkv * self.time_gamma[:T, :]     # time_gamma[:T,1] 广播 -> [B, T, C]



class RWKV_ChannelMix(nn.Module):
    """
    RWKV 的 channel-mix 部分
    """
    def __init__(self, layer_id, n_embd, n_ffn, hidden_sz, n_attn, n_head, ctx_len):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        hidden_sz = 5*n_ffn//2
        self.key = nn.Linear(n_embd, hidden_sz)
        self.value = nn.Linear(n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, n_embd)
        self.receptance = nn.Linear(n_embd, n_embd)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B,T,C = x.shape
        x_shift = torch.cat([self.time_shift(x[:,:,:C//2]), x[:,:,C//2:]], dim=-1)
        k = self.key(x_shift)
        v = self.value(x_shift)
        r = self.receptance(x_shift)

        wkv = self.weight(torch.nn.functional.mish(k)*v)
        rwkv = torch.sigmoid(r)*wkv
        return rwkv


class Block(nn.Module):
    """
    RWKV Block: LN -> time-mix -> LN -> channel-mix
    """
    def __init__(self, n_embd, n_attn, n_head, ctx_len, n_ffn, hidden_sz, model_type="RWKV", layer_id=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # default model_type='RWKV'
        self.attn = RWKV_TimeMix(layer_id, n_embd, n_attn, n_head, ctx_len)
        self.mlp = RWKV_ChannelMix(layer_id, n_embd, n_ffn, hidden_sz, n_attn, n_head, ctx_len)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


if __name__=="__main__":
    x = torch.randn(4,8,256)
    block = Block(n_embd=256,n_attn=256,n_head=4,ctx_len=300,n_ffn=256,hidden_sz=256)
    y = block.forward(x)
    print("test shape:", y.shape)  # => [4,8,256]
