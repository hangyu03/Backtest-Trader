# DLinear_v10.py
# ========================================================================
import torch
import torch.nn as nn
import numpy as np

from .RWKV import Block, RWKV_Init


def DLinear_Init(module, min_val=-5e-2, max_val=8e-2):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, min_val, max_val)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend.
    x: [B,T,C]
    """
    def __init__(self, kernel_size=8, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        B,T,C = x.shape
        front = x[:,0:1,:].repeat(1,(self.kernel_size-1)//2,1)
        end = x[:,-1:,:].repeat(1,(self.kernel_size-1)//2,1)
        x_pad = torch.cat([front,x,end], dim=1)
        x_perm = x_pad.permute(0,2,1)
        x_avg = self.avg(x_perm)
        return x_avg.permute(0,2,1)


class series_decomp(nn.Module):
    """
    Series decomposition:
      - 128->256->Block->256->128
      - final_trend = a*moving_mean + (1-a)*rwkv_out
      - residual = x - final_trend
    """
    def __init__(self, kernel_size, enc_in=128, n_head=4, ctx_len=300):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size)

        # project in: [128 -> 256]
        self.proj_in = nn.Linear(enc_in, 256)

        # RWKV block(256)
        self.rwkv = Block(
            n_embd=256,
            n_attn=256,
            n_head=n_head,
            ctx_len=ctx_len,
            n_ffn=256,
            hidden_sz=256,
            model_type="RWKV",
            layer_id=0
        )
        RWKV_Init(self.rwkv, vocab_size=256, n_embd=256, rwkv_emb_scale=1.0)

        # project out: [256 -> 128]
        self.proj_out = nn.Linear(256, enc_in)

        self.a = nn.Parameter(torch.tensor(0.6), requires_grad=True)

    def forward(self, x):
        """
        x: [B,T,128]
        => moving_mean => [B,T,128]
        => x->proj_in(128->256)->rwkv->256->proj_out->128
        => final_trend
        => residual
        => return residual, final_trend
        """
        moving_mean = self.moving_avg(x)

        x_256 = self.proj_in(x)
        y_256 = self.rwkv(x_256)
        rwkv_out = self.proj_out(y_256)

        final_trend = moving_mean*self.a + rwkv_out*(1 - self.a)
        residual = x - final_trend
        return residual, final_trend


class DLinear(nn.Module):
    """
    Decomposition-Linear v10
    - seq_len, pred_len
    - enc_in=128
    - kernel_size
    - series_decomp => [B,T,128]
    - linear => => [B,pred_len,128]
    - return (trend_output, seasonal_output)
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in=128,
        kernel_size=3,
        individual=False,
        n_head=4,
        ctx_len=300
    ):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = enc_in

        # series_decomp: do RWKV(256) inside
        self.decompsition = series_decomp(
            kernel_size=kernel_size,
            enc_in=enc_in,
            n_head=n_head,
            ctx_len=ctx_len
        )

        if individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        """
        x: [B,T,128]
        => decompsition => (residual, trend) => [B,T,128], [B,T,128]
        => permute => linear => permute => => (trend_output, season_output)
        """
        seasonal_init, trend_init = self.decompsition(x)  # => [B,T,128],[B,T,128]

        # => [B,128,T]
        seasonal_init = seasonal_init.permute(0,2,1)
        trend_init = trend_init.permute(0,2,1)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device
            )
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # => [B,pred_len,128]
        seasonal_output = seasonal_output.permute(0,2,1)
        trend_output = trend_output.permute(0,2,1)

        return trend_output, seasonal_output


# test
if __name__=="__main__":
    x_sample = torch.randn(4,8,128)
    model = DLinear(seq_len=8, pred_len=8, enc_in=128, kernel_size=5, n_head=4, ctx_len=300)
    DLinear_Init(model)
    trend, season = model(x_sample)
    print("trend.shape:", trend.shape)   # [4,8,128]
    print("season.shape:", season.shape) # [4,8,128]
