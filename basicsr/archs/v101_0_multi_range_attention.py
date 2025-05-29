import torch
import torch.nn as nn
import torch.nn.functional as F
from natten.functional import na2d_qk, na2d_av
from torch.utils.checkpoint import checkpoint
from torch.nn.init import trunc_normal_
from fvcore.nn import FlopCountAnalysis, parameter_count
import time

class MultiRangeAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 kernel_sizes: list,
                 dilations: list = None,
                 use_checkpoint: bool = False,
                 use_mixed_precision: bool = False,
                 qkv_bias: bool = True,
                 proj_drop: float = 0.0,
                 attn_drop: float = 0.0,
                 use_rpb: bool = True):
        """
        Multi-Range Attention module with relative positional bias.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_sizes = kernel_sizes
        self.k = len(kernel_sizes)
        self.dilations = dilations if dilations is not None else [1] * self.k
        self.use_checkpoint = use_checkpoint
        self.use_mixed_precision = use_mixed_precision
        self.use_rpb = use_rpb

        # Split QKV into k branches
        self.channels = []
        for i in range(self.k):
            if i == 0:
                ch = dim * 3 - dim * 3 // self.k * (self.k - 1)
            else:
                ch = dim * 3 // self.k
            assert ch % 3 == 0
            self.channels.append(ch)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_rpb:
            self.rpb = nn.ParameterList()
            for ks in kernel_sizes:
                head_dim = num_heads // self.k
                size = 2 * ks - 1
                param = nn.Parameter(torch.zeros(head_dim, size, size))
                trunc_normal_(param, std=0.02)
                self.rpb.append(param)
        else:
            self.rpb = [None] * self.k

    def forward_branch(self, x_branch, kernel_size, dilation, rpb):
        B, H, W, C = x_branch.shape
        heads = self.num_heads // self.k
        head_dim = self.dim // self.num_heads
        qkv = x_branch.view(B, H, W, 3, heads, head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = head_dim ** -0.5
        q = q * scale

        attn = na2d_qk(
            q, k,
            kernel_size=(kernel_size, kernel_size),
            dilation=(dilation, dilation),
            rpb=rpb
        )
        attn = self.attn_drop(attn.softmax(dim=-1))
        y = na2d_av(
            attn, v,
            kernel_size=(kernel_size, kernel_size),
            dilation=(dilation, dilation)
        )
        y = y.permute(0, 2, 3, 1, 4).reshape(B, H, W, C // 3)
        return y

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        qkv = self.qkv(x)
        branches = torch.split(qkv, self.channels, dim=-1)

        results = []

        for i in range(self.k):
            fn = lambda b: self.forward_branch(b, self.kernel_sizes[i], self.dilations[i], self.rpb[i])
            if self.use_checkpoint:
                y = checkpoint(fn, branches[i])
            else:
                y = fn(branches[i])
            results.append(y)

        out = torch.cat(results, dim=-1)
        out = self.proj_drop(self.proj(out))
        out = out.permute(0, 3, 1, 2)  # BHWC -> BCHW
        return out


# -------------------------------
# Test & Profiling
# -------------------------------

def profile_module(module, x):
    flops = FlopCountAnalysis(module, x)
    params = parameter_count(module)
    print("FLOPs: {:.2f}M".format(flops.total() / 1e6))
    print("Parameters: {:.2f}K".format(params[''] / 1e3))

def test_multi_range_attention(use_checkpoint=False, use_amp=False):
    print(f"\n[TEST] Checkpoint: {use_checkpoint} | AMP: {use_amp}")
    B, C, H, W = 2, 60, 61, 67  # Non-even spatial dims
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiRangeAttention(
        dim=C,
        num_heads=6,
        kernel_sizes=[7, 9, 11],
        dilations=[1, 1, 1],
        use_checkpoint=use_checkpoint,
        use_mixed_precision=use_amp,
        use_rpb=True
    ).to(device)

    x = torch.randn(B, C, H, W).to(device)

    # profile_module(model, x)

    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        start = time.time()
        out = model(x)
        torch.cuda.synchronize()
        end = time.time()

    print("Output shape:", out.shape)
    print("Inference time: {:.2f} ms".format((end - start) * 1000))
    print("Peak memory (MB): {:.2f}".format(torch.cuda.max_memory_allocated() / 1024**2))
    torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    test_multi_range_attention(use_checkpoint=False, use_amp=False)
    test_multi_range_attention(use_checkpoint=True, use_amp=False)
    test_multi_range_attention(use_checkpoint=False, use_amp=True)
    test_multi_range_attention(use_checkpoint=True, use_amp=True)
