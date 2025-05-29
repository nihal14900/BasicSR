import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table

class ChannelWiseSelfAttention(nn.Module):
    """
    Channel-Wise Self-Attention (CW-SA)
    Applies attention across channel dimensions, suitable for input shape (B, C, H, W).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, use_checkpoint=False, use_amp=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_checkpoint = use_checkpoint
        self.use_amp = use_amp

    def forward_function(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial dimensions and transpose to (B, N, C)
        x_flat = x.flatten(2).transpose(1, 2)

        # Linear projection to Q, K, V
        qkv = self.qkv(x_flat)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)

        # Transpose to channel-attention shape
        q = F.normalize(q.transpose(-2, -1), dim=-1)  # (B, num_heads, head_dim, N)
        k = F.normalize(k.transpose(-2, -1), dim=-1)
        v = v.transpose(-2, -1)  # (B, num_heads, head_dim, N)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (B, num_heads, head_dim, head_dim)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)  # (B, num_heads, head_dim, N)
        out = out.transpose(-2, -1).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Reshape back to (B, C, H, W)
        return out.transpose(1, 2).view(B, C, H, W)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            if self.use_amp:
                with autocast():
                    return checkpoint(self.forward_function, x)
            else:
                return checkpoint(self.forward_function, x)
        else:
            if self.use_amp:
                with autocast():
                    return self.forward_function(x)
            else:
                return self.forward_function(x)


# ------------------ TESTING SCRIPT ------------------

def test_cwsa():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define input tensor with non-even spatial size
    B, C, H, W = 2, 96, 65, 87
    x = torch.randn(B, C, H, W).to(device)

    # Instantiate model
    model = ChannelWiseSelfAttention(dim=C, num_heads=8, use_checkpoint=True, use_amp=True).to(device)
    model.eval()

    # Measure inference time
    with torch.no_grad():
        t0 = time.time()
        y = model(x)
        torch.cuda.synchronize()
        t1 = time.time()

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print(f"Inference time: {(t1 - t0) * 1000:.2f} ms")

    # Check FLOPs
    flops = FlopCountAnalysis(model, x)
    print("FLOPs (G):", flops.total() / 1e9)
    print(parameter_count_table(model))

    # Memory usage (approximate, via CUDA)
    torch.cuda.reset_peak_memory_stats()
    _ = model(x)
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"Peak memory usage: {peak_memory:.2f} MB")

    # Sanity check
    assert y.shape == x.shape, "Output shape mismatch"
    print("Sanity check passed.")


if __name__ == "__main__":
    test_cwsa()
