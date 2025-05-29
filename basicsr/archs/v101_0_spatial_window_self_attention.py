import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

def img2windows(img, H_sp, W_sp):
    """Split image into non-overlapping windows."""
    B, C, H, W = img.shape
    pad_r = (W_sp - W % W_sp) % W_sp
    pad_b = (H_sp - H % H_sp) % H_sp
    img = F.pad(img, (0, pad_r, 0, pad_b))  # Pad right and bottom
    Hp, Wp = img.shape[2], img.shape[3]
    img = img.view(B, C, Hp // H_sp, H_sp, Wp // W_sp, W_sp)
    windows = img.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, H_sp * W_sp, C)
    return windows, Hp, Wp

def windows2img(windows, H_sp, W_sp, Hp, Wp, B):
    """Merge windows into full image."""
    C = windows.shape[-1]
    x = windows.view(B, Hp // H_sp, Wp // W_sp, H_sp, W_sp, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
    x = x[:, :Hp, :Wp, :].permute(0, 3, 1, 2).contiguous()
    return x

class CrossSpatialWindowAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, shared_dim=96, num_heads=6, window_size=(8, 8),
                 qkv_bias=True, qk_scale=None, proj_drop=0.0,
                 use_checkpoint=False, use_autocast=True):
        """
        Cross-Spatial Window Attention that supports different query/key-value channel sizes.
        """
        super().__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.shared_dim = shared_dim
        self.num_heads = num_heads
        self.H_sp, self.W_sp = window_size
        self.scale = qk_scale or (shared_dim // num_heads) ** -0.5
        self.use_checkpoint = use_checkpoint
        self.use_autocast = use_autocast

        self.q_proj = nn.Linear(dim_q, shared_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim_kv, shared_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim_kv, shared_dim, bias=qkv_bias)

        self.output_proj = nn.Linear(shared_dim, dim_q)  # back to query's original dim
        self.dropout = nn.Dropout(proj_drop)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.H_sp - 1) * (2 * self.W_sp - 1), num_heads)
        )
        coords_h = torch.arange(self.H_sp)
        coords_w = torch.arange(self.W_sp)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.H_sp - 1
        relative_coords[:, :, 1] += self.W_sp - 1
        relative_coords[:, :, 0] *= 2 * self.W_sp - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _forward(self, x_q, x_kv):
        B, _, H, W = x_q.shape
        q_windows, Hp, Wp = img2windows(x_q, self.H_sp, self.W_sp)
        k_windows, _, _ = img2windows(x_kv, self.H_sp, self.W_sp)
        v_windows, _, _ = img2windows(x_kv, self.H_sp, self.W_sp)

        q = self.q_proj(q_windows).view(-1, self.H_sp * self.W_sp, self.num_heads, self.shared_dim // self.num_heads).transpose(1, 2)
        k = self.k_proj(k_windows).view(-1, self.H_sp * self.W_sp, self.num_heads, self.shared_dim // self.num_heads).transpose(1, 2)
        v = self.v_proj(v_windows).view(-1, self.H_sp * self.W_sp, self.num_heads, self.shared_dim // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rel_bias = rel_bias.view(self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1).permute(2, 0, 1)
        attn = attn + rel_bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, self.shared_dim)
        out = self.output_proj(out)
        out = self.dropout(out)

        out = windows2img(out, self.H_sp, self.W_sp, Hp, Wp, B)
        return out[:, :, :H, :W]

    def forward(self, x_q, x_kv):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x_q, x_kv)
        if self.use_autocast:
            with autocast():
                return self._forward(x_q, x_kv)
        else:
            return self._forward(x_q, x_kv)

# ==================== Test Code ====================

@torch.no_grad()
def profile_module(module, x_q, x_kv):
    torch.cuda.reset_peak_memory_stats()
    module.eval()
    start = time.time()
    with autocast():
        y = module(x_q, x_kv)
    torch.cuda.synchronize()
    print(f"Inference time: {time.time() - start:.4f}s")
    print(f"Output shape: {y.shape}")
    print(f"Max memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

    flops = FlopCountAnalysis(module, (x_q, x_kv))
    print("FLOPs:")
    print(f"{flops.total() / 1e9:.2f} GFLOPs")
    print(parameter_count_table(module))

def sanity_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C_q, C_kv, H, W = 2, 64, 96, 65, 97  # Uneven dims, different channels

    x_q = torch.randn(B, C_q, H, W, device=device)
    x_kv = torch.randn(B, C_kv, H, W, device=device)

    module = CrossSpatialWindowAttention(
        dim_q=C_q, dim_kv=C_kv, shared_dim=128, num_heads=8, window_size=(8, 8),
        use_checkpoint=True, use_autocast=True
    ).to(device)

    output = module(x_q, x_kv)
    assert output.shape == x_q.shape, "Shape mismatch with query input"

    profile_module(module, x_q, x_kv)

if __name__ == "__main__":
    sanity_check()
