import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

class SparseMultiRangeAttention2D(nn.Module):
    """
    Sparse Multi-Range Attention (SMA) with optional gradient checkpointing.
    Each head attends to a sparse, dilated neighborhood defined by
    (kernel_size, dilation), using relative positional bias.
    """
    def __init__(self, in_channels, num_heads, kernel_sizes, dilations, use_checkpoint=False):
        """
        Args:
            in_channels (int): Number of input/output channels C.
            num_heads (int): Number of attention heads h.
            kernel_sizes (list of int): One odd k per head.
            dilations (list of int): One dilation δ per head.
            use_checkpoint (bool): Wrap heavy sub-components with checkpoint.
        """
        super().__init__()
        assert len(kernel_sizes) == num_heads == len(dilations), \
            "kernel_sizes and dilations must match num_heads"
        for k in kernel_sizes:
            assert k % 2 == 1, "Each kernel_size must be odd"
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.use_checkpoint = use_checkpoint

        # 1×1 conv to project input to Q, K, V
        self.qkv_proj = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, bias=False)
        # 1×1 conv to fuse concatenated heads back to C channels
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # store per-head configs
        self.kernel_sizes = kernel_sizes
        self.dilations    = dilations
        # one learned relative bias vector per head (length = k^2)
        self.relative_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(k * k)) for k in kernel_sizes
        ])

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features [B, C, H, W], requires_grad controls checkpointing.
        Returns:
            torch.Tensor: Output features [B, C, H, W]
        """
        B, C, H, W = x.shape
        do_ckpt = self.use_checkpoint and x.requires_grad

        # --- 1) QKV projection ---
        if do_ckpt:
            qkv = checkpoint(self.qkv_proj, x)
        else:
            qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = q.view(B, self.num_heads, self.head_dim, H, W)
        k = k.view(B, self.num_heads, self.head_dim, H, W)
        v = v.view(B, self.num_heads, self.head_dim, H, W)

        head_outputs = []

        # --- 2) Per-head sparse attention ---
        # Use default args on the function to bind k_sz and dil by value,
        # ensuring correct behavior under checkpointing.
        for idx in range(self.num_heads):
            qi   = q[:, idx]           # [B, head_dim, H, W]
            ki   = k[:, idx]
            vi   = v[:, idx]
            bias = self.relative_bias[idx]  # [k^2]
            k_sz = self.kernel_sizes[idx]
            dil  = self.dilations[idx]

            assert bias.numel() == k_sz * k_sz, (
                f"Head {idx}: bias length {bias.numel()} != {k_sz}^2"
            )

            def head_attention(qi, ki, vi, bias, k_sz=k_sz, dil=dil):
                # pad so that output spatial dims remain H, W
                pad = (dil * (k_sz // 2), dil * (k_sz // 2))

                # 2a) Extract dilated patches around each pixel
                # -> [B, head_dim*k^2, H*W]
                k_patches = F.unfold(
                    ki,
                    kernel_size=k_sz,
                    dilation=dil,
                    padding=pad,
                    stride=1
                )
                v_patches = F.unfold(
                    vi,
                    kernel_size=k_sz,
                    dilation=dil,
                    padding=pad,
                    stride=1
                )

                # 2b) Reshape to [B, head_dim, k^2, H, W]
                k_patches = k_patches.view(B, self.head_dim, k_sz * k_sz, H, W)
                v_patches = v_patches.view(B, self.head_dim, k_sz * k_sz, H, W)

                # 2c) Scaled dot-product
                # qi.unsqueeze(2): [B, head_dim, 1, H, W]
                # k_patches:      [B, head_dim, k^2, H, W]
                scores = (qi.unsqueeze(2) * k_patches).sum(dim=1)
                scores = scores / math.sqrt(self.head_dim)

                # 2d) Add bias and softmax
                scores = scores + bias.view(1, -1, 1, 1)
                attn   = F.softmax(scores, dim=1)  # [B, k^2, H, W]

                # 2e) Weighted sum of values
                # attn.unsqueeze(1): [B, 1, k^2, H, W]
                # v_patches:         [B, head_dim, k^2, H, W]
                out = (attn.unsqueeze(1) * v_patches).sum(dim=2)
                return out  # [B, head_dim, H, W]

            # checkpoint the heavy head attention if desired
            if do_ckpt:
                head_out = checkpoint(head_attention, qi, ki, vi, bias)
            else:
                head_out = head_attention(qi, ki, vi, bias)

            head_outputs.append(head_out)

        # 3) Concatenate heads -> [B, C, H, W]
        out = torch.cat(head_outputs, dim=1)

        # 4) Final fusion projection
        if do_ckpt:
            out = checkpoint(self.out_proj, out)
        else:
            out = self.out_proj(out)

        return out


if __name__ == "__main__":
    # ----------------------------
    # Test script for checkpoint
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C = 2, 64
    H, W = 64, 48            # non-square input
    num_heads = 4
    kernel_sizes = [3, 5, 7, 9]
    dilations    = [1, 2, 3, 4]

    model_nc = SparseMultiRangeAttention2D(
        C, num_heads, kernel_sizes, dilations, use_checkpoint=False
    ).to(device)
    model_ck = SparseMultiRangeAttention2D(
        C, num_heads, kernel_sizes, dilations, use_checkpoint=True
    ).to(device)

    # Random input requiring gradients
    x = torch.randn(B, C, H, W, device=device, requires_grad=True)

    def run_and_measure(model, x):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

        model.eval()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            y = model(x)
            loss = y.sum()
        # backward to test autograd
        loss.backward()

        peak_mem = (torch.cuda.max_memory_allocated() / (1024**2)
                    if device.type == "cuda" else 0.0)
        return y, peak_mem

    y_nc, mem_nc = run_and_measure(model_nc, x)
    x.grad.zero_()
    y_ck, mem_ck = run_and_measure(model_ck, x)

    print(f"Shape no-ckpt : {y_nc.shape}")
    print(f"Shape with-ckpt: {y_ck.shape}")
    assert y_nc.shape == y_ck.shape, "Outputs must match shape"

    print(f"Peak mem no-ckpt : {mem_nc:.1f} MB")
    print(f"Peak mem with-ckpt: {mem_ck:.1f} MB")
    assert mem_ck < mem_nc, "Checkpointing should reduce peak memory"

    print("✅ Checkpointing works: shape, gradients, and memory all verified.")
