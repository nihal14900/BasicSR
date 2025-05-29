import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import time
from thop import profile
from natten.functional import na2d_qk, na2d_av


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from natten.functional import na2d_qk, na2d_av


class SparseMultiRangeCrossAttention(nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        kernel_sizes=[7, 9, 11],
        dilations=[1, 1, 1],
        dim_proj: int = None,  # shared dim to project q/k/v into
        qkv_bias=True,
        qk_scale=None,
        rel_pos_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_checkpoint=False,
        use_mixed_precision=True,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), "kernel_sizes and dilations must be same length"
        self.use_checkpoint = use_checkpoint
        self.use_mixed_precision = use_mixed_precision

        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.dim_proj = dim_proj or dim_q  # fallback to query dim

        self.num_heads = num_heads
        self.head_dim = self.dim_proj // num_heads
        assert self.dim_proj % num_heads == 0, "dim_proj must be divisible by num_heads"
        self.scale = qk_scale or self.head_dim ** -0.5

        self.k = len(kernel_sizes)
        self.kernel_sizes = [(ks, ks) for ks in kernel_sizes]
        self.dilations = [(d, d) for d in dilations]

        self.q_proj = nn.Linear(dim_q, self.dim_proj, bias=qkv_bias)
        self.k_proj = nn.Linear(dim_kv, self.dim_proj, bias=qkv_bias)
        self.v_proj = nn.Linear(dim_kv, self.dim_proj, bias=qkv_bias)

        if rel_pos_bias:
            self.rpb = nn.ParameterList([
                nn.Parameter(torch.zeros(num_heads // self.k, 2 * ks - 1, 2 * ks - 1))
                for ks in kernel_sizes
            ])
        else:
            self.rpb = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(self.dim_proj, dim_q)
        self.out_drop = nn.Dropout(proj_drop)

        self.split_sizes = []
        for i in range(self.k):
            if i == 0:
                ch = self.dim_proj * 3 - (self.dim_proj * 3 // self.k) * (self.k - 1)
            else:
                ch = self.dim_proj * 3 // self.k
            self.split_sizes.append(ch)

    def _cross_attention_block(self, q, k, v, idx):
        B, Hq, Wq, _ = q.shape
        Bk, Hk, Wk, _ = k.shape
        H_i = self.num_heads // self.k
        d = self.head_dim

        q = q.reshape(B, Hq, Wq, H_i, d).permute(0, 3, 1, 2, 4).contiguous()
        k = k.reshape(B, Hk, Wk, H_i, d).permute(0, 3, 1, 2, 4).contiguous()
        v = v.reshape(B, Hk, Wk, H_i, d).permute(0, 3, 1, 2, 4).contiguous()

        q = q * self.scale

        attn = na2d_qk(
            q, k,
            kernel_size=self.kernel_sizes[idx],
            dilation=self.dilations[idx],
            is_causal=False,
            rpb=self.rpb[idx] if self.rpb is not None else None,
        )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = na2d_av(
            attn, v,
            kernel_size=self.kernel_sizes[idx],
            dilation=self.dilations[idx],
            is_causal=False,
        )
        out = out.permute(0, 2, 3, 1, 4).reshape(B, Hq, Wq, H_i * d)
        return out

    def _forward_internal(self, x, context):
        B, _, Hq, Wq = x.shape
        _, _, Hk, Wk = context.shape

        # BCHW -> BHWC
        x = x.permute(0, 2, 3, 1).contiguous()
        context = context.permute(0, 2, 3, 1).contiguous()

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q_chunks = torch.split(q, [ch // 3 for ch in self.split_sizes], dim=-1)
        k_chunks = torch.split(k, [ch // 3 for ch in self.split_sizes], dim=-1)
        v_chunks = torch.split(v, [ch // 3 for ch in self.split_sizes], dim=-1)

        outs = []
        for i in range(self.k):
            out = self._cross_attention_block(q_chunks[i], k_chunks[i], v_chunks[i], i)
            outs.append(out)

        out = torch.cat(outs, dim=-1)
        out = self.out_drop(self.out_proj(out))
        return out.permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW

    from torch.amp import autocast

    def forward(self, x, context):
        if self.use_checkpoint and self.training:
            def custom_forward(*inputs):
                with autocast("cuda", enabled=self.use_mixed_precision):
                    return self._forward_internal(*inputs)
            return torch.utils.checkpoint.checkpoint(custom_forward, x, context)
        else:
            with autocast("cuda", enabled=self.use_mixed_precision):
                return self._forward_internal(x, context)



# ---------------- TEST HARNESS ----------------

def profile_model(model, x, context):
    model.eval()
    with torch.no_grad(), autocast(device_type='cuda'):
        start = time.time()
        out = model(x, context)
        torch.cuda.synchronize()
        end = time.time()
        print(f"Inference time: {(end - start) * 1000:.2f} ms")

    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    macs, params = profile(model, inputs=(x, context), verbose=False)
    print(f"MACs: {macs / 1e9:.2f} G | Params: {params / 1e6:.2f} M")
    return out


def test_cross_smra():
    B, C, H, W = 2, 60, 63, 67  # odd dimensions
    x = torch.randn(B, C, H, W, device='cuda', requires_grad=True)
    context = torch.randn(B, 4*C, H, W).cuda()

    model = SparseMultiRangeCrossAttention(
        dim_q=C,
        dim_kv=4*C,
        num_heads=6,
        kernel_sizes=[7, 9, 11],
        dilations=[4, 4, 4],
        dim_proj=C,
        rel_pos_bias=True,
        attn_drop=0.1,
        proj_drop=0.1,
        use_checkpoint=True,
        use_mixed_precision=True,
    ).cuda()

    print("Sanity check and profiling...")
    y = profile_model(model, x, context)
    assert y.shape == x.shape, f"Expected output shape {x.shape}, got {y.shape}"

    print("Testing mixed precision + checkpointing (train mode)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    for i in range(2):
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            y = model(x, context)
            loss = y.mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"Step {i + 1} done. Loss: {loss.item():.4f}")

    print("✅ Gradient checkpointing + mixed precision training successful!")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    test_cross_smra()
