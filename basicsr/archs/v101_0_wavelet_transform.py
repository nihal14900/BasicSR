import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast
from pytorch_wavelets import DWTForward, DWTInverse
from fvcore.nn import FlopCountAnalysis
import time


class DWTWrapper(nn.Module):
    """
    4-band Discrete Wavelet Transform using pytorch_wavelets
    Returns LL, LH, HL, HH.
    """
    def __init__(self, wave='haar', use_checkpoint=False, use_mixed_precision=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_mixed_precision = use_mixed_precision
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)  # single-level DWT

    def _forward(self, x):
        yl, yh = self.dwt(x)
        # Split the high frequency bands
        yh = yh[0]  # shape: [B, C, 3, H/2, W/2]
        lh, hl, hh = yh[:, :, 0], yh[:, :, 1], yh[:, :, 2]
        return yl, lh, hl, hh

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._wrapped, x, use_reentrant=False)
        return self._wrapped(x)

    def _wrapped(self, x):
        with autocast(device_type='cuda', enabled=self.use_mixed_precision):
            return self._forward(x)


class IDWTWrapper(nn.Module):
    """
    Inverse DWT that takes LL, LH, HL, HH and reconstructs the original tensor.
    """
    def __init__(self, wave='haar', use_checkpoint=False, use_mixed_precision=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_mixed_precision = use_mixed_precision
        self.idwt = DWTInverse(mode='zero', wave=wave)

    def _forward(self, ll, lh, hl, hh):
        yh = torch.stack([lh, hl, hh], dim=2)  # shape: [B, C, 3, H, W]
        return self.idwt((ll, [yh]))

    def forward(self, ll, lh, hl, hh):
        if self.use_checkpoint and self.training:
            return checkpoint(self._wrapped, ll, lh, hl, hh, use_reentrant=False)
        return self._wrapped(ll, lh, hl, hh)

    def _wrapped(self, ll, lh, hl, hh):
        with autocast(device_type='cuda', enabled=self.use_mixed_precision):
            return self._forward(ll, lh, hl, hh)


def test_dwt_idwt():
    print("Testing DWT/IDWT using pytorch_wavelets...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    b, c, h, w = 2, 3, 127, 129
    x = torch.randn(b, c, h, w, device=device, dtype=dtype)

    dwt = DWTWrapper(wave='haar', use_checkpoint=True, use_mixed_precision=True).to(device)
    idwt = IDWTWrapper(wave='haar', use_checkpoint=True, use_mixed_precision=True).to(device)

    dwt.train()
    idwt.train()

    ll, lh, hl, hh = dwt(x)
    print(ll.shape)
    x_rec = idwt(ll, lh, hl, hh)
    x_rec = x_rec[:, :, :h, :w]

    err = (x - x_rec).abs().mean().item()
    print(f"Reconstruction error: {err:.6f}")
    assert err < 1e-2, "High reconstruction error"

    # Inference time
    dwt.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            _ = dwt(x)
        torch.cuda.synchronize()
        print(f"DWT avg time: {(time.time() - start) / 10 * 1000:.2f} ms")

    # FLOPs and memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = dwt(x)
    mem = torch.cuda.max_memory_allocated() / 1e6
    flops = FlopCountAnalysis(dwt, x)
    print(f"Memory: {mem:.2f} MB")
    print(f"FLOPs: {flops.total() / 1e6:.2f} MFLOPs")

    print("All tests passed.")


if __name__ == "__main__":
    test_dwt_idwt()
