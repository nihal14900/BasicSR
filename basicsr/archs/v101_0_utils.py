# modules/utils.py (UPDATED PixelUnshuffle and PixelShuffle)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3GN(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class PixelUnshuffle(nn.Module):
    """
    Pixel Unshuffle with padding support.
    Returns:
        x_unshuffled: Downsampled tensor
        (orig_H, orig_W): Original spatial dimensions
    """
    def __init__(self, downscale_factor):
        super().__init__()
        self.r = downscale_factor

    def forward(self, x):
        B, C, H, W = x.shape
        pad_H = (self.r - H % self.r) % self.r
        pad_W = (self.r - W % self.r) % self.r

        if pad_H > 0 or pad_W > 0:
            x = F.pad(x, (0, pad_W, 0, pad_H), mode='reflect')

        x_unshuffled = F.pixel_unshuffle(x, self.r)
        return x_unshuffled, (H, W)


class PixelShuffle(nn.Module):
    """
    Pixel Shuffle with unpadding support.
    Takes (x, orig_shape) and removes any padding after upscaling.
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x, orig_shape):
        x_shuffled = F.pixel_shuffle(x, self.r)
        H, W = orig_shape
        return x_shuffled[:, :, :H, :W]


# 🧪 TEST SUITE
def test_utils():
    print("Testing Conv3GN, PixelUnshuffle, and PixelShuffle...")
    B, C, H, W = 2, 16, 73, 91  # Uneven dimensions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(B, C, H, W, device=device)

    # Conv3GN
    conv_block = Conv3GN(C).to(device)
    y = conv_block(x)
    assert y.shape == x.shape, "Conv3GN shape mismatch"
    print("✅ Conv3GN passed.")

    # PixelUnshuffle → PixelShuffle round-trip test
    unshuffle = PixelUnshuffle(2).to(device)
    shuffle = PixelShuffle(2).to(device)

    x_down, orig_shape = unshuffle(x)
    x_up = shuffle(x_down, orig_shape)

    assert x_up.shape == x.shape, f"Shape mismatch after shuffle cycle: {x_up.shape} != {x.shape}"
    print("✅ PixelUnshuffle & PixelShuffle round-trip passed.")

if __name__ == "__main__":
    test_utils()
