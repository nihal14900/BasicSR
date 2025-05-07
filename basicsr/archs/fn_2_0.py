import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class FreqGroupNorm(nn.Module):
    """
    FreqGroupNorm performs normalization in the frequency domain,
    mimicking GroupNorm: It divides the channel dimension into groups
    and, for each frequency bin (i, j), computes statistics over the channels
    in each group only. Learnable affine parameters (weight and bias) are defined
    per channel, making the module agnostic to the input's spatial resolution.
    
    Given:
        x ∈ ℝ^(B×C×H×W),
    with num_groups = G (must divide C), let M = C/G.
    
    For each group g and frequency bin (i, j) for an instance b:
    
      μ_(b,g,i,j) = (1/M) ∑_{m=1}^{M} x[b, g, m, i, j]
      σ²_(b,g,i,j) = (1/M) ∑_{m=1}^{M} (x[b, g, m, i, j] - μ_(b,g,i,j))²
      
    Then, for each element:
    
      x_norm[b, g, m, i, j] = (x[b, g, m, i, j] - μ_(b,g,i,j)) / √(σ²_(b,g,i,j) + ε)
      
    Finally, reshape back to (B, C, H, W) and apply per-channel affine parameters:
    
      y[b, c, i, j] = γ[c] * x_norm[b, c, i, j] + β[c],
      
    where γ and β are learnable parameters of shape (C,) and are broadcasted over (B, H, W).
    """
    def __init__(self, num_channels: int, num_groups: int, eps: float = 1e-5, affine: bool = True):
        super(FreqGroupNorm, self).__init__()
        assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups."
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine

        if self.affine:
            # Affine parameters are defined per-channel and not per-spatial location.
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (B, C, H, W)
        Returns:
            Normalized tensor with the same shape.
        """
        B, C, H, W = x.shape
        # Reshape x to (B, num_groups, C//num_groups, H, W)
        x = x.view(B, self.num_groups, C // self.num_groups, H, W)
        # Compute mean and variance over the channel dimension within each group,
        # for each instance and each frequency bin (i,j) independently.
        mean = x.mean(dim=2, keepdim=True)       # shape: (B, num_groups, 1, H, W)
        var  = x.var(dim=2, keepdim=True, unbiased=False)  # shape: (B, num_groups, 1, H, W)
        x = (x - mean) / torch.sqrt(var + self.eps)
        # Reshape back to original shape: (B, C, H, W)
        x = x.view(B, C, H, W)
        if self.affine:
            # Apply per-channel affine transformation.
            # weight and bias are of shape (C,) and are broadcast to (B, C, H, W)
            x = self.weight.view(1, C, 1, 1) * x + self.bias.view(1, C, 1, 1)
        return x


# -------------------- Test Code -------------------- #

def test_freq_group_norm_mixed_precision():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create a random input tensor with shape (B, C, H, W).
    # For example: Batch=8, Channels=32, Height=40, Width=40.
    x = torch.randn(8, 32, 40, 40, device=device)
    print("Input shape:", x.shape)

    # Instantiate FreqGroupNorm with num_channels=32 and num_groups=8 (each group has 4 channels).
    norm = FreqGroupNorm(num_channels=32, num_groups=8, eps=1e-5, affine=True).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Use mixed precision if a CUDA device is available.
    with autocast(enabled=(device.type == "cuda")):
        y = norm(x)
    print("Output shape:", y.shape)

    if device.type == "cuda":
        mem_peak = torch.cuda.max_memory_allocated(device)
        print("Peak GPU memory allocated (Mbytes):", mem_peak / 1024 / 1024)

    # Calculate and print total learnable parameters.
    total_params = sum(p.numel() for p in norm.parameters() if p.requires_grad)
    print("Total number of learnable parameters:", total_params)

if __name__ == "__main__":
    test_freq_group_norm_mixed_precision()
