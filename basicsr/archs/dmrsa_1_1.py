import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class DenseMultiRangeAttention(nn.Module):
    r"""
    Implements Dense (Multi-Range) Attention.

    Given an input feature map X ∈ ℝ^(B x C x H x W), this module computes
    multi-head self-attention over multiple local regions (with sizes defined in
    `range_sizes`) and then concatenates the outputs over all ranges before
    projecting back to the original dimension.

    The same module can be used in both the spatial and frequency domains.
    For a frequency branch, X should be the DCT-transformed features.
    """
    def __init__(self, embed_dim, num_heads=4, range_sizes=[7, 9, 11]):
        """
        Args:
            embed_dim (int): Input (and output) channel dimension.
            num_heads (int): Number of attention heads.
            range_sizes (list of int): List of local region sizes (e.g., 7, 9, 11).
        """
        super(DenseMultiRangeAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projection layers (implemented as 1x1 convolutions)
        self.q_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        # Output projection to fuse concatenated outputs from all ranges
        self.out_proj = nn.Conv2d(embed_dim * len(range_sizes), embed_dim, kernel_size=1)
        
        self.range_sizes = range_sizes
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        """
        Forward pass for Dense Multi-Range Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape

        # Compute Q, K, V via 1x1 conv projections.
        Q = self.q_proj(x)  # (B, C, H, W)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape Q, K, V into (B, num_heads, head_dim, H, W)
        Q = Q.view(B, self.num_heads, self.head_dim, H, W)
        K = K.view(B, self.num_heads, self.head_dim, H, W)
        V = V.view(B, self.num_heads, self.head_dim, H, W)

        # Flatten the spatial dimensions for Q: (B, num_heads, head_dim, H*W)
        Q_flat = Q.view(B, self.num_heads, self.head_dim, H * W)

        outputs = []
        # Process each range size
        for k in self.range_sizes:
            pad = k // 2  # Padding to maintain spatial dimensions

            # Reshape K and V for unfolding: (B*num_heads, head_dim, H, W)
            K_reshaped = K.view(B * self.num_heads, self.head_dim, H, W)
            V_reshaped = V.view(B * self.num_heads, self.head_dim, H, W)

            # Use F.unfold to extract local patches:
            # Output shape: (B*num_heads, head_dim * k*k, H*W)
            K_patches = F.unfold(K_reshaped, kernel_size=k, padding=pad)
            V_patches = F.unfold(V_reshaped, kernel_size=k, padding=pad)

            # Reshape patches: (B, num_heads, head_dim, k*k, H*W)
            K_patches = K_patches.view(B, self.num_heads, self.head_dim, k*k, H*W)
            V_patches = V_patches.view(B, self.num_heads, self.head_dim, k*k, H*W)

            # Compute scaled dot-product attention in the local region.
            # Q_flat has shape: (B, num_heads, head_dim, H*W)
            # K_patches has shape: (B, num_heads, head_dim, k*k, H*W)
            # Using einsum with letters only:
            attn_scores = torch.einsum('bndh,bndrh->bnrh', Q_flat, K_patches) / self.scale
            # attn_scores: (B, num_heads, k*k, H*W)

            # Apply softmax over the patch dimension (k*k)
            attn_weights = F.softmax(attn_scores, dim=2)  # (B, num_heads, k*k, H*W)

            # Compute weighted sum of V patches:
            weighted_sum = torch.einsum('bnrh,bndrh->bndh', attn_weights, V_patches)
            # weighted_sum: (B, num_heads, head_dim, H*W)
            
            outputs.append(weighted_sum)

        # Concatenate outputs from all ranges along the head_dim dimension.
        # The concatenated shape becomes (B, num_heads, head_dim * n_ranges, H*W)
        concat_output = torch.cat(outputs, dim=2)
        # Reshape to (B, num_heads * (head_dim * n_ranges), H, W)
        concat_output = concat_output.view(B, self.num_heads * (self.head_dim * len(self.range_sizes)), H, W)
        
        # Project concatenated output back to the original embed_dim.
        out = self.out_proj(concat_output)
        return out

# Test code to check autocast and memory consumption
if __name__ == "__main__":
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 8, 32, 64, 64
    dummy_input = torch.randn(B, C, H, W).to(device)

    # Instantiate model
    model = DenseMultiRangeAttention(embed_dim=C, num_heads=1, range_sizes=[3, 5]).to(device)
    model.eval()

    # Reset CUDA peak memory stats if using GPU
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    # Wrap forward pass in autocast for mixed precision computations
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            output = model(dummy_input)
    end_time = time.time()

    # Check output shape
    print("Output shape:", output.shape)  # Expected: (B, C, H, W)

    # Report peak GPU memory usage (only if using CUDA)
    if device.type == 'cuda':
        peak_memory_MB = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU memory allocated with autocast: {peak_memory_MB:.2f} MB")

    # Print forward pass time in milliseconds
    print("Forward pass time with autocast: {:.2f} ms".format((end_time - start_time) * 1000))