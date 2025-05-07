import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_dct as dct

"""
This code implements Channel-Wise Cross-Attention (Frequency → Spatial) using the chunking technique.
Spatial features are converted to frequency space via a 2D DCT (using torch_dct.dct_2d with norm='ortho').
Then, queries are obtained from frequency features (already in DCT domain) and keys/values are obtained
from the converted spatial features. The attention is computed exactly but in chunks to reduce the memory footprint.
"""

# ---------------------------
# DCT Operation using torch-dct
# ---------------------------
def dct_2d(x):
    """
    Applies the 2D DCT-II on the last two dimensions using torch-dct.
    
    Args:
        x (Tensor): Input tensor of shape (B, C, H, W) in the spatial domain.
        
    Returns:
        Tensor: Output tensor in the frequency domain of shape (B, C, H, W).
    """
    return dct.dct_2d(x, norm='ortho')

# ---------------------------
# Chunked Attention Helper
# ---------------------------
def chunked_attention(Q, K, V, chunk_size):
    """
    Computes scaled dot-product attention in chunks along the sequence length dimension,
    thereby avoiding the creation of a full (HW x HW) attention matrix.
    
    Args:
        Q (Tensor): Query tensor of shape (B, num_heads, L, d)
        K (Tensor): Key tensor of shape (B, num_heads, S, d) with S = H*W.
        V (Tensor): Value tensor of shape (B, num_heads, S, d)
        chunk_size (int): The size of chunks to process at a time.
        
    Returns:
        Tensor: Attention output of shape (B, num_heads, L, d)
    """
    B, h, L, d = Q.shape
    outputs = []
    scale = 1.0 / np.sqrt(d)
    # Iterate over query in chunks of chunk_size (L == H*W)
    for i in range(0, L, chunk_size):
        Q_chunk = Q[:, :, i:i+chunk_size, :]  # (B, h, chunk_size, d)
        # Compute attention scores for this chunk: (B, h, chunk_size, S)
        attn_scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        output_chunk = torch.matmul(attn_weights, V)  # (B, h, chunk_size, d)
        outputs.append(output_chunk)
    return torch.cat(outputs, dim=2)

# ---------------------------
# Channel-Wise Cross-Attention (Frequency -> Spatial) with Chunking
# ---------------------------
class ChannelWiseCrossAttention_FrequencyToSpatial_Chunk(nn.Module):
    def __init__(self, in_channels, num_heads=8, chunk_size=128):
        """
        Initializes the Channel-Wise Cross-Attention block.
        Query is taken from frequency features (in DCT domain),
        while key and value are derived from spatial features (converted to frequency via DCT).
        The attention computation is performed in chunks along the sequence dimension to save memory.
        
        Args:
            in_channels (int): Number of channels.
            num_heads (int): Number of attention heads.
            chunk_size (int): Chunk size for processing the attention scores.
        """
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.chunk_size = chunk_size
        
        # 1x1 Convolutions as linear projections.
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, freq_feat, spatial_feat):
        """
        Forward pass for Channel-Wise Cross-Attention (Frequency -> Spatial) with chunking.
        
        Args:
            freq_feat (Tensor): Frequency features (in DCT domain), shape (B, C, H, W).
            spatial_feat (Tensor): Spatial features (raw), shape (B, C, H, W).
            
        Returns:
            Tensor: Output feature map of shape (B, C, H, W).
        """
        B, C, H, W = freq_feat.shape
        assert spatial_feat.shape == (B, C, H, W), "Spatial and frequency features must have the same shape"
        
        # Step 1: Convert spatial features into the frequency domain using DCT.
        spatial_freq = dct_2d(spatial_feat)  # (B, C, H, W)
        
        # Step 2: Compute linear projections.
        Q = self.q_proj(freq_feat)         # (B, C, H, W) from frequency features.
        K = self.k_proj(spatial_freq)        # (B, C, H, W) from spatial features converted to frequency.
        V = self.v_proj(spatial_freq)        # (B, C, H, W)
        
        # Step 3: Reshape for multi-head attention.
        # From (B, C, H, W) to (B, num_heads, head_dim, H*W)
        Q = Q.view(B, self.num_heads, self.head_dim, H * W)
        K = K.view(B, self.num_heads, self.head_dim, H * W)
        V = V.view(B, self.num_heads, self.head_dim, H * W)
        
        # Permute to (B, num_heads, H*W, head_dim)
        Q = Q.permute(0, 1, 3, 2).contiguous()
        K = K.permute(0, 1, 3, 2).contiguous()
        V = V.permute(0, 1, 3, 2).contiguous()
        
        # Step 4: Compute scaled dot-product attention in chunks.
        # Q: (B, num_heads, L, head_dim) where L = H*W.
        attn_output = chunked_attention(Q, K, V, self.chunk_size)  # (B, num_heads, L, head_dim)
        
        # Step 5: Reassemble multi-head output and project.
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous()  # (B, num_heads, head_dim, L)
        attn_output = attn_output.view(B, C, H, W)  # Reshape to (B, C, H, W)
        out = self.out_proj(attn_output)
        
        # Debug prints: output shapes and peak GPU memory usage.
        # print("Frequency Input Shape:", freq_feat.shape)
        # print("Spatial Input Shape:", spatial_feat.shape)
        # print("Spatial Converted to Frequency (after DCT):", spatial_freq.shape)
        # print("Output Shape:", out.shape)
        # if freq_feat.is_cuda:
        #     mem_MB = torch.cuda.memory_allocated() / (1024 ** 2)
        #     print(f"Peak GPU Memory Allocated: {mem_MB:.2f} MB")
            
        return out

# ---------------------------
# Testing Script: Shape Checking, Memory Consumption, Mixed Precision
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set dummy input dimensions (B, C, H, W)
    B, C, H, W = 8, 32, 64, 64  # adjust as needed; note: higher C or spatial dims mean larger matrices
    # Frequency features (in DCT domain)
    freq_input = torch.randn(B, C, H, W).to(device)
    # Spatial features (raw)
    spatial_input = torch.randn(B, C, H, W).to(device)
    
    # Instantiate the channel-wise cross-attention module using chunking.
    # For a sequence length L = H*W, choose a chunk_size (e.g. 256 if H*W is large)
    model = ChannelWiseCrossAttention_FrequencyToSpatial_Chunk(in_channels=C, num_heads=8, chunk_size=64).to(device)
    model.eval()
    
    # Reset peak CUDA memory statistics (if applicable)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    # Forward pass within external mixed-precision context.
    start_time = time.time()
    with torch.no_grad():
        # Use torch.cuda.amp.autocast for mixed precision; note the new recommended syntax if desired.
        with torch.cuda.amp.autocast():
            output = model(freq_input, spatial_input)
    end_time = time.time()
    
    # Output shape check.
    print("Final Output Shape:", output.shape)  # Expected: (B, C, H, W)
    
    # Report peak GPU memory usage.
    if device.type == 'cuda':
        peak_memory_MB = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU Memory Allocated: {peak_memory_MB:.2f} MB")
    
    # Report forward pass time in milliseconds.
    print("Forward pass time: {:.2f} ms".format((end_time - start_time) * 1000))
