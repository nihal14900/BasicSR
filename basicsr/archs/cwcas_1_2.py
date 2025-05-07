import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_dct as dct

"""
The torch-dct library implements DCT in terms of built-in FFT operations so that back propagation works
through it on both CPU and GPU.

Usage example (from the documentation):
    import torch_dct as dct
    x = torch.randn(200)
    X = dct.dct(x, norm='ortho')   # Computes 1-D DCT-II
    y = dct.idct(X, norm='ortho')  # Computes the scaled DCT-III
For multidimensional transforms, simply replace dct and idct by dct_2d, idct_2d, etc.
"""

# ---------------------------
# iDCT using torch-dct (2-D version)
# ---------------------------
def idct_2d(x):
    """
    Applies the 2D inverse DCT (DCT-III) using torch-dct on the last two dimensions.
    
    Args:
        x (Tensor): Input tensor of shape (B, C, H, W) in the DCT domain.
    
    Returns:
        Tensor: Output tensor of shape (B, C, H, W) in the spatial domain.
    """
    return dct.idct_2d(x, norm='ortho')

# ---------------------------
# Helper: Chunked Attention
# ---------------------------
def chunked_attention(Q, K, V, chunk_size):
    """
    Computes scaled dot-product attention in chunks along the query dimension,
    thus avoiding the memory cost of creating the full (HW x HW) attention matrix.
    
    Args:
        Q (Tensor): Query tensor of shape (B, num_heads, L, d)
        K (Tensor): Key tensor of shape (B, num_heads, L, d)
        V (Tensor): Value tensor of shape (B, num_heads, L, d)
        chunk_size (int): The number of tokens to process in each chunk.
        
    Returns:
        Tensor: Attention output of shape (B, num_heads, L, d)
    """
    B, h, L, d = Q.shape
    outputs = []
    scale = 1.0 / np.sqrt(d)
    for i in range(0, L, chunk_size):
        Q_chunk = Q[:, :, i:i+chunk_size, :]               # Shape: (B, h, chunk_size, d)
        # Compute attention scores: (B, h, chunk_size, L)
        attn_scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        output_chunk = torch.matmul(attn_weights, V)         # (B, h, chunk_size, d)
        outputs.append(output_chunk)
    return torch.cat(outputs, dim=2)  # Concatenate along token dimension

# ---------------------------
# Channel-Wise Cross-Attention (Spatial -> Frequency) with Chunking
# ---------------------------
class ChannelWiseCrossAttention_SpatialToFrequency_Chunk(nn.Module):
    def __init__(self, in_channels, num_heads, chunk_size=256):
        """
        Initializes the Channel-Wise Cross Attention block.
        In this variant, the query is computed from the spatial features while
        the frequency features (provided in the DCT domain) are first converted to
        the spatial domain via iDCT to obtain key and value. The scaled dot-product
        attention is computed in chunks to save memory.
        
        Args:
            in_channels (int): Number of channels in the input features.
            num_heads (int): Number of attention heads.
            chunk_size (int): Number of tokens to process per attention chunk.
        """
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.chunk_size = chunk_size
        
        # 1x1 convolutions as linear projections.
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, spatial_feat, freq_feat):
        """
        Forward pass for Channel-Wise Cross-Attention (Spatial -> Frequency) using chunked attention.
        
        Args:
            spatial_feat (Tensor): Spatial features, shape (B, C, H, W).
            freq_feat (Tensor): Frequency features (in DCT domain), shape (B, C, H, W).
            
        Returns:
            Tensor: Output feature map of shape (B, C, H, W).
        """
        B, C, H, W = spatial_feat.shape
        assert freq_feat.shape == (B, C, H, W), "Frequency feature shape must match spatial feature shape"
        
        # Step 1: Convert frequency features to spatial domain using iDCT.
        freq_spatial = idct_2d(freq_feat)   # (B, C, H, W)
        
        # Step 2: Compute the linear projections.
        # Query from spatial features.
        Q = self.q_proj(spatial_feat)       # (B, C, H, W)
        # Key and Value from converted frequency features.
        K = self.k_proj(freq_spatial)       # (B, C, H, W)
        V = self.v_proj(freq_spatial)       # (B, C, H, W)
        
        # Step 3: Reshape for multi-head attention.
        # Reshape from (B, C, H, W) to (B, num_heads, head_dim, H*W)
        Q = Q.view(B, self.num_heads, self.head_dim, H * W)
        K = K.view(B, self.num_heads, self.head_dim, H * W)
        V = V.view(B, self.num_heads, self.head_dim, H * W)
        # Permute to (B, num_heads, H*W, head_dim)
        Q = Q.permute(0, 1, 3, 2).contiguous()
        K = K.permute(0, 1, 3, 2).contiguous()
        V = V.permute(0, 1, 3, 2).contiguous()
        
        # Step 4: Compute scaled dot-product attention in chunks.
        attn_output = chunked_attention(Q, K, V, self.chunk_size)  # (B, num_heads, H*W, head_dim)
        
        # Step 5: Reassemble multi-head outputs.
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous()  # (B, num_heads, head_dim, H*W)
        attn_output = attn_output.view(B, C, H, W)  # (B, C, H, W)
        out = self.out_proj(attn_output)           # (B, C, H, W)
        
        # Debug prints: shapes and memory consumption.
        # print("Spatial Input Shape:", spatial_feat.shape)
        # print("Frequency Input Shape:", freq_feat.shape)
        # print("Converted Frequency (after iDCT) Shape:", freq_spatial.shape)
        # print("Output Shape:", out.shape)
        # if spatial_feat.is_cuda:
        #     mem_MB = torch.cuda.memory_allocated() / (1024 ** 2)
        #     print(f"Peak GPU Memory Allocated: {mem_MB:.2f} MB")
            
        return out

# ---------------------------
# Testing Script with Shape Checking, Memory Consumption, and Mixed Precision
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define dummy inputs (B, C, H, W).
    B, C, H, W = 8, 32, 64, 64  
    spatial_input = torch.randn(B, C, H, W).to(device)
    freq_input = torch.randn(B, C, H, W).to(device)  # Simulated DCT coefficients.
    
    # Choose a chunk size. For H*W = 4096 tokens, chunking by 64 tokens means 64 iterations.
    # You can experiment with different chunk sizes. Here we set it to 64.
    chunk_size = 64
    
    # Instantiate the module with chunking.
    model = ChannelWiseCrossAttention_SpatialToFrequency_Chunk(in_channels=C, num_heads=1, chunk_size=chunk_size).to(device)
    model.eval()
    
    # Reset CUDA peak memory statistics (if on GPU).
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    # Run a forward pass under external mixed-precision context.
    start_time = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model(spatial_input, freq_input)
    end_time = time.time()
    
    # Check final output shape.
    print("Final Output Shape:", output.shape)  # Expected: (B, C, H, W)
    
    # Report peak GPU memory usage.
    if device.type == 'cuda':
        peak_memory_MB = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU Memory Allocated: {peak_memory_MB:.2f} MB")
    
    # Report forward pass time in milliseconds.
    print("Forward pass time: {:.2f} ms".format((end_time - start_time) * 1000))
