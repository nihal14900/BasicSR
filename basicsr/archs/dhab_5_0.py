import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct  # pip install torch-dct

# Import provided attention modules.
from cwcas_1_2 import ChannelWiseCrossAttention_SpatialToFrequency_Chunk
from cwcsf_1_2 import ChannelWiseCrossAttention_FrequencyToSpatial_Chunk
from dmrsa_1_1 import DenseMultiRangeAttention
from fn_2_0 import FreqGroupNorm
# from smrsa_1_0 import SparseMultiRangeAttention
from smrsa_2_1 import SparseMultiRangeAttention2D as SparseMultiRangeAttention
from swcaf_2_0 import FrequencyToSpatialCrossAttention
from swcas_2_0 import ShiftedWindowCrossAttention

# ---------------------------
# Simple FeedForward Block (FFN)
# ---------------------------
class FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion=4):
        super(FeedForward, self).__init__()
        hidden_dim = embed_dim * expansion
        self.fc1 = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return out

# ---------------------------
# Dual-Helix Attention Block (D-HAB) with merged projection & fusion.
# ---------------------------
class DualHelixAttentionBlock(nn.Module):
    r"""
    Dual-Helix Attention Block (D-HAB) that aggregates spatial and frequency features,
    with a merged input projection and fusion pathway.

    The input X ∈ ℝ^(B, embed_dim, H, W) is first expanded to 2*embed_dim
    via a learned 1×1 convolution (proj_conv), then split equally into a spatial branch
    and a frequency branch (each with embed_dim channels). After processing, the branches are
    concatenated and added to the expanded skip connection, then compressed back to embed_dim
    via out_conv and normalized.
    """
    def __init__(self, embed_dim, num_heads=4, expansion=4,
                 spatial_dense_ranges=[7, 9, 11],
                 spatial_sparse_ranges=[3, 5],
                 freq_dense_ranges=[7, 9, 11],
                 freq_sparse_ranges=[3, 5],
                 freq_sparse_dilations=[2, 3],
                 spatial_chunk_size=64,
                 window_size=8,
                 dropout=0.0):
        super(DualHelixAttentionBlock, self).__init__()
        # Merged projection: from embed_dim to 2*embed_dim.
        # We then split into two branches, each with embed_dim channels.
        self.branch_dim = embed_dim

        self.proj_conv = nn.Conv2d(embed_dim, 2 * embed_dim, kernel_size=1)

        # ---------------------------
        # Spatial Branch Modules (operating on branch_dim channels)
        # ---------------------------
        self.spatial_initial_conv = nn.Conv2d(self.branch_dim, self.branch_dim, kernel_size=3, padding=1)
        self.spatial_inter_conv = nn.Conv2d(self.branch_dim, self.branch_dim, kernel_size=1)
        self.act = nn.GELU()

        self.spatial_norm1 = nn.GroupNorm(num_groups=1, num_channels=self.branch_dim)
        self.spatial_norm2 = nn.GroupNorm(num_groups=1, num_channels=self.branch_dim)
        self.spatial_norm3 = nn.GroupNorm(num_groups=1, num_channels=self.branch_dim)
        self.spatial_norm4 = nn.GroupNorm(num_groups=1, num_channels=self.branch_dim)

        self.spatial_dense_attn = DenseMultiRangeAttention(embed_dim=self.branch_dim,
                                                            num_heads=num_heads,
                                                            range_sizes=spatial_dense_ranges)
        self.spatial_ffn1 = FeedForward(self.branch_dim, expansion=expansion)
        self.spatial_cross_attn_stage1 = ShiftedWindowCrossAttention(embed_dim=self.branch_dim,
                                                                      num_heads=num_heads,
                                                                      window_size=window_size,
                                                                      shift_size=window_size // 2,
                                                                      dropout=dropout,
                                                                      debug=False)
        self.spatial_ffn2 = FeedForward(self.branch_dim, expansion=expansion)
        # self.spatial_sparse_attn = SparseMultiRangeAttention(embed_dim=self.branch_dim,
        #                                                       num_heads=num_heads,
        #                                                       range_sizes=spatial_sparse_ranges,
        #                                                       dilation_factors=freq_sparse_dilations)
        self.spatial_sparse_attn = SparseMultiRangeAttention(self.branch_dim, 
                                                             4, 
                                                             [3, 5, 7, 9], 
                                                             [1, 2, 3, 4], 
                                                             use_checkpoint=True)
        self.spatial_ffn3 = FeedForward(self.branch_dim, expansion=expansion)
        self.spatial_cross_attn_stage2 = ChannelWiseCrossAttention_SpatialToFrequency_Chunk(in_channels=self.branch_dim,
                                                                                             num_heads=1,
                                                                                             chunk_size=spatial_chunk_size)
        self.spatial_ffn4 = FeedForward(self.branch_dim, expansion=expansion)

        # ---------------------------
        # Frequency Branch Modules (operating on branch_dim channels)
        # ---------------------------
        self.freq_initial_conv = nn.Conv2d(self.branch_dim, self.branch_dim, kernel_size=1)
        self.freq_inter_conv = nn.Conv2d(self.branch_dim, self.branch_dim, kernel_size=1)

        self.freq_act = nn.GELU()
        self.freq_norm1 = FreqGroupNorm(num_channels=self.branch_dim, num_groups=1)
        self.freq_norm2 = FreqGroupNorm(num_channels=self.branch_dim, num_groups=1)
        self.freq_norm3 = FreqGroupNorm(num_channels=self.branch_dim, num_groups=1)
        self.freq_norm4 = FreqGroupNorm(num_channels=self.branch_dim, num_groups=1)
        self.freq_dense_attn = DenseMultiRangeAttention(embed_dim=self.branch_dim,
                                                         num_heads=num_heads,
                                                         range_sizes=freq_dense_ranges)
        self.freq_ffn1 = FeedForward(self.branch_dim, expansion=expansion)
        self.freq_cross_attn_stage1 = FrequencyToSpatialCrossAttention(embed_dim=self.branch_dim,
                                                                        num_heads=num_heads,
                                                                        window_size=window_size,
                                                                        shift_size=window_size // 2,
                                                                        dropout=dropout,
                                                                        debug=False)
        self.freq_ffn2 = FeedForward(self.branch_dim, expansion=expansion)
        # self.freq_sparse_attn = SparseMultiRangeAttention(embed_dim=self.branch_dim,
        #                                                    num_heads=num_heads,
        #                                                    range_sizes=freq_sparse_ranges,
        #                                                    dilation_factors=freq_sparse_dilations)
        self.freq_sparse_attn = SparseMultiRangeAttention(self.branch_dim, 
                                                             4, 
                                                             [3, 5, 7, 9], 
                                                             [1, 2, 3, 4], 
                                                             use_checkpoint=True)
        self.freq_ffn3 = FeedForward(self.branch_dim, expansion=expansion)
        self.freq_cross_attn_stage2 = ChannelWiseCrossAttention_FrequencyToSpatial_Chunk(in_channels=self.branch_dim,
                                                                                           num_heads=num_heads,
                                                                                           chunk_size=spatial_chunk_size)
        self.freq_ffn4 = FeedForward(self.branch_dim, expansion=expansion)

        # ---------------------------
        # Final merged output: compress from 2*branch_dim back to embed_dim.
        # ---------------------------
        self.out_conv = nn.Conv2d(2 * self.branch_dim, embed_dim, kernel_size=1)
        self.out_norm = nn.GroupNorm(num_groups=1, num_channels=embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # Expand: project input from embed_dim -> 2*embed_dim and split into two branches.
        x_expanded = self.proj_conv(x)  # Shape: (B, 2*embed_dim, H, W)
        x_s, x_f = torch.chunk(x_expanded, 2, dim=1)  # Each branch: (B, embed_dim, H, W)

        # ---------------------------
        # Spatial branch processing.
        # ---------------------------
        s0 = self.act(self.spatial_initial_conv(x_s))

        # ---------------------------
        # Frequency branch processing.
        # ---------------------------
        orig_dtype = x_f.dtype
        x_f = x_f.float()
        x_f = dct.dct_2d(x_f, norm="ortho")
        x_f = x_f.to(orig_dtype)
        f0 = self.act(self.freq_initial_conv(x_f))

        # ---------------------------
        # Stage 1: Self-attention and FFN on both branches.
        # ---------------------------
        s1 = self.spatial_norm1(s0)
        s1 = s1 + self.spatial_dense_attn(s1)
        s1 = s1 + self.spatial_ffn1(s1)

        f1 = self.freq_norm1(f0)
        f1 = f1 + self.freq_dense_attn(f1)
        f1 = f1 + self.freq_ffn1(f1)

        # ---------------------------
        # Stage 2: Cross-attention and FFN.
        # ---------------------------
        s2 = self.spatial_norm2(s1)
        s2 = s2 + self.spatial_cross_attn_stage1(s2, f1)
        s2 = s2 + self.spatial_ffn2(s2)

        f2 = self.freq_norm2(f1)
        f2 = f2 + self.freq_cross_attn_stage1(f2, s1)
        f2 = f2 + self.freq_ffn2(f2)

        # ---------------------------
        # Intermediate processing.
        # ---------------------------
        s3 = self.act(self.spatial_inter_conv(s2))
        f3 = self.act(self.freq_inter_conv(f2))

        # ---------------------------
        # Stage 3: Additional self-attention and FFN.
        # ---------------------------
        s4 = self.spatial_norm3(s3)
        s4 = s4 + self.spatial_sparse_attn(s4)
        s4 = s4 + self.spatial_ffn3(s4)

        f4 = self.freq_norm3(f3)
        f4 = f4 + self.freq_sparse_attn(f4)
        f4 = f4 + self.freq_ffn3(f4)

        # ---------------------------
        # Stage 4: Final cross-attention and FFN.
        # ---------------------------
        s5 = self.spatial_norm4(s4)
        s5 = s5 + self.spatial_cross_attn_stage2(s5, f4)
        s5 = s5 + self.spatial_ffn4(s5)

        f5 = self.freq_norm4(f4)
        f5 = f5 + self.freq_cross_attn_stage2(f5, s4)
        f5 = f5 + self.freq_ffn4(f5)

        # Inverse DCT to return frequency branch to spatial domain.
        orig_dtype = f5.dtype
        f5 = f5.float()
        f5 = dct.idct_2d(f5, norm="ortho")
        f5 = f5.to(orig_dtype)

        # ---------------------------
        # Merge branches: concatenate and apply the residual connection.
        # ---------------------------
        fused = torch.cat([s5, f5], dim=1)  # Shape: (B, 2*embed_dim, H, W)
        res = x_expanded + fused  # Residual connection with the expanded input.

        # Final compression back to the original embed_dim.
        out = self.out_conv(res)
        out = self.out_norm(out)
        return out

# ---------------------------
# Testing: Main function (shape, mixed precision, memory consumption)
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)
    
    # Define dummy input: Batch x embed_dim x H x W (embed_dim is the original channel size)
    B, embed_dim, H, W = 8, 64, 64, 64
    dummy_input = torch.randn(B, embed_dim, H, W, device=device)
    
    model = DualHelixAttentionBlock(embed_dim=embed_dim,
                                    num_heads=2,
                                    expansion=2,
                                    spatial_dense_ranges=[3, 5],
                                    spatial_sparse_ranges=[5, 7],
                                    freq_dense_ranges=[3, 5],
                                    freq_sparse_ranges=[5, 7],
                                    freq_sparse_dilations=[2, 3],
                                    spatial_chunk_size=64,
                                    window_size=8,
                                    dropout=0.0).to(device)
    model.eval()
    
    # Reset CUDA peak memory stats if on GPU.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    start_time = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            output = model(dummy_input)
    end_time = time.time()
    
    # Check output properties.
    print("Output shape:", output.shape)  # Expected: (B, embed_dim, H, W)
    print("Output dtype:", output.dtype)
    
    if device.type == "cuda":
        peak_mem_MB = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print("Peak GPU Memory Allocated: {:.2f} MB".format(peak_mem_MB))
    
    print("Forward pass time: {:.2f} ms".format((end_time - start_time) * 1000))
