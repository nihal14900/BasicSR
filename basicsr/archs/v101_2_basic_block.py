import torch
import time
from torch.cuda.amp import autocast

from basicsr.archs.v101_0_utils import Conv3GN, PixelUnshuffle, PixelShuffle
from basicsr.archs.v101_0_wavelet_transform import DWTWrapper, IDWTWrapper
from basicsr.archs.v101_0_multi_range_attention import MultiRangeAttention
from basicsr.archs.v101_0_channel_wise_self_attention import ChannelWiseSelfAttention
from basicsr.archs.v101_0_spatial_window_self_attention import CrossSpatialWindowAttention
from basicsr.archs.v101_0_sparse_multi_range_attention import SparseMultiRangeCrossAttention

class MRSAEncoderBlock(torch.nn.Module):
    def __init__(self, channels, num_heads, kernel_sizes, dilations, ffn_expansion, norm_groups):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations

        self.ffn_expansion = ffn_expansion
        self.norm_groups = norm_groups

        self.attention = MultiRangeAttention(self.channels, self.num_heads, self.kernel_sizes, self.dilations, False, False, True, 0, 0, True)
        self.norm1 = torch.nn.GroupNorm(self.norm_groups, self.channels)
        self.norm2 = torch.nn.GroupNorm(self.norm_groups, self.channels)

        self.ffn = torch.nn.Sequential(
            torch.nn.Conv2d(self.channels, self.ffn_expansion * self.channels, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(self.ffn_expansion * self.channels, self.channels, 1)
        )
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class CWSAEncoderBlock(torch.nn.Module):
    def __init__(self, channels, num_heads, ffn_expansion, norm_groups):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.ffn_expansion = ffn_expansion
        self.norm_groups = norm_groups

        self.attention = ChannelWiseSelfAttention(self.channels, self.num_heads, True, None, 0, 0, False, False)
        self.norm1 = torch.nn.GroupNorm(self.norm_groups, self.channels)
        self.norm2 = torch.nn.GroupNorm(self.norm_groups, self.channels)

        self.ffn = torch.nn.Sequential(
            torch.nn.Conv2d(self.channels, self.ffn_expansion * self.channels, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(self.ffn_expansion * self.channels, self.channels, 1)
        )
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SWCAEncoderBlock(torch.nn.Module):
    def __init__(self, q_channels, kv_channels, hidden_channels, num_heads, window_size, ffn_expansion, q_norm_groups, kv_norm_groups):
        super().__init__()
        self.q_channels = q_channels
        self.kv_channels = kv_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.window_size = window_size

        self.ffn_expansion = ffn_expansion
        self.q_norm_groups = q_norm_groups
        self.kv_norm_groups = kv_norm_groups

        self.attention = CrossSpatialWindowAttention(self.q_channels, self.kv_channels, self.hidden_channels, self.num_heads, self.window_size, True, None, 0, False, False)
        self.q_norm1 = torch.nn.GroupNorm(self.q_norm_groups, self.q_channels)
        self.kv_norm1 = torch.nn.GroupNorm(self.kv_norm_groups, self.kv_channels)
        self.norm2 = torch.nn.GroupNorm(self.q_norm_groups, self.q_channels)

        self.ffn = torch.nn.Sequential(
            torch.nn.Conv2d(self.q_channels, self.ffn_expansion * self.q_channels, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(self.ffn_expansion * self.q_channels, self.q_channels, 1)
        )
    def forward(self, x_q, x_kv):
        q, kv = self.q_norm1(x_q), self.kv_norm1(x_kv)
        x = x_q + self.attention(q, kv)
        x = x + self.ffn(self.norm2(x))
        return x

class SMRCAEncoderBlock(torch.nn.Module):
    def __init__(self, q_channels, kv_channels, hidden_channels, num_heads, kernel_sizes, dilations, ffn_expansion, q_norm_groups, kv_norm_groups):
        super().__init__()
        self.q_channels = q_channels
        self.kv_channels = kv_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations

        self.ffn_expansion = ffn_expansion
        self.q_norm_groups = q_norm_groups
        self.kv_norm_groups = kv_norm_groups

        self.attention = SparseMultiRangeCrossAttention(self.q_channels, self.kv_channels, self.num_heads, self.kernel_sizes, [4 * dilation for dilation in self.dilations], self.hidden_channels, True, None, True, 0, 0, False, False)
        self.q_norm1 = torch.nn.GroupNorm(self.q_norm_groups, self.q_channels)
        self.kv_norm1 = torch.nn.GroupNorm(self.kv_norm_groups, self.kv_channels)
        self.norm2 = torch.nn.GroupNorm(self.q_norm_groups, self.q_channels)

        self.ffn = torch.nn.Sequential(
            torch.nn.Conv2d(self.q_channels, self.ffn_expansion * self.q_channels, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(self.ffn_expansion * self.q_channels, self.q_channels, 1)
        )
    def forward(self, x_q, x_kv):
        q, kv = self.q_norm1(x_q), self.kv_norm1(x_kv)
        x = x_q + self.attention(q, kv)
        x = x + self.ffn(self.norm2(x))
        return x



class BasicBlock(torch.nn.Module):
    def __init__(self, input_channels, norm_groups, num_heads, kernel_sizes, dilations, window_size, ffn_expansion, use_checkpoint, use_mixed_precision):
        super().__init__()

        self.channels = input_channels
        self.groups = norm_groups
        self.num_heads = num_heads
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.window_size = window_size
        self.ffn_expansion = ffn_expansion
        self.use_checkpoint = use_checkpoint
        self.use_mixed_precision = use_mixed_precision

        # Wavelet Transforms
        self.dwt = DWTWrapper('haar', self.use_checkpoint, self.use_mixed_precision)
        self.idwt = IDWTWrapper('haar', self.use_checkpoint, self.use_mixed_precision)

        # Rearrange
        self.unshuffle = PixelUnshuffle(2)
        self.shuffle = PixelShuffle(2)

        # Self Attentions in Path 1
        self.path1_sa1  = MRSAEncoderBlock(self.channels, self.num_heads, self.kernel_sizes, self.dilations, self.ffn_expansion, self.channels // 8)
        self.path1_sa21 = CWSAEncoderBlock(self.channels, self.num_heads, self.ffn_expansion, self.channels // 8)
        self.path1_sa22 = CWSAEncoderBlock(self.channels, self.num_heads, self.ffn_expansion, self.channels // 8)
        self.path1_sa23 = CWSAEncoderBlock(self.channels, self.num_heads, self.ffn_expansion, self.channels // 8)
        self.path1_sa24 = CWSAEncoderBlock(self.channels, self.num_heads, self.ffn_expansion, self.channels // 8)

        # Self Attentions in Path 2
        self.path2_sa11 = CWSAEncoderBlock(self.channels, self.num_heads, self.ffn_expansion, self.channels // 8)
        self.path2_sa12 = CWSAEncoderBlock(self.channels, self.num_heads, self.ffn_expansion, self.channels // 8)
        self.path2_sa13 = CWSAEncoderBlock(self.channels, self.num_heads, self.ffn_expansion, self.channels // 8)
        self.path2_sa14 = CWSAEncoderBlock(self.channels, self.num_heads, self.ffn_expansion, self.channels // 8)
        self.path2_sa2  = MRSAEncoderBlock(self.channels, self.num_heads, self.kernel_sizes, self.dilations, self.ffn_expansion, self.channels // 8)

        # Cross Attentions in Path 1
        self.path1_ca1  = SWCAEncoderBlock(self.channels, self.channels, self.channels, self.num_heads, self.window_size, self.ffn_expansion, self.channels // 8, self.channels // 8)
        self.path1_ca21 = SMRCAEncoderBlock(self.channels, self.channels, self.channels, self.num_heads, self.kernel_sizes, self.dilations, self.ffn_expansion, self.channels // 8, self.channels // 8)
        self.path1_ca22 = SMRCAEncoderBlock(self.channels, self.channels, self.channels, self.num_heads, self.kernel_sizes, self.dilations, self.ffn_expansion, self.channels // 8, self.channels // 8)
        self.path1_ca23 = SMRCAEncoderBlock(self.channels, self.channels, self.channels, self.num_heads, self.kernel_sizes, self.dilations, self.ffn_expansion, self.channels // 8, self.channels // 8)
        self.path1_ca24 = SMRCAEncoderBlock(self.channels, self.channels, self.channels, self.num_heads, self.kernel_sizes, self.dilations, self.ffn_expansion, self.channels // 8, self.channels // 8)

        # Cross Attentions in Path 2
        self.path2_ca11 = SWCAEncoderBlock(self.channels, self.channels, self.channels, self.num_heads, self.window_size, self.ffn_expansion, self.channels // 8, self.channels // 8)
        self.path2_ca12 = SWCAEncoderBlock(self.channels, self.channels, self.channels, self.num_heads, self.window_size, self.ffn_expansion, self.channels // 8, self.channels // 8)
        self.path2_ca13 = SWCAEncoderBlock(self.channels, self.channels, self.channels, self.num_heads, self.window_size, self.ffn_expansion, self.channels // 8, self.channels // 8)
        self.path2_ca14 = SWCAEncoderBlock(self.channels, self.channels, self.channels, self.num_heads, self.window_size, self.ffn_expansion, self.channels // 8, self.channels // 8)
        self.path2_ca2  = SMRCAEncoderBlock(self.channels, self.channels, self.channels, self.num_heads, self.kernel_sizes, self.dilations, self.ffn_expansion, self.channels // 8, self.channels // 8)

        # Convolutions in Path 1
        self.path1_conv1  = Conv3GN(self.channels, self.groups)
        self.path1_conv2  = Conv3GN(self.channels, self.groups)

        self.path1_conv31 = Conv3GN(self.channels, self.groups)
        self.path1_conv32 = Conv3GN(self.channels, self.groups)
        self.path1_conv33 = Conv3GN(self.channels, self.groups)
        self.path1_conv34 = Conv3GN(self.channels, self.groups)

        self.path1_conv41 = Conv3GN(self.channels, self.groups)
        self.path1_conv42 = Conv3GN(self.channels, self.groups)
        self.path1_conv43 = Conv3GN(self.channels, self.groups)
        self.path1_conv44 = Conv3GN(self.channels, self.groups)

        # Convolutions in Path 2
        self.path2_conv11 = Conv3GN(self.channels, self.groups)
        self.path2_conv12 = Conv3GN(self.channels, self.groups)
        self.path2_conv13 = Conv3GN(self.channels, self.groups)
        self.path2_conv14 = Conv3GN(self.channels, self.groups)

        self.path2_conv21 = Conv3GN(self.channels, self.groups)
        self.path2_conv22 = Conv3GN(self.channels, self.groups)
        self.path2_conv23 = Conv3GN(self.channels, self.groups)
        self.path2_conv24 = Conv3GN(self.channels, self.groups)

        self.path2_conv3  = Conv3GN(self.channels, self.groups)
        self.path2_conv4  = Conv3GN(self.channels, self.groups)


        # Feature Fusion
        self.path1_ca_fusion = torch.nn.Conv2d(4 * 4 * self.channels, 4 * self.channels, 1)
        self.path2_ca_fusion = torch.nn.Conv2d(4 * 4 * self.channels, 4 * self.channels, 1)
        self.fusion = torch.nn.Conv2d(2 * self.channels, self.channels, 1)


        self.gate = torch.nn.Sequential(
            torch.nn.Conv2d(2 * self.channels, self.channels, 1),
            torch.nn.Sigmoid()
        )
        self.proj_x = torch.nn.Conv2d(self.channels, self.channels, 1)

    def forward(self, x):

        # Feature Transform
        y = x
        z1, z2, z3, z4 = self.dwt(x.float())

        # print("feature transform: ", x.shape, y.shape, z1.shape)

        # Convolution 1
        y = self.path1_conv1(x)
        z1, z2, z3, z4 = self.path2_conv11(z1), self.path2_conv12(z2), self.path2_conv13(z3), self.path2_conv14(z4)
        # Self Attention 1
        y = self.path1_sa1(y)
        z1, z2, z3, z4 = self.path2_sa11(z1), self.path2_sa12(z2), self.path2_sa13(z3), self.path2_sa14(z4)

        # print("stage 1: ", x.shape, y.shape, z1.shape)

        # Convolution 2
        y = self.path1_conv2(x)
        z1, z2, z3, z4 = self.path2_conv21(z1), self.path2_conv22(z2), self.path2_conv23(z3), self.path2_conv24(z4)
        # Cross Attention 2
        y1, y2, y3, y4 = self.dwt(y.float())
        z = self.idwt(z1.float(), z2.float(), z3.float(), z4.float())

        y = self.path1_ca1(y, z)
        z1, z2, z3, z4 = self.path2_ca11(z1, y1), self.path2_ca12(z2, y2), self.path2_ca13(z3, y3), self.path2_ca14(z4, y4)

        # print("stage 2: ", x.shape, y.shape, z1.shape)

        # Domain Swap
        y1, y2, y3, y4 = self.dwt(y.float())
        z = self.idwt(z1.float(), z2.float(), z3.float(), z4.float())

        # print("domain swap: ", x.shape, y1.shape, z.shape)

        # Convolution 3
        y1, y2, y3, y4 = self.path1_conv31(y1), self.path1_conv32(y2), self.path1_conv33(y3), self.path1_conv34(y4)
        z = self.path2_conv3(z)
        # Self Attention 2
        y1, y2, y3, y4 = self.path1_sa21(y1), self.path1_sa22(y2), self.path1_sa23(y3), self.path1_sa24(y4)
        z = self.path2_sa2(z)

        # print("stage 3: ", x.shape, y1.shape, z.shape)

        # Convolution 4
        y1, y2, y3, y4 = self.path1_conv41(y1), self.path1_conv42(y2), self.path1_conv43(y3), self.path1_conv44(y4)
        z = self.path2_conv4(z)
        # Cross Attention 2
        y = self.idwt(y1.float(), y2.float(), y3.float(), y4.float())
        z1, z2, z3, z4 = self.dwt(z.float())

        y1, y2, y3, y4 = self.path1_ca21(y1, z1), self.path1_ca22(y2, z2), self.path1_ca23(y3, z3), self.path1_ca24(y4, z4)
        z = self.path2_ca2(z, y)

        # print("stage 4: ", x.shape, y1.shape, z.shape)

        # Feature Transform
        y = self.idwt(y1.float(), y2.float(), y3.float(), y4.float())
        z = z

        # print("pre-fusion: ", x.shape, y.shape, z.shape)

        # Feature Fusion
        yz = self.fusion(torch.cat([y, z], 1))
        # concat = torch.cat([y, z], 1)
        # gate = self.gate(concat)
        # gated = gate * y + (1 - gate) * z
        # fused = self.fusion(concat)
        # yz = gated + fused
        # x = self.proj_x(x)

        # print("post-fusion: ", x.shape, yz.shape)

        # Residual
        return x + yz

def test_BasicBlock(mixed_precision=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 8, 64, 64, 64
    input = torch.randn(B, C, H, W).to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    block = BasicBlock(C, C // 8, 8, [5, 7], [1, 1], (8, 8), 2, False, False).to(device)
    block.eval()

    if not mixed_precision:
        input = input.float()
        block = block.float()

    # Timing
    start = time.time()

    with autocast(enabled=mixed_precision):
        output = block(input)

    torch.cuda.synchronize()  # Ensure accurate timing
    elapsed = time.time() - start

    print(f"\n{'Mixed' if mixed_precision else 'Full'} Precision")
    print(f"Output shape: {output.shape}")
    print(f"Inference time: {elapsed:.4f} seconds")

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Memory allocated: {allocated:.2f} MB")
        print(f"Memory reserved : {reserved:.2f} MB")
        print(f"Peak allocated  : {peak:.2f} MB")



if __name__ == "__main__":
    # Run both tests
    test_BasicBlock(mixed_precision=False)   # Full precision
    test_BasicBlock(mixed_precision=True)    # Mixed precision