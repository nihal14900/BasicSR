import torch
from torch import nn as nn
import torch.nn.functional as F

from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.v101_2_basic_block import BasicBlock


@ARCH_REGISTRY.register()
class EDSR(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.upscale_factor = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.body = make_layer(BasicBlock, num_block, input_channels=num_feat, norm_groups=4, num_heads=4, kernel_sizes=[3, 5], dilations=[1, 1], window_size=(8, 8), ffn_expansion=2, use_checkpoint=False, use_mixed_precision=False)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        scale = self.upscale_factor
        pad_h = (scale - h % scale) % scale
        pad_w = (scale - w % scale) % scale
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        x = x[:, :, :h * scale, :w * scale]

        return x


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {total:,} ({total / 1e6:.2f}M)")

    try:
        from fvcore.nn import parameter_count_table
        print(parameter_count_table(model))
    except ImportError:
        print("Install fvcore for detailed breakdown: pip install fvcore")

# Example usage
if __name__ == "__main__":
    model = EDSR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=8,  # change back to 16 for full model
        upscale=4,
        res_scale=1,
        img_range=255.,
        rgb_mean=(0.4488, 0.4371, 0.4040)
    )

    input_tensor = torch.randn(1, 3, 64, 64)

    # count_parameters(model)
    # Run FLOP analysis
    flops = FlopCountAnalysis(model, input_tensor)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n🔍 FLOPs & Parameter Summary:")
    print(flop_count_table(flops))
    print(parameter_count_table(model))
    print(f"\nTotal Parameters: {params:,} ({params / 1e6:.2f}M)")
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")

