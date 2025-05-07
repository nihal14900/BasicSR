import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftedWindowCrossAttention(nn.Module):
    r"""
    Shifted Window Cross-Attention Module (Spatial → Frequency).

    This module implements cross-attention where the query (Q) is obtained
    from the spatial domain (after applying a cyclic shift) and the keys/values
    (K, V) are obtained from the frequency domain (features from a global DCT,
    hence not spatially structured). Window partitioning is then performed on
    the queries and keys/values to compute local attention. After attention
    computation, window outputs are merged and the original shift is reversed.

    Args:
        embed_dim (int): Dimensionality of input feature channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size (M) for window partitioning.
        shift_size (int, optional): Amount of cyclic shift (default: M // 2).
        dropout (float): Dropout rate applied on attention weights.
        debug (bool): If True, prints shapes and memory consumption info.
        
    Inputs:
        X_s: Spatial features from the spatial path, shape (B, C, H, W).
        X_f: Frequency features from the frequency path (global DCT),
             shape (B, C, H, W).
             
    Returns:
        torch.Tensor: Output tensor of shape (B, C, H, W).
    """
    def __init__(self, embed_dim, num_heads=8, window_size=8, shift_size=None, dropout=0.0, debug=False):
        super(ShiftedWindowCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        # Use cyclic shift only on spatial features.
        self.shift_size = shift_size if shift_size is not None else window_size // 2
        self.dropout = dropout
        self.debug = debug

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for query (spatial) and key/value (frequency)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, X_s, X_f):
        """
        Args:
            X_s (torch.Tensor): Spatial domain features (B, C, H, W).
            X_f (torch.Tensor): Frequency domain features (B, C, H, W).
        
        Returns:
            torch.Tensor: Output tensor (B, C, H, W).
        """
        B, C, H, W = X_s.shape
        
        # -------------------------------
        # 0. Center Padding if needed
        # -------------------------------
        # Compute total padding required in height and width to reach multiple of window_size.
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        # Compute symmetric (center) padding amounts.
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        if pad_h or pad_w:
            X_s = F.pad(X_s, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            X_f = F.pad(X_f, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        # Update spatial dimensions to the padded sizes.
        _, _, H_pad, W_pad = X_s.shape

        # ===========================
        # 1. Apply cyclic shift to X_s (spatial domain)
        # ===========================
        shifted_X_s = self.cyclic_shift(X_s, shift=(self.shift_size, self.shift_size))
        if self.debug:
            print("After cyclic shift, X_s shape:", shifted_X_s.shape)

        # ===========================
        # 2. Linear Projections
        # We project using learned matrices.
        # For proper use of nn.Linear, we first transpose to (B, H, W, C)
        # then reshape to (B, H*W, C).
        # ===========================
        q = self.q_proj(shifted_X_s.permute(0, 2, 3, 1)).reshape(B, H_pad * W_pad, C)  # (B, HW, C)
        k = self.k_proj(X_f.permute(0, 2, 3, 1)).reshape(B, H_pad * W_pad, C)            # (B, HW, C)
        v = self.v_proj(X_f.permute(0, 2, 3, 1)).reshape(B, H_pad * W_pad, C)            # (B, HW, C)
        if self.debug:
            print("Projected shapes: q:", q.shape, "k:", k.shape, "v:", v.shape)

        # ===========================
        # 3. Reshape for Multi-Head Attention
        # Rearranging to (B, num_heads, HW, head_dim)
        # ===========================
        q = q.view(B, H_pad * W_pad, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, H_pad * W_pad, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, H_pad * W_pad, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # ===========================
        # 4. Partition into Windows
        # Partition the queries (from shifted spatial domain) as well as keys/values into local windows.
        # ===========================
        q_windows, num_windows = self.window_partition(q, H_pad, W_pad, self.window_size)
        k_windows, _ = self.window_partition(k, H_pad, W_pad, self.window_size)
        v_windows, _ = self.window_partition(v, H_pad, W_pad, self.window_size)
        if self.debug:
            print("After window partition:")
            print("  q_windows:", q_windows.shape, "k_windows:", k_windows.shape, "v_windows:", v_windows.shape)

        # ===========================
        # 5. Compute Scaled Dot-Product Attention (per window)
        # No relative position bias is added because the frequency domain (global DCT) lacks spatial structure.
        # ===========================
        attn = (q_windows @ k_windows.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out_windows = attn @ v_windows  # (B*num_windows, num_heads, window_size*window_size, head_dim)
        if self.debug:
            print("Attention window output shape:", out_windows.shape)

        # ===========================
        # 6. Merge Windows and Reverse Shift
        # Merge windowed outputs back to the padded spatial resolution.
        # Then reverse the cyclic shift applied to the spatial path.
        # ===========================
        out = self.window_reverse(out_windows, H_pad, W_pad, self.window_size)  # shape: (B, num_heads, H_pad*W_pad, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, H_pad * W_pad, C)  # (B, HW, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        out = out.view(B, H_pad, W_pad, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H_pad, W_pad)
        out = self.reverse_cyclic_shift(out, shift=(self.shift_size, self.shift_size))
        
        # ===========================
        # 7. Remove Padding if Added
        # Crop the output back to the original spatial size (H, W).
        # ===========================
        if pad_h or pad_w:
            out = out[:, :, pad_top: pad_top + H, pad_left: pad_left + W]
        if self.debug:
            print("Final output shape:", out.shape)
            if torch.cuda.is_available():
                print("CUDA memory allocated (Mbytes):", torch.cuda.memory_allocated() / 1024 / 1024)
        return out

    def cyclic_shift(self, x, shift):
        r"""
        Perform a cyclic (roll) shift on the input tensor along height and width.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            shift (tuple): Tuple (shift_h, shift_w) for vertical and horizontal shift.
        
        Returns:
            torch.Tensor: Shifted tensor.
        """
        shift_h, shift_w = shift
        return torch.roll(x, shifts=(shift_h, shift_w), dims=(2, 3))

    def reverse_cyclic_shift(self, x, shift):
        r"""
        Reverse the cyclic shift by shifting in the opposite direction.

        Args:
            x (torch.Tensor): Tensor of shape (B, C, H, W).
            shift (tuple): The original shift (shift_h, shift_w).
        
        Returns:
            torch.Tensor: Tensor with the cyclic shift reversed.
        """
        shift_h, shift_w = shift
        return torch.roll(x, shifts=(-shift_h, -shift_w), dims=(2, 3))

    def window_partition(self, x, H, W, window_size):
        r"""
        Partition the input tensor into non-overlapping windows.

        Args:
            x (torch.Tensor): Input tensor of shape (B, num_heads, H*W, head_dim).
            H (int): Height of the (possibly padded) feature map.
            W (int): Width of the (possibly padded) feature map.
            window_size (int): Size of the window.
        
        Returns:
            windows (torch.Tensor): Partitioned windows, shape (B * num_windows, num_heads, window_size*window_size, head_dim).
            num_windows (int): Number of windows per image.
        """
        B, num_heads, L, head_dim = x.shape
        assert L == H * W, "Input spatial dimension mismatches H * W"
        x = x.view(B, num_heads, H, W, head_dim)
        num_windows_H = H // window_size
        num_windows_W = W // window_size
        windows = x.view(B, num_heads, num_windows_H, window_size, num_windows_W, window_size, head_dim)
        windows = windows.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        windows = windows.view(B * num_windows_H * num_windows_W, num_heads, window_size * window_size, head_dim)
        return windows, num_windows_H * num_windows_W

    def window_reverse(self, windows, H, W, window_size):
        r"""
        Reverse the window partition to reconstruct the feature map.

        Args:
            windows (torch.Tensor): Windows tensor of shape (B*num_windows, num_heads, window_size*window_size, head_dim).
            H (int): (Padded) height of the feature map.
            W (int): (Padded) width of the feature map.
            window_size (int): Size of the window.
        
        Returns:
            x (torch.Tensor): Reconstructed tensor of shape (B, num_heads, H*W, head_dim).
        """
        num_windows = windows.shape[0]
        num_windows_H = H // window_size
        num_windows_W = W // window_size
        B = num_windows // (num_windows_H * num_windows_W)
        num_heads, head_dim = windows.shape[1], windows.shape[-1]
        x = windows.view(B, num_windows_H, num_windows_W, num_heads, window_size, window_size, head_dim)
        x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous()
        x = x.view(B, num_heads, H * W, head_dim)
        return x

# ===========================
# Test and Debug Block
# ===========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, C, H, W = 8, 32, 49, 65  # Example dimensions that are NOT multiples of the window size
    window_size = 8
    num_heads = 1

    # Create dummy inputs.
    # X_s: Spatial features (e.g., from a CNN or previous layer)
    # X_f: Frequency features (global DCT outputs; note these are not spatially structured)
    X_s = torch.randn(B, C, H, W, device=device)
    X_f = torch.randn(B, C, H, W, device=device)

    # Instantiate the Shifted Window Cross-Attention module with debug flag enabled.
    model = ShiftedWindowCrossAttention(embed_dim=C, num_heads=num_heads,
                                          window_size=window_size, shift_size=window_size // 2, debug=True).to(device)

    # Mixed precision: enable autocast if supported and on CUDA.
    use_amp = device == "cuda"
    with torch.cuda.amp.autocast(enabled=use_amp):
        output = model(X_s, X_f)
        print("Final output shape:", output.shape)

    # Report memory consumption if on GPU.
    if device == "cuda":
        print("CUDA memory allocated (Mbytes):", torch.cuda.memory_allocated() / 1024 / 1024)
        print("CUDA memory reserved (Mbytes):", torch.cuda.memory_reserved() / 1024 / 1024)
