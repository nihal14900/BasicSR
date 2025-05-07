import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyToSpatialCrossAttention(nn.Module):
    r"""
    Frequency-to-Spatial Cross-Attention Module with Shifted Windows.
    
    This module implements cross-attention where the query is obtained from the
    frequency domain (global DCT features) and is first summarized via global
    average pooling (since global DCT destroys spatial structure). The keys and
    values come from the spatial domain. The spatial features are shifted (using a
    cyclic shift) and partitioned into non-overlapping windows; the shifted windows 
    on the spatial side are used for local attention computation.
    
    Args:
        embed_dim (int): Channel dimension of the input features.
        num_heads (int): Number of attention heads.
        window_size (int): Size M of non-overlapping windows on the spatial domain.
        shift_size (int, optional): Shift for cyclic shifting on spatial features.
            Default is window_size // 2.
        dropout (float): Dropout rate applied on attention weights and final projection.
        debug (bool): If True, prints shape information and CUDA memory consumption (in MB).
    
    Inputs:
        X_f (torch.Tensor): Frequency features from global DCT, shape (B, C, H, W).
        X_s (torch.Tensor): Spatial features, shape (B, C, H, W).
    
    Returns:
        torch.Tensor: Output tensor of shape (B, C, H, W).
    """
    def __init__(self, embed_dim, num_heads=8, window_size=8, shift_size=None, dropout=0.0, debug=False):
        super(FrequencyToSpatialCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size if shift_size is not None else window_size // 2
        self.dropout = dropout
        self.debug = debug
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # For frequency query: use global average pooling (since global DCT is not spatially structured)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # For spatial key/value projections, applied on shifted spatial features
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, X_f, X_s):
        """
        Args:
            X_f (torch.Tensor): Frequency features (B, C, H, W).
            X_s (torch.Tensor): Spatial features (B, C, H, W).
        
        Returns:
            torch.Tensor: Output tensor (B, C, H, W).
        """
        B, C, H_orig, W_orig = X_f.shape
        # Check that both inputs have the same shape.
        assert X_s.shape == X_f.shape, "X_f and X_s must have the same shape."
        
        # ---------------------------------------------------------------------
        # 0. Apply center (symmetric) padding to X_s if needed so that its spatial
        # dimensions become multiples of window_size.
        # ---------------------------------------------------------------------
        pad_h = (self.window_size - H_orig % self.window_size) % self.window_size
        pad_w = (self.window_size - W_orig % self.window_size) % self.window_size
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        if pad_h or pad_w:
            X_s = F.pad(X_s, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        # New spatial size for X_s after padding
        _, _, H_pad, W_pad = X_s.shape
        
        # ---------------------------------------------------------------------
        # 1. Frequency Query: Global average pooling on the original frequency input.
        # ---------------------------------------------------------------------
        Q_global = self.global_pool(X_f)  # shape: (B, C, 1, 1)
        Q_global = Q_global.view(B, C)      # shape: (B, C)
        Q = self.q_proj(Q_global)           # shape: (B, C)
        # Reshape for multi-head attention: (B, num_heads, 1, head_dim)
        Q = Q.view(B, self.num_heads, 1, self.head_dim)
        
        # ---------------------------------------------------------------------
        # 2. Spatial Key and Value: Apply cyclic shift to the padded spatial features.
        # ---------------------------------------------------------------------
        shifted_X_s = self.cyclic_shift(X_s, shift=(self.shift_size, self.shift_size))
        
        # Project K and V.
        # For nn.Linear, first permute to (B, H_pad, W_pad, C) then reshape to (B, H_pad*W_pad, C)
        K = self.k_proj(shifted_X_s.permute(0, 2, 3, 1)).reshape(B, H_pad * W_pad, C)
        V = self.v_proj(shifted_X_s.permute(0, 2, 3, 1)).reshape(B, H_pad * W_pad, C)
        
        # Reshape for multi-head: (B, H_pad*W_pad, C) -> (B, num_heads, H_pad*W_pad, head_dim)
        K = K.view(B, H_pad * W_pad, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, H_pad * W_pad, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # ---------------------------------------------------------------------
        # 3. Window Partition for K and V.
        # Partition K and V into non-overlapping windows of size (window_size x window_size)
        # using the padded spatial dimensions.
        # ---------------------------------------------------------------------
        K_windows, num_windows = self.window_partition(K, H_pad, W_pad, self.window_size)
        V_windows, _ = self.window_partition(V, H_pad, W_pad, self.window_size)
        # K_windows, V_windows shape: (B * num_windows, num_heads, window_area, head_dim)
        
        # ---------------------------------------------------------------------
        # 4. Align Q with Windows: Broadcast global Q to every window.
        # Q shape: (B, num_heads, 1, head_dim) -> replicate to get (B * num_windows, num_heads, 1, head_dim)
        # ---------------------------------------------------------------------
        Q_windows = Q.unsqueeze(1).expand(B, num_windows, self.num_heads, 1, self.head_dim)
        Q_windows = Q_windows.contiguous().view(B * num_windows, self.num_heads, 1, self.head_dim)
        
        # ---------------------------------------------------------------------
        # 5. Compute Scaled Dot-Product Attention for each window.
        # ---------------------------------------------------------------------
        attn = (Q_windows @ K_windows.transpose(-2, -1)) * self.scale  # shape: (B*num_windows, num_heads, 1, window_area)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out_windows = attn @ V_windows  # shape: (B*num_windows, num_heads, 1, head_dim)
        
        # ---------------------------------------------------------------------
        # 6. Merge Windows: Aggregate outputs from all windows by taking the mean.
        # ---------------------------------------------------------------------
        out_windows = out_windows.view(B, num_windows, self.num_heads, 1, self.head_dim)
        out = out_windows.mean(dim=1)  # shape: (B, num_heads, 1, head_dim)
        
        # ---------------------------------------------------------------------
        # 7. Combine heads and apply the final linear projection.
        # ---------------------------------------------------------------------
        out = out.permute(0, 2, 1, 3).contiguous().view(B, 1, C)  # shape: (B, 1, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        # Expand the global result to every spatial location using the original dimensions.
        out = out.view(B, C, 1, 1).expand(B, C, H_orig, W_orig)
        
        # ---------------------------------------------------------------------
        # 8. Debug prints: shapes and CUDA memory (in MB)
        # ---------------------------------------------------------------------
        if self.debug:
            print("Frequency-to-Spatial Cross-Attention:")
            print("  Q_global shape:", Q_global.shape)
            print("  Q shape after proj & head split:", Q.shape)
            print("  Padded X_s shape:", X_s.shape)
            print("  Shifted X_s shape:", shifted_X_s.shape)
            print("  K shape (multi-head):", K.shape, "V shape (multi-head):", V.shape)
            print("  K_windows shape:", K_windows.shape, "V_windows shape:", V_windows.shape)
            print("  Q_windows shape:", Q_windows.shape)
            print("  attn shape:", attn.shape)
            print("  out_windows shape:", out_windows.shape)
            if torch.cuda.is_available():
                mem_MB = torch.cuda.memory_allocated() / (1024*1024)
                print("CUDA memory allocated (MB): {:.2f}".format(mem_MB))
        
        return out
    
    def cyclic_shift(self, x, shift):
        """
        Perform cyclic shift (roll) on tensor along the height and width.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            shift (tuple): (shift_h, shift_w) amount.
        
        Returns:
            torch.Tensor: Shifted tensor.
        """
        shift_h, shift_w = shift
        return torch.roll(x, shifts=(shift_h, shift_w), dims=(2, 3))
    
    def window_partition(self, x, H, W, window_size):
        """
        Partition the multi-head tensor into non-overlapping windows.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, num_heads, H*W, head_dim).
            H (int): Height of the feature map.
            W (int): Width of the feature map.
            window_size (int): Size of the window.
        
        Returns:
            windows (torch.Tensor): Partitioned tensor of shape
                (B * num_windows, num_heads, window_size*window_size, head_dim).
            num_windows (int): Number of windows per image.
        """
        B, num_heads, L, head_dim = x.shape
        assert L == H * W, "Input token length {} does not equal H*W ({})".format(L, H * W)
        x = x.view(B, num_heads, H, W, head_dim)
        num_windows_H = H // window_size
        num_windows_W = W // window_size
        windows = x.view(B, num_heads, num_windows_H, window_size, num_windows_W, window_size, head_dim)
        windows = windows.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        windows = windows.view(B * (num_windows_H * num_windows_W), num_heads, window_size * window_size, head_dim)
        return windows, num_windows_H * num_windows_W

# ===========================
# Test the Module
# ===========================
if __name__ == "__main__":
    import torch.cuda.amp as amp
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, C, H, W = 8, 32, 49, 65  # example dimensions that are NOT multiples of the window size
    window_size = 8
    num_heads = 1

    # Create dummy inputs:
    # X_f: Frequency features (global DCT features), shape (B, C, H, W)
    # X_s: Spatial features, shape (B, C, H, W)
    X_f = torch.randn(B, C, H, W, device=device)
    X_s = torch.randn(B, C, H, W, device=device)

    model = FrequencyToSpatialCrossAttention(
        embed_dim=C, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2, debug=True
    ).to(device)

    use_amp = (device == "cuda")
    with amp.autocast(enabled=use_amp):
        output = model(X_f, X_s)
        print("Final output shape:", output.shape)

    if device == "cuda":
        mem_alloc_MB = torch.cuda.memory_allocated() / (1024*1024)
        print("CUDA memory allocated (MB): {:.2f}".format(mem_alloc_MB))
