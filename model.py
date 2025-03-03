import torch
import numpy
from attention import FlashAttention, MSA
from models.cheby import ChebyKANLayer
from models.effkan import KANLinear
from models.fastkan import FastKANLayer
from models.nfkan import NaiveFourierKANLayer
from models.sinekan import SineKANLayer

import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """
    A standard Transformer encoder block with:
      - LayerNorm -> Multi-head Attention -> Residual
      - LayerNorm -> FeedForward -> Residual
    """
    def __init__(self, d_model, n_heads, feedforward_dim=128, attn_type="vanilla"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MSA(d_model, n_heads, type=attn_type)  # your MSA or KAN-based attention
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_dim, d_model),
        )

    def forward(self, x):
        # x shape: (batch, n_tokens, d_model)
        # 1) Pre-LN + Attention -> Residual
        x = x + self.attn(self.norm1(x))
        # 2) Pre-LN + FeedForward -> Residual
        x = x + self.ff(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
        A simplified Vision Transformer with:
         - Patchify input -> linear projection
         - [class] token + position embeddings
         - N TransformerBlocks
         - Final classification head
    """

    def __init__(self, chw, n_patches=7, n_blocks=4, d_hidden=64, n_heads=2, out_d=10, type: str = "vanilla"):
        super(VisionTransformer, self).__init__()

        self.chw = chw
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d_hidden = d_hidden

        assert chw[1] % n_patches == 0
        assert chw[2] % n_patches == 0

        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches)

        # Input dimension after flattening one patch
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])

        # Simple linear mapper from patch to d_hidden
        if type == "vanilla" or type == "flash-attn":
            self.linear_mapper = nn.Linear(self.input_d, d_hidden)
        elif type == "efficientkan":
            self.linear_mapper = KANLinear(self.input_d, d_hidden)
        elif type == "sine":
            self.linear_mapper = SineKANLayer(self.input_d, d_hidden, grid_size=28)
        elif type == "fourier":
            self.linear_mapper = NaiveFourierKANLayer(self.input_d, d_hidden, grid_size=28)
        elif type == "cheby":
            self.linear_mapper = ChebyKANLayer(self.input_d, d_hidden, 4)
        elif type == "fast":
            self.linear_mapper = FastKANLayer(self.input_d, d_hidden)
        else:
            raise ValueError(f"Unknown transformer type: {type}")

        # Classification token
        self.v_class = nn.Parameter(torch.randn(1, d_hidden))

        # Positional embedding: (num_patches + 1) x d_hidden
        self.register_buffer(
            'pos_embeddings',
            self.positional_embeddings(n_patches ** 2 + 1, d_hidden),
            persistent=False
        )

        # Encoder blocks
        if type == "flash-attn":
            # If you want to keep your flash-based blocks
            self.blocks = nn.ModuleList([FlashAttention(dim=d_hidden, heads=n_heads) for _ in range(n_blocks)])
        else:
            # Standard block with feed-forward + residual + layernorm
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model=d_hidden, n_heads=n_heads,
                                 feedforward_dim=4*d_hidden,
                                 attn_type=type)
                for _ in range(n_blocks)
            ])

        # Final classification head (simple LN -> Linear)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, out_d)
        )

    def patchify(self, images, n_patches):
        """
        Break down image into patches (n_patches x n_patches).
        """
        n, c, h, w = images.shape
        patches = torch.zeros(n, n_patches**2, self.input_d, device=images.device)
        patch_size_h = h // n_patches
        patch_size_w = w // n_patches
        idx = 0
        for i in range(n_patches):
            for j in range(n_patches):
                patch = images[:, :, i*patch_size_h:(i+1)*patch_size_h,
                                      j*patch_size_w:(j+1)*patch_size_w]
                patches[:, idx] = patch.flatten(start_dim=1)
                idx += 1
        return patches

    def positional_embeddings(self, seq_length, d):
        """
        Create a sinusoidal or custom positional embedding. 
        For simplicity, here's a sample sine/cos approach.
        """
        result = torch.ones(seq_length, d)
        for i in range(seq_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i][j] = numpy.sin(i / 10000 ** (j / d))
                else:
                    result[i][j] = numpy.cos(i / 10000 ** (j / d))
        return result

    def forward(self, images):
        # 1) Turn [B, C, H, W] into [B, (n_patches^2), patch_size^2*C]
        patches = self.patchify(images, self.n_patches)  # (B, #patches, input_d)
        # 2) Map to d_hidden
        tokens = self.linear_mapper(patches)  # (B, #patches, d_hidden)
        # 3) Prepend class token
        bsz = tokens.size(0)
        class_token = self.v_class.unsqueeze(0).expand(bsz, -1, -1)  # (B, 1, d_hidden)
        tokens = torch.cat((class_token, tokens), dim=1)  # (B, #patches+1, d_hidden)
        # 4) Add positional embeddings
        tokens = tokens + self.pos_embeddings[: tokens.size(1), :].to(tokens.device)

        # 5) Pass through each Transformer block
        out = tokens
        if isinstance(self.blocks[0], FlashAttention):
            # If using your flash blocks directly
            for blk in self.blocks:
                out = blk(out)  # or something if you want LN+res outside
        else:
            for blk in self.blocks:
                out = blk(out)

        # 6) Take the [class] token (index=0) for classification
        cls_token = out[:, 0]

        # 7) Classification head
        logits = self.mlp_head(cls_token)
        return logits
