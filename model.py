import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import FlashAttention, MSA
from models.cheby import ChebyKANLayer
from models.effkan import KANLinear
from models.fastkan import FastKANLayer
from models.nfkan import NaiveFourierKANLayer
from models.sinekan import SineKANLayer


class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block with LN -> MSA -> Residual, LN -> FF -> Residual
    """

    def __init__(self, d_model, n_heads, feedforward_dim=128, attn_type="vanilla"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MSA(d_model, n_heads, type=attn_type)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_dim, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Simplified Vision Transformer for MNIST:
    - patchify (7x7) -> linear
    - class token + pos embeddings
    - N transformer blocks
    - final classification head
    """

    def __init__(self, chw, n_patches=7, n_blocks=4, d_hidden=64, n_heads=2,
                 out_d=10, type: str = "vanilla"):
        super(VisionTransformer, self).__init__()

        self.chw = chw
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d_hidden = d_hidden

        assert chw[1] % n_patches == 0
        assert chw[2] % n_patches == 0

        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches)

        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])

        if type == "vanilla" or type == "flash-attn":
            self.linear_mapper = nn.Linear(self.input_d, d_hidden)
        elif type == "efficientkan":
            self.linear_mapper = KANLinear(self.input_d, d_hidden)
        elif type == "sine":
            self.linear_mapper = SineKANLayer(self.input_d, d_hidden, grid_size=28)
        elif type == "fourier":
            self.linear_mapper = NaiveFourierKANLayer(self.input_d, d_hidden, gridsize=28)
        elif type == "cheby":
            self.linear_mapper = ChebyKANLayer(self.input_d, d_hidden, 4)
        elif type == "fast":
            self.linear_mapper = FastKANLayer(self.input_d, d_hidden)
        else:
            raise ValueError(f"Unknown transformer type: {type}")

        self.v_class = nn.Parameter(torch.randn(1, d_hidden))

        self.register_buffer(
            'pos_embeddings',
            self.positional_embeddings(n_patches ** 2 + 1, d_hidden),
            persistent=False
        )

        if type == "flash-attn":
            self.blocks = nn.ModuleList([FlashAttention(dim=d_hidden, heads=n_heads)
                                         for _ in range(n_blocks)])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model=d_hidden,
                                 n_heads=n_heads,
                                 feedforward_dim=4*d_hidden,
                                 attn_type=type)
                for _ in range(n_blocks)
            ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, out_d)
        )

    def patchify(self, images, n_patches):
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
        result = torch.ones(seq_length, d)
        for i in range(seq_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i][j] = torch.sin(torch.tensor(i / 10000 ** (j / d)))
                else:
                    result[i][j] = torch.cos(torch.tensor(i / 10000 ** (j / d)))
        return result

    def forward(self, images):
        patches = self.patchify(images, self.n_patches)
        tokens = self.linear_mapper(patches)
        bsz = tokens.size(0)
        class_token = self.v_class.unsqueeze(0).expand(bsz, -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        tokens = tokens + self.pos_embeddings[: tokens.size(1), :].to(tokens.device)

        out = tokens
        if isinstance(self.blocks[0], FlashAttention):
            for blk in self.blocks:
                out = blk(out)
        else:
            for blk in self.blocks:
                out = blk(out)

        cls_token = out[:, 0]
        logits = self.mlp_head(cls_token)
        return logits
