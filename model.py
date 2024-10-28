import torch
import numpy
from attention import FlashAttention, MSA
from models.cheby import ChebyKANLayer
from models.effkan import KANLinear
from models.fastkan import FastKANLayer
from models.nfkan import NaiveFourierKANLayer
from models.sinekan import SineKANLayer


class VisionTransformer(torch.nn.Module):
    """
        Initializes a Vision Transformer (ViT) module.

        Args:
            chw (list/tuple of 3 ints): The input image shape.
            n_patches (int, optional): The number of patches to split the image into. Defaults to 10.
            n_blocks (int, optional): The number of blocks in the transformer encoder. Defaults to 2.
            d_hidden (int, optional): The number of hidden dimensions in the transformer encoder. Defaults to 8.
            n_heads (int, optional): The number of attention heads in each block. Defaults to 2.
            out_d (int, optional): The number of output dimensions. Defaults to 10.

        Returns:
            None
    """

    def __init__(self, chw, n_patches=10, n_blocks=2, d_hidden=8, n_heads=2, out_d=10, type: str = "vanilla"):
        super(VisionTransformer, self).__init__()

        self.chw = chw
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d_hidden = d_hidden

        assert chw[1] % n_patches == 0
        assert chw[2] % n_patches == 0

        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapping
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        if type == "vanilla" or type == "flash-attn":
            self.linear_mapper = torch.nn.Linear(
                self.input_d, self.d_hidden)
        elif type == "efficientkan":
            self.linear_mapper = KANLinear(
                self.input_d, self.d_hidden)
        elif type == "sine":
            self.linear_mapper = SineKANLayer(
                self.input_d, self.d_hidden, grid_size=28)
        elif type == "fourier":
            self.linear_mapper = NaiveFourierKANLayer(
                self.input_d, self.d_hidden, grid_size=28)
        elif type == "cheby":
            self.linear_mapper = ChebyKANLayer(self.input_d, self.d_hidden, 4)
        elif type == "fast":
            self.linear_mapper = FastKANLayer(
                self.input_d, self.d_hidden)
        else:
            print("Variant not available.")

        # Classification token
        self.v_class = torch.nn.Parameter(torch.rand(1, self.d_hidden))

        # Positional embedding
        self.register_buffer('pos_embeddings', self.positional_embeddings(n_patches ** 2 + 1, d_hidden),
                             persistent=False)

        # Encoder blocks
        if type == "flash-attn":
            self.blocks = torch.nn.ModuleList(
                [FlashAttention(d_hidden, n_heads) for _ in range(n_blocks)])
        else:
            self.blocks = torch.nn.ModuleList(
                [MSA(d_hidden, n_heads, type=type) for _ in range(n_blocks)]
            )

        self.mlp = torch.nn.Sequential(
            SineKANLayer(self.d_hidden, out_d, grid_size=4),
            torch.nn.Softmax(dim=-1)
        )

    def patchify(self, images, n_patches):
        """
        The purpose of this function is to break down the main image into multiple sub-images and map them.

        Args:
            images (_type_): The image passeed into this function.
            n_patches (_type_): The number of sub-images that will be created.
        """

        n, c, h, w = images.shape
        assert h == w, "Only for square images"

        # The equation to calculate the patches
        patches = torch.zeros(n, n_patches**2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = image[:, i * patch_size: (
                        i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches

    def positional_embeddings(self, seq_length, d):
        """
        the purpose of this function is to find high and low interaction of a word with surrounding words.
        We can do so by the following equation below:

        Args:
            seq_length (int): The length of the sequence/sentence
            d (int): The dimension of the embedding
        """

        result = torch.ones(seq_length, d)
        for i in range(seq_length):
            for j in range(d):
                result[i][j] = numpy.sin(
                    i / 10000 ** (j / d)) if j % 2 == 0 else numpy.cos(i / 10000 ** (j / d))
        return result

    def forward(self, images):
        n, c, h, w = images.shape
        patches = self.patchify(images, self.n_patches).to(
            self.pos_embeddings.device)

        # running tokenization
        tokens = self.linear_mapper(patches)
        tokens = torch.cat((self.v_class.expand(n, 1, -1), tokens), dim=1)
        out = tokens + self.pos_embeddings.repeat(n, 1, 1)

        for block in self.blocks:
            out = block(out)

        out = out[:, 0]
        return self.mlp(out)
