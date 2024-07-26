import torch
import numpy

numpy.random.seed(42)
torch.manual_seed(42)

def patchify(images, n_patches): 
    """
    The purpose of this function is to break down the main image into multiple sub-images and map them. 

    Args:
        images (_type_): The image passeed into this function. 
        n_patches (_type_): The number of sub-images that will be created.
    """
    
    n, c, h, w = images.shape
    assert h == w, "Only for square images"
    
    patches = torch.zeros(n, n_patches**2, h * w * c // n_patches ** 2) # The equation to calculate the patches
    patch_size = h // n_patches
    
    for idx, image in enumerate(images):
        for i in range(n_patches): 
            for j in range(n_patches): 
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def positional_embeddings(seq_length, d): 
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
            result[i][j] = numpy.sin(i / 10000 ** (j / d)) if j % 2 == 0 else numpy.cos(i / 10000 ** (j/ d))
    return result

class MSA(torch.nn.Module): 
    """
        Initializes the Multi-Head Self-Attention (MSA) module with the given dimensions.

        Args:
            d (int): The total dimension of the input.
            n_heads (int): The number of attention heads.

        Returns:
            None
    """
    def __init__(self, d, n_heads): 
        super(MSA, self).__init__()
        self.d = d 
        self.n_heads = n_heads
        
        assert d % n_heads == 0 
        d_head = int(d / n_heads)
        
        self.q_mappings = torch.nn.ModuleList([torch.nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = torch.nn.ModuleList([torch.nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = torch.nn.ModuleList([torch.nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, sequence): 
        result = [] 
        for sequence in sequence: 
            seq_res = [] 
            for head in range(self.n_heads): 
                q_map = self.q_mappings[head]
                k_map = self.k_mappings[head]
                v_map = self.v_mappings[head]
                
                seq = sequence[:, head*self.d_head: (head+1)*self.d_head]
                q, k, v = q_map(seq), k_map(seq), v_map(seq)
                
                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_res.append(attention @ v)
            result.append(torch.hstack(seq_res))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class Residual(torch.nn.Module): 
    """
        Initializes a Residual module.

        Args:
            d_hidden (int): The number of hidden dimensions.
            n_heads (int): The number of attention heads.
            mlp_ratio (int, optional): The ratio of the number of hidden dimensions in the MLP layer. Defaults to 4.

        Returns:
            None
    """
    def __init___(self, d_hidden, n_heads, mlp_ratio=4): 
        super(Residual, self).__init__()
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.norm1 = torch.nn.LayerNorm(d_hidden)
        self.mhsa = MSA(d_hidden, n_heads)
        self.ml = torch.nn.Sequential(
            torch.nn.Linear(d_hidden, mlp_ratio * d_hidden), 
            torch.nn.GELU(), 
            torch.nn.Linear(mlp_ratio * d_hidden, d_hidden)
        )
        
    def forward(self, x): 
        out = x = self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(x))
        return out
    
class ViT(torch.nn.Module): 
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
    def __init__(self, chw, n_patches=10, n_blocks=2, d_hidden=8, n_heads=2, out_d=10): 
        super(ViT, self).__init__()
        
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
        self.linear_mapper = torch.nn.Linear(self.input_d, self.d_hidden)

        # Classification token
        self.v_class = torch.nn.Parameter(torch.rand(1, self.d_hidden))

        # Positional embedding
        self.register_buffer('positional_embeddings', positional_embeddings(n_patches ** 2 + 1, d_hidden),
                             persistent=False)

        # Encoder blocks
        self.blocks = torch.nn.ModuleList([MSA(d_hidden, n_heads) for _ in range(n_blocks)])

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.d_hidden, out_d),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, images):
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # running tokenization
        tokens = self.linear_mapper(patches)
        tokens = torch.cat((self.v_class.expand(n, 1, -1), tokens), dim=1)
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        for block in self.blocks:
            out = block(out)

        out = out[:, 0]
        return self.mlp(out)