import torch
import numpy

numpy.random.seed(42)
torch.manual_seed(42)

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