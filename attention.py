import torch
from torch import nn
from einops import rearrange
from torch.cuda.amp import autocast, GradScaler
from torch.nn import DataParallel
from models.effkan import KANLinear
from models.sinekan import SineKANLayer
from models.cheby import ChebyKANLayer
from models.fastkan import FastKANLayer
from utils import default, FlashAttentionFunction  # from your local utils or wherever it's defined


class FlashAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=8,
        dim_head=64,
        causal=False,
        q_bucket_size=512,
        k_bucket_size=1024,
        parallel=False,
        mixed_precision=False
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.parallel = parallel
        self.mixed_precision = mixed_precision

        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

        if self.parallel:
            self.model = DataParallel(self)
        if self.mixed_precision:
            self.scaler = GradScaler()

    def forward(
        self,
        x,
        context=None,
        mask=None,
        q_bucket_size=None,
        k_bucket_size=None,
    ):
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)

        h = self.heads
        context = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        if self.parallel:
            num_gpus = torch.cuda.device_count()
            x_chunks = x.split(x.size(0) // num_gpus)
            x_chunks = [chunk.to(f'cuda:{i}')
                        for i, chunk in enumerate(x_chunks)]
            q = x_chunks

        if self.mixed_precision:
            with autocast():
                out = FlashAttentionFunction.apply(
                    q, k, v, mask, self.causal, q_bucket_size, k_bucket_size)
        else:
            out = FlashAttentionFunction.apply(
                q, k, v, mask, self.causal, q_bucket_size, k_bucket_size)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MSA(nn.Module):
    """
    Example multi-head attention that can use KAN-based mappings or vanilla linear.
    """

    def __init__(self, d, n_heads=4, type: str = "vanilla"):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads
        assert d % n_heads == 0

        d_head = d // n_heads

        if type in ["vanilla", "flash-attn", "fourier"]:
            self.q_mappings = nn.ModuleList(
                [nn.Linear(d_head, d_head) for _ in range(n_heads)])
            self.k_mappings = nn.ModuleList(
                [nn.Linear(d_head, d_head) for _ in range(n_heads)])
            self.v_mappings = nn.ModuleList(
                [nn.Linear(d_head, d_head) for _ in range(n_heads)])
        elif type == "efficientkan":
            self.q_mappings = nn.ModuleList(
                [KANLinear(d_head, d_head) for _ in range(n_heads)])
            self.k_mappings = nn.ModuleList(
                [KANLinear(d_head, d_head) for _ in range(n_heads)])
            self.v_mappings = nn.ModuleList(
                [KANLinear(d_head, d_head) for _ in range(n_heads)])
        elif type == "fast":
            self.q_mappings = nn.ModuleList(
                [FastKANLayer(d_head, d_head) for _ in range(n_heads)])
            self.k_mappings = nn.ModuleList(
                [FastKANLayer(d_head, d_head) for _ in range(n_heads)])
            self.v_mappings = nn.ModuleList(
                [FastKANLayer(d_head, d_head) for _ in range(n_heads)])
        elif type == "sine":
            self.q_mappings = nn.ModuleList(
                [SineKANLayer(d_head, d_head, grid_size=4) for _ in range(n_heads)])
            self.k_mappings = nn.ModuleList(
                [SineKANLayer(d_head, d_head, grid_size=4) for _ in range(n_heads)])
            self.v_mappings = nn.ModuleList(
                [SineKANLayer(d_head, d_head, grid_size=4) for _ in range(n_heads)])
        elif type == "cheby":
            self.q_mappings = nn.ModuleList(
                [ChebyKANLayer(d_head, d_head, 4) for _ in range(n_heads)])
            self.k_mappings = nn.ModuleList(
                [ChebyKANLayer(d_head, d_head, 4) for _ in range(n_heads)])
            self.v_mappings = nn.ModuleList(
                [ChebyKANLayer(d_head, d_head, 4) for _ in range(n_heads)])
        else:
            raise ValueError(f"{type} invalid. Please use a different argument.")

        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        # sequences => (batch_size, n_tokens, d)
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                # slice for one head
                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            # concat across heads
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
