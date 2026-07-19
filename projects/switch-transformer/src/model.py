import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def build_rope_cache(seq_len, head_dim, base=10000.0):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, theta)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=512):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)
        cos, sin = build_rope_cache(max_seq_len, self.head_dim)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def _apply_rope(self, x, T):
        return x * self.rope_cos[:T] + rotate_half(x) * self.rope_sin[:T]

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.W_qkv(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate='tanh'))

class SwitchFFN(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, capacity_factor=1.25):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([FFN(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        B, T, C = x.shape
        tokens = x.reshape(-1, C)
        N = tokens.size(0)

        with torch.autocast(device_type=x.device.type, enabled=False):
            probs = F.softmax(self.router(tokens.float()), dim=-1)
        gate, choice = probs.max(dim=-1)

        counts = torch.bincount(choice, minlength=self.n_experts)
        f = counts.float() / N
        P = probs.mean(dim=0)
        self.aux_loss = self.n_experts * (f * P).sum()
        self.load = f.detach()
        self.last_choice = choice.detach().view(B, T)

        capacity = max(1, int(self.capacity_factor * N / self.n_experts))
        out = torch.zeros_like(tokens)
        dropped = 0
        for e in range(self.n_experts):
            idx = (choice == e).nonzero(as_tuple=True)[0]
            if idx.numel() > capacity:
                dropped += idx.numel() - capacity
                idx = idx[:capacity]
            if idx.numel() == 0:
                continue
            out[idx] = (self.experts[e](tokens[idx]) * gate[idx, None]).to(out.dtype)
        self.drop_frac = dropped / N
        return out.reshape(B, T, C)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len=512, n_experts=0, capacity_factor=1.25):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        if n_experts > 0:
            self.ffn = SwitchFFN(d_model, d_ff, n_experts, capacity_factor)
        else:
            self.ffn = FFN(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len=512,
                 n_experts=0, capacity_factor=1.25, moe_every=2):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_experts = n_experts
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len,
                             n_experts if i % moe_every == 1 else 0, capacity_factor)
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def moe_layers(self):
        return [(i, b.ffn) for i, b in enumerate(self.blocks) if isinstance(b.ffn, SwitchFFN)]

    def aux_loss(self):
        return sum(m.aux_loss for _, m in self.moe_layers())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
