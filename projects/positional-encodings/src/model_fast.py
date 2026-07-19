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
    def __init__(self, d_model, n_heads, max_seq_len=512, masked=True, use_alibi=False, use_rope=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.masked = masked
        self.use_alibi = use_alibi
        self.use_rope = use_rope
        self.W_qkv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)

        if use_alibi:
            slopes = self._get_alibi_slopes(n_heads)
            positions = torch.arange(max_seq_len)
            rel_dist = (positions[:, None] - positions[None, :]).clamp(min=0)
            alibi_bias = -slopes[:, None, None] * rel_dist[None, :, :]
            if masked:
                causal = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
                alibi_bias = alibi_bias.masked_fill(causal[None], float('-inf'))
            self.register_buffer('alibi_bias', alibi_bias)

        if use_rope:
            cos, sin = build_rope_cache(max_seq_len, self.head_dim)
            self.register_buffer('rope_cos', cos)
            self.register_buffer('rope_sin', sin)

    @staticmethod
    def _get_alibi_slopes(n_heads):
        start = 2 ** (-8 / n_heads)
        return torch.tensor([start ** (i + 1) for i in range(n_heads)])

    def _apply_rope(self, x, T):
        return x * self.rope_cos[:T] + rotate_half(x) * self.rope_sin[:T]

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.W_qkv(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = self._apply_rope(q, T)
            k = self._apply_rope(k, T)

        if self.use_alibi:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.alibi_bias[None, :, :T, :T])
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=self.masked)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len=512, masked=True, use_alibi=False, use_rope=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_seq_len, masked, use_alibi, use_rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.down(F.gelu(self.up(self.norm2(x)), approximate='tanh'))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len=512, pos_encoding='learned'):
        super().__init__()
        assert pos_encoding in ('learned', 'alibi', 'rope', 'none')
        self.pos_encoding = pos_encoding
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model) if pos_encoding == 'learned' else None
        use_alibi = pos_encoding == 'alibi'
        use_rope = pos_encoding == 'rope'
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, True, use_alibi, use_rope)
            for _ in range(n_layers)
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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
