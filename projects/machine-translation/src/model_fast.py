"""Same seq2seq model as model.py, but built on torch's cuDNN nn.LSTM.

This is the version used for training; model.py is the hand-written reference.
The encoder packs padded sequences so trailing pad steps don't affect the final
state.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


class FastEncoder(nn.Module):
    def __init__(self, vocab_size, n_layers, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=n_layers, batch_first=True)

    def forward(self, src, src_len=None):
        emb = self.embedding(src)
        if src_len is not None:
            emb = pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.lstm(emb)
        return h_n, c_n


class FastDecoder(nn.Module):
    def __init__(self, vocab_size, n_layers, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt_in, state):
        out, _ = self.lstm(self.embedding(tgt_in), state)
        return self.output(out)

    @torch.no_grad()
    def generate(self, state, bos_id=BOS_ID, eos_id=EOS_ID, max_len=100, device="cuda"):
        B = state[0].shape[1]
        tok = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outputs = [[] for _ in range(B)]
        for _ in range(max_len):
            out, state = self.lstm(self.embedding(tok), state)
            nxt = self.output(out[:, -1]).argmax(dim=-1)
            for i in range(B):
                if not finished[i]:
                    outputs[i].append(nxt[i].item())
                    if nxt[i].item() == eos_id:
                        finished[i] = True
            if finished.all():
                break
            tok = nxt.unsqueeze(1)
        return outputs


class Seq2SeqFast(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = FastEncoder(cfg["src_vocab_size"], cfg["n_layers"], cfg["embed_size"], cfg["hidden_size"])
        self.decoder = FastDecoder(cfg["tgt_vocab_size"], cfg["n_layers"], cfg["embed_size"], cfg["hidden_size"])

    def forward(self, src, tgt_in, src_len=None):
        state = self.encoder(src, src_len)
        return self.decoder(tgt_in, state)
