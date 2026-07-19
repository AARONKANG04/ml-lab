import torch
import torch.nn as nn

def init_zero_states(n_layers, batch_size, hidden_size, device):
    return [
        (torch.zeros(batch_size, hidden_size, device=device),
         torch.zeros(batch_size, hidden_size, device=device))
        for _ in range(n_layers)
    ]

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size, 4*hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4*hidden_size)

    def forward(self, x, h_prev, c_prev):
        gates = self.W_ih(x) + self.W_hh(h_prev)
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = c_prev * f + i * g
        h = o * torch.tanh(c)
        return h, c
    

class LSTM(nn.Module):
    def __init__(self, n_layers, input_size, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size) 
            for i in range(n_layers)
        ])

    def forward(self, x_t, states):
        new_states = []
        input_to_layer = x_t
        for layer, (h_prev, c_prev) in zip(self.layers, states):
            h, c = layer(input_to_layer, h_prev, c_prev)
            new_states.append((h, c))
            input_to_layer = h
        return input_to_layer, new_states
    

class OutputHead(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, h_top):
        return self.proj(h_top)


class Encoder(nn.Module):
    def __init__(self, vocab_size, n_layers, embed_size, hidden_size):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = LSTM(n_layers, embed_size, hidden_size)

    def forward(self, x):
        B, T = x.shape
        embedded = self.embedding(x)
        states = init_zero_states(self.n_layers, B, self.hidden_size, device=x.device)
        for t in range(T):
            _, states = self.lstm(embedded[:, t, :], states)
        return states
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_layers, embed_size, hidden_size):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = LSTM(n_layers, embed_size, hidden_size)
        self.output = OutputHead(vocab_size, hidden_size)

    def forward(self, tgt_in, states):
        B, T = tgt_in.shape
        embedded = self.embedding(tgt_in)

        all_logits = []
        for t in range(T):
            h_top, states = self.lstm(embedded[:, t, :], states)
            logits_t = self.output(h_top)
            all_logits.append(logits_t)

        return torch.stack(all_logits, dim=1)
    
    @torch.no_grad()
    def generate(self, states, bos_id, eos_id, max_len, device):
        B = states[0][0].shape[0]
        input_tok = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outputs = [[] for _ in range(B)]

        for _ in range(max_len):
            embedded = self.embedding(input_tok[:, 0])
            h_top, states = self.lstm(embedded, states)
            logits = self.output(h_top)
            next_token = logits.argmax(dim=-1)

            for i in range(B):
                if not finished[i]:
                    outputs[i].append(next_token[i].item())
                    if next_token[i].item() == eos_id:
                        finished[i] = True

            if finished.all():
                break

            input_tok = next_token.unsqueeze(1)

        return outputs


class Seq2Seq(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg["src_vocab_size"], cfg["n_layers"], cfg["embed_size"], cfg["hidden_size"])
        self.decoder = Decoder(cfg["tgt_vocab_size"], cfg["n_layers"], cfg["embed_size"], cfg["hidden_size"])

    def forward(self, src, tgt_in):
        states = self.encoder(src)
        return self.decoder(tgt_in, states)