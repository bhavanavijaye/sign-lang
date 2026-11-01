"""Slide-style model: keypoints -> Bi-LSTM (sign patterns) -> Transformer decoder -> English words.

 - BiLSTMTransformer : Bi-LSTM encoder + Transformer decoder (sequence-to-sequence), trained from scratch
 - BiLSTMClassifier  : Bi-LSTM + softmax over the known sentences (closed-set baseline)
"""
import math
import torch, torch.nn as nn
from data import FEAT_DIM

PAD, BOS, EOS, UNK = 0, 1, 2, 3


class PosEnc(nn.Module):
    def __init__(self, d, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class KpEncoder(nn.Module):
    """(B,T,kp_dim) keypoints -> (B,T,2*hid) Bi-LSTM features."""

    def __init__(self, kp_dim=FEAT_DIM, hid=256, layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(kp_dim), nn.Linear(kp_dim, hid), nn.GELU(), nn.Dropout(0.2))
        self.lstm = nn.LSTM(hid, hid, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout)

    def forward(self, kp, mask):
        x = self.proj(kp)
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out


class BiLSTMTransformer(nn.Module):
    def __init__(self, vocab_size, kp_dim=FEAT_DIM, hid=256, d_model=256, nhead=4, n_dec=3):
        super().__init__()
        self.d = d_model
        self.enc = KpEncoder(kp_dim, hid)
        self.enc_out = nn.Linear(2 * hid, d_model)
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PosEnc(d_model)
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                           dropout=0.2, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, n_dec)
        self.out = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal(L, device):
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1)

    def forward(self, kp, mask, tgt_in):
        mem = self.enc_out(self.enc(kp, mask))
        y = self.pos(self.emb(tgt_in) * math.sqrt(self.d))
        y = self.dec(y, mem, tgt_mask=self._causal(tgt_in.size(1), tgt_in.device),
                     tgt_key_padding_mask=(tgt_in == PAD),
                     memory_key_padding_mask=(mask == 0))
        return self.out(y)

    @torch.no_grad()
    def greedy(self, kp, mask, max_len=24):
        mem = self.enc_out(self.enc(kp, mask))
        mem_pad = (mask == 0)
        B = kp.size(0)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=kp.device)
        done = torch.zeros(B, dtype=torch.bool, device=kp.device)
        for _ in range(max_len):
            y = self.pos(self.emb(ys) * math.sqrt(self.d))
            y = self.dec(y, mem, tgt_mask=self._causal(ys.size(1), kp.device),
                         memory_key_padding_mask=mem_pad)
            nxt = self.out(y[:, -1]).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        return ys[:, 1:]


class BiLSTMClassifier(nn.Module):
    def __init__(self, n_classes, kp_dim=FEAT_DIM, hid=256):
        super().__init__()
        self.enc = KpEncoder(kp_dim, hid)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * hid, n_classes)

    def forward(self, kp, mask):
        out = self.enc(kp, mask)
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)
        return self.fc(self.drop(pooled))