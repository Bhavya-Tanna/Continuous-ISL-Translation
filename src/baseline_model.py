"""Reference attention BiGRU encoder + GRU decoder baseline; training loop intentionally omitted."""
import torch
from torch import nn


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.enc = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.dec = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, enc_outputs, hidden):
        scores = self.score(torch.tanh(self.enc(enc_outputs) + self.dec(hidden).unsqueeze(1))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


class ISLSeq2SeqBaseline(nn.Module):
    def __init__(self, vocab_size, input_dim=512, encoder_hidden=256, decoder_hidden=512,
                 embedding_dim=256, attention_dim=256, pad_id=0, bos_id=1, eos_id=2, dropout=0.1):
        super().__init__()
        self.encoder = nn.GRU(input_dim, encoder_hidden, batch_first=True, bidirectional=True)
        enc_dim = encoder_hidden * 2
        self.bridge = nn.Linear(enc_dim, decoder_hidden)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.attention = AdditiveAttention(enc_dim, decoder_hidden, attention_dim)
        self.decoder = nn.GRUCell(embedding_dim + enc_dim, decoder_hidden)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(decoder_hidden + enc_dim + embedding_dim, vocab_size)
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.vocab_size = vocab_size

    def encode(self, features):
        outputs, hidden = self.encoder(features)
        # hidden [2, B, H] for one bidirectional layer
        init = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return outputs, torch.tanh(self.bridge(init))

    def forward(self, features, target_ids):
        """features [B,T,F], target_ids [B,L] BOS...EOS; returns [B,L-1,V] logits."""
        enc_outputs, hidden = self.encode(features)
        logits = []
        for token in target_ids[:, :-1].unbind(dim=1):
            emb = self.dropout(self.embedding(token))
            context, _ = self.attention(enc_outputs, hidden)
            hidden = self.decoder(torch.cat([emb, context], dim=-1), hidden)
            logits.append(self.output(torch.cat([hidden, context, emb], dim=-1)))
        if not logits:
            return features.new_zeros((features.shape[0], 0, self.vocab_size))
        return torch.stack(logits, dim=1)

    @torch.inference_mode()
    def greedy_decode(self, features, max_len=60):
        self.eval()
        enc_outputs, hidden = self.encode(features)
        token = torch.full((features.shape[0],), self.bos_id, dtype=torch.long, device=features.device)
        outputs = []
        finished = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
        for _ in range(max_len):
            emb = self.dropout(self.embedding(token))
            context, _ = self.attention(enc_outputs, hidden)
            hidden = self.decoder(torch.cat([emb, context], dim=-1), hidden)
            token = self.output(torch.cat([hidden, context, emb], dim=-1)).argmax(-1)
            outputs.append(token)
            finished |= token.eq(self.eos_id)
            if bool(finished.all()): break
        return torch.stack(outputs, dim=1) if outputs else torch.empty((features.shape[0], 0), dtype=torch.long, device=features.device)
