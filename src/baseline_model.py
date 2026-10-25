"""Reference attention BiGRU encoder + GRU decoder baseline."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.enc = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.dec = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, enc_outputs, hidden, feature_mask=None):
        scores = self.score(torch.tanh(self.enc(enc_outputs) + self.dec(hidden).unsqueeze(1))).squeeze(-1)
        if feature_mask is not None:
            mask_value = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(~feature_mask.bool(), mask_value)
        weights = torch.softmax(scores, dim=1)
        if feature_mask is not None:
            weights = weights.masked_fill(~feature_mask.bool(), 0.0)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


class ISLSeq2SeqBaseline(nn.Module):
    def __init__(self, vocab_size, input_dim=512, encoder_hidden=256, decoder_hidden=512,
                 embedding_dim=256, attention_dim=256, pad_id=0, bos_id=1, eos_id=2, unk_id=3, dropout=0.1,
                 use_embedding_in_output=True, word_dropout=0.0):
        super().__init__()
        self.encoder = nn.GRU(input_dim, encoder_hidden, batch_first=True, bidirectional=True)
        enc_dim = encoder_hidden * 2
        self.bridge = nn.Linear(enc_dim, decoder_hidden)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.attention = AdditiveAttention(enc_dim, decoder_hidden, attention_dim)
        self.decoder = nn.GRUCell(embedding_dim + enc_dim, decoder_hidden)
        self.dropout = nn.Dropout(dropout)
        self.use_embedding_in_output = bool(use_embedding_in_output)
        self.word_dropout = float(word_dropout)
        out_dim = decoder_hidden + enc_dim + (embedding_dim if self.use_embedding_in_output else 0)
        self.output = nn.Linear(out_dim, vocab_size)
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = pad_id, bos_id, eos_id, unk_id
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.dropout_p = dropout

    def encode(self, features, feature_mask=None):
        if feature_mask is None:
            outputs, hidden = self.encoder(features)
        else:
            lengths = feature_mask.long().sum(dim=1).clamp(min=1)
            packed = pack_padded_sequence(
                features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, hidden = self.encoder(packed)
            outputs, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=features.size(1)
            )
        init = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return outputs, torch.tanh(self.bridge(init))

    def forward(self, features, target_ids, feature_mask=None):
        """features [B,T,F], target_ids [B,L] BOS...EOS; returns [B,L-1,V] logits for target_ids[:, 1:]."""
        enc_outputs, hidden = self.encode(features, feature_mask)
        decoder_inputs = target_ids[:, :-1]
        if self.training and self.word_dropout > 0.0 and decoder_inputs.size(1) > 1:
            decoder_inputs = decoder_inputs.clone()
            eligible = (decoder_inputs[:, 1:] != self.pad_id) & (decoder_inputs[:, 1:] != self.bos_id)
            mask = (torch.rand(eligible.shape, device=decoder_inputs.device) < self.word_dropout) & eligible
            decoder_inputs[:, 1:][mask] = self.unk_id
        logits = []
        for token in decoder_inputs.unbind(dim=1):
            emb = self.dropout(self.embedding(token))
            context, _ = self.attention(enc_outputs, hidden, feature_mask)
            hidden = self.decoder(torch.cat([emb, context], dim=-1), hidden)
            out_repr = torch.cat([hidden, context, emb], dim=-1) if self.use_embedding_in_output else torch.cat([hidden, context], dim=-1)
            logits.append(self.output(out_repr))
        if not logits:
            return features.new_zeros((features.shape[0], 0, self.vocab_size))
        return torch.stack(logits, dim=1)

    @torch.inference_mode()
    def greedy_decode(self, features, feature_mask=None, max_len=60):
        was_training = self.training
        self.eval()
        try:
            enc_outputs, hidden = self.encode(features, feature_mask)
            token = torch.full((features.shape[0],), self.bos_id, dtype=torch.long, device=features.device)
            outputs = []
            finished = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
            for _ in range(max_len):
                emb = self.embedding(token)
                context, _ = self.attention(enc_outputs, hidden, feature_mask)
                hidden = self.decoder(torch.cat([emb, context], dim=-1), hidden)
                out_repr = torch.cat([hidden, context, emb], dim=-1) if self.use_embedding_in_output else torch.cat([hidden, context], dim=-1)
                logits = self.output(out_repr)
                logits[:, self.pad_id] = torch.finfo(logits.dtype).min
                token = logits.argmax(-1)
                token = torch.where(finished, torch.full_like(token, self.pad_id), token)
                outputs.append(token)
                finished = finished | token.eq(self.eos_id)
                if bool(finished.all()):
                    break
            if not outputs:
                return torch.empty((features.shape[0], 0), dtype=torch.long, device=features.device)
            return torch.stack(outputs, dim=1)
        finally:
            self.train(was_training)


class ISLSeq2SeqDecoupled(ISLSeq2SeqBaseline):
    """Controlled experimental variant: decoder output projection uses decoder hidden
    and attended visual context, omitting the direct previous-token embedding concatenation."""
    def __init__(self, *args, **kwargs):
        kwargs["use_embedding_in_output"] = False
        super().__init__(*args, **kwargs)
