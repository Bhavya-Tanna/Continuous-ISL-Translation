"""Tokenizer and vocabulary helpers. Build vocabulary from training targets only."""
import re
from collections import Counter
from dataclasses import dataclass

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")


def tokenize(text):
    return [t.lower() for t in PATTERN.findall(str(text))]


@dataclass
class Vocabulary:
    stoi: dict
    itos: list

    @property
    def pad_id(self): return self.stoi[PAD]
    @property
    def bos_id(self): return self.stoi[BOS]
    @property
    def eos_id(self): return self.stoi[EOS]
    @property
    def unk_id(self): return self.stoi[UNK]

    def encode(self, text, add_boundaries=True):
        ids = [self.stoi.get(t, self.unk_id) for t in tokenize(text)]
        return [self.bos_id, *ids, self.eos_id] if add_boundaries else ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            token = self.itos[int(i)] if 0 <= int(i) < len(self.itos) else UNK
            if token == EOS: break
            if token not in {PAD, BOS}: tokens.append(token)
        return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(tokens))


def build_vocab(texts, min_freq=1):
    counts = Counter(t for text in texts for t in tokenize(text))
    words = sorted(w for w, count in counts.items() if count >= min_freq and w not in SPECIALS)
    itos = SPECIALS + words
    return Vocabulary({w: i for i, w in enumerate(itos)}, itos)
