import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.baseline_model import ISLSeq2SeqBaseline
from src.text import tokenize, build_vocab


def test_tokenizer():
    assert tokenize("Hello, world! 2026") == ["hello", ",", "world", "!", "2026"]
    vocab = build_vocab(["Hello, world!"])
    assert vocab.decode(vocab.encode("Hello, world!")) == "hello, world!"


def test_model_shapes():
    model = ISLSeq2SeqBaseline(12, encoder_hidden=16, decoder_hidden=32, embedding_dim=16, attention_dim=16)
    features = torch.randn(2, 7, 512)
    targets = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]])
    assert model(features, targets).shape == (2, 3, 12)
    assert model.greedy_decode(features, max_len=4).shape[0] == 2
