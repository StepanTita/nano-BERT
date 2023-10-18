from typing import List

import torch


class WordTokenizer:
    def __init__(self, vocab: List[str], sep=' ', max_seq_len=16, special_tokens: List[str] | None = None):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.sep = sep
        if special_tokens is None:
            self.special_tokens = [
                '[PAD]',
                '[CLS]',
                '[SEP]',
                '[UNK]',
            ]

        self.vocab = {word: i for i, word in enumerate(self.special_tokens + sorted(vocab))}
        self.de_vocab = {i: word for word, i in self.vocab.items()}

    def encode(self, sentence: str | List[str]):
        if isinstance(sentence, str):
            return self.encode(sentence.split(self.sep))
        sentence = sentence[:self.max_seq_len - 2]
        sentence = [w.lower() for w in sentence]
        sentence = ['[CLS]'] + sentence + ['[SEP]'] + ['[PAD]'] * (self.max_seq_len - len(sentence) - 2)
        return torch.tensor([self.vocab[w] if w in self.vocab else self.vocab['[UNK]'] for w in sentence],
                            dtype=torch.long)

    def decode(self, ids: List, ignore_special=True):
        return self.sep.join(
            [self.de_vocab[id] for id in ids if self.de_vocab[id] not in self.special_tokens or not ignore_special])

    def __call__(self, sentence: str | List[str]):
        if isinstance(sentence, str):
            return self.encode(sentence.split(self.sep))
        return self.encode(sentence)

    def __repr__(self):
        return f'Tokenizer[vocab={len(self.vocab)},{self.special_tokens=},{self.sep=},{self.max_seq_len=}]'
