from typing import List

import torch


class WordTokenizer:
    """
    The WordTokenizer class provides a flexible and customizable tokenization tool for processing textual data.
    It encapsulates the logic to convert input sentences or lists of words into sequences of token IDs, suitable for nano BERT.
    This tokenizer adds special tokens, such as '[PAD]', '[CLS]', '[SEP]', and '[UNK]', to the vocabulary,
    It offers methods to encode input text, decode token IDs back to human-readable sentences, and can be used as a callable object for quick tokenization.
    The class is designed to handle both single sentences and batches of sentences.
    """

    def __init__(self, vocab: List[str], sep=' ', max_seq_len=16, special_tokens: List[str] | None = None):
        # Call the constructor of the parent class (object)
        super().__init__()

        # Initialize tokenizer properties
        self.max_seq_len = max_seq_len  # Maximum sequence length for tokenization
        self.sep = sep  # Separator used to split input sentences into words
        if special_tokens is None:
            # Default special tokens used in tokenization if not provided
            self.special_tokens = [
                '[PAD]',
                '[CLS]',
                '[SEP]',
                '[UNK]',
            ]

        # Create a vocabulary dictionary with special tokens and input vocabulary
        self.vocab = {word: i for i, word in enumerate(self.special_tokens + sorted(vocab))}
        # Create a reverse vocabulary dictionary for decoding purposes
        self.de_vocab = {i: word for word, i in self.vocab.items()}

    # Method to encode input sentence(s) into token IDs
    def encode(self, sentence: str | List[str]):
        """
        :param sentence: a string (will be split by 'sep') or a list of tokens (already split so each word will be encoded)
        :return: torch.Tensor((max_seq_len,), dtype=torch.long) - ids of encoded tokens
        """
        # If input is a string, split it into a list of words
        if isinstance(sentence, str):
            return self.encode(sentence.split(self.sep))

        # Truncate the input sentence to fit within the maximum sequence length
        sentence = sentence[:self.max_seq_len - 2]
        # Convert words to lowercase
        sentence = [w.lower() for w in sentence]
        # Add special tokens ([CLS], [SEP]) and padding tokens ([PAD]) to the input sentence
        sentence = ['[CLS]'] + sentence + ['[SEP]'] + ['[PAD]'] * (self.max_seq_len - len(sentence) - 2)
        # Map words to their corresponding IDs in the vocabulary or use [UNK] token if not found
        return torch.tensor([self.vocab[w] if w in self.vocab else self.vocab['[UNK]'] for w in sentence],
                            dtype=torch.long)

    # Method to decode token IDs back to original sentence
    def decode(self, ids: List, ignore_special=True):
        # Join token IDs to form a sentence, ignoring special tokens if specified
        return self.sep.join(
            [self.de_vocab[id] for id in ids if self.de_vocab[id] not in self.special_tokens or not ignore_special])

    # Callable method to allow using the object as a function for tokenization
    def __call__(self, sentence: str | List[str]):
        # If input is a string, tokenize it; otherwise, tokenize the list of words
        if isinstance(sentence, str):
            return self.encode(sentence.split(self.sep))
        return self.encode(sentence)

    # Method to provide a string representation of the tokenizer object
    def __repr__(self):
        return f'Tokenizer[vocab={len(self.vocab)},{self.special_tokens=},{self.sep=},{self.max_seq_len=}]'
