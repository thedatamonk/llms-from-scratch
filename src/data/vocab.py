"""
References: https://github.com/jsbaan/transformer-from-scratch/blob/main/vocabulary.py
"""
import re
from typing import List, Optional


class Vocabulary:
    BOS = "BOS"
    EOS = "EOS"
    PAD = "PAD"

    def __init__(self, list_of_sentences: Optional[List[str]]):
        self.token2index = {self.BOS: 0, self.EOS: 1, self.PAD: 2}
        self.index2token = {v: k for k, v in self.token2index.items()}
        if not list_of_sentences:
            return
        for sentence in list_of_sentences:
            self.add_tokens(self.tokenize(sentence))

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Adds tokens the vocabulary

        :param tokens:
        :return: None
        """
        for token in tokens:
            if token not in self.token2index:
                i = len(self.token2index.items())
                self.token2index[token] = i
                self.index2token[i] = token

    def tokenize(self, sentence: str, add_special_tokens: bool = True) -> List[str]:
        """
        Split on all tokens and punctuation. Optionally adds BOS and EOS tokens.

        :param sentence:
        :param add_special_tokens:
        :return: List of string tokens
        """
        tokens = re.findall(r"\w+|[^\s\w]+", sentence)
        if add_special_tokens:
            tokens = [self.BOS] + tokens + [self.EOS]
        return tokens

    def encode(self, sentence: str, add_special_tokens: bool = True) -> List[int]:
        """
        Converts a string to a list of token indices given the vocabulary

        :param sentence: a string representing a sentence
        :param add_special_tokens: whether or not to add a bos and eos token
        :return: list of token indices
        """
        tokens = self.tokenize(sentence, add_special_tokens)
        return [self.token2index[token] for token in tokens]

    def batch_encode(
        self, sentences: List[str], padding=True, add_special_tokens: bool = False
    ) -> List[List[int]]:
        """
        Convert a list of string sentences to nested list of token indices. Optionally adds padding & bos+eos tokens

        :param sentences: A list of sentences to be encoded into a batch
        :param padding: Boolean that allows for padding up to the longest sequence in the batch
        :param add_special_tokens: Boolean that allows for adding a BOS and EOS token to each sentence in the batch
        :return: nested list of tokenized sequences
        """
        tokenized_sentences = [
            self.encode(sentence, add_special_tokens) for sentence in sentences
        ]
        if padding:
            max_length = max([len(tokens) for tokens in tokenized_sentences])
            tokenized_sentences = [
                s + ((max_length - len(s)) * [self.token2index[self.PAD]])
                for s in tokenized_sentences
            ]
        return tokenized_sentences
