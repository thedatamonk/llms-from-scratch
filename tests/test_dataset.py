from src.data.vocab import Vocabulary
from src.data.dataloader import construct_future_mask, construct_batches
import torch
import unittest

class TestVocabulary(unittest.TestCase):
    maxDiff = None

    def test_tokenize(self):
        input_sequence = "Hello my name is Joris and I was born with the name Joris."
        output = Vocabulary([]).tokenize(input_sequence)
        self.assertEqual(
            [
                "BOS",
                "Hello",
                "my",
                "name",
                "is",
                "Joris",
                "and",
                "I",
                "was",
                "born",
                "with",
                "the",
                "name",
                "Joris",
                ".",
                "EOS",
            ],
            output,
        )

    def test_init_vocab(self):
        input_sentences = ["Hello my name is Joris and I was born with the name Joris."]
        vocab = Vocabulary(input_sentences)
        expected = {
            "BOS": 0,
            "EOS": 1,
            "PAD": 2,
            "Hello": 3,
            "my": 4,
            "name": 5,
            "is": 6,
            "Joris": 7,
            "and": 8,
            "I": 9,
            "was": 10,
            "born": 11,
            "with": 12,
            "the": 13,
            ".": 14,
        }
        self.assertEqual(vocab.token2index, expected)

    def test_encode(self):
        input_sentences = ["Hello my name is Joris and I was born with the name Joris."]
        vocab = Vocabulary(input_sentences)
        output = vocab.encode(input_sentences[0])
        self.assertEqual(output, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5, 7, 14, 1])

    def test_encode_no_special_tokens(self):
        input_sentences = ["Hello my name is Joris and I was born with the name Joris."]
        vocab = Vocabulary(input_sentences)
        output = vocab.encode(input_sentences[0], add_special_tokens=False)
        self.assertEqual(output, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5, 7, 14])

    def test_batch_encode(self):
        input_sentences = [
            "This is one sentence",
            "This is another, much longer sentence",
            "Short sentence",
        ]
        vocab = Vocabulary(input_sentences)
        output = vocab.batch_encode(input_sentences, add_special_tokens=False)
        self.assertEqual(
            output,
            [[3, 4, 5, 6, 2, 2, 2], [3, 4, 7, 8, 9, 10, 6], [11, 6, 2, 2, 2, 2, 2]],
        )



class TestUtils(unittest.TestCase):
    def test_construct_future_mask(self):
        mask = construct_future_mask(3)
        torch.testing.assert_close(
            mask,
            torch.BoolTensor(
                [[True, False, False], [True, True, False], [True, True, True]]
            ),
        )

    def test_construct_future_mask_first_decoding_step(self):
        mask = construct_future_mask(1)
        torch.testing.assert_close(
            mask, torch.BoolTensor([[True]]),
        )

    def test_construct_batches(self):
        corpus = [
            {"en": "This is an english sentence.", "nl": "Dit is een Nederlandse zin."},
            {"en": "The weather is nice today.", "nl": "Het is lekker weer vandaag."},
            {
                "en": "Yesterday I drove to a city called Amsterdam in my brand new car.",
                "nl": "Ik reed gisteren in mijn gloednieuwe auto naar Amsterdam.",
            },
            {
                "en": "You can pick up your laptop at noon tomorrow.",
                "nl": "Je kunt je laptop morgenmiddag komen ophalen.",
            },
        ]
        en_sentences, nl_sentences = (
            [d["en"] for d in corpus],
            [d["nl"] for d in corpus],
        )
        vocab = Vocabulary(en_sentences + nl_sentences)
        batches, masks = construct_batches(
            corpus, vocab, batch_size=2, src_lang_key="en", tgt_lang_key="nl"
        )
        torch.testing.assert_close(
            batches["src"],
            [
                torch.IntTensor(
                    [[0, 3, 4, 5, 6, 7, 8, 1], [0, 9, 10, 4, 11, 12, 8, 1]]
                ),
                torch.IntTensor(
                    [
                        [0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 8, 1],
                        [0, 26, 27, 28, 29, 30, 31, 32, 33, 34, 8, 1, 2, 2, 2, 2],
                    ]
                ),
            ],
        )