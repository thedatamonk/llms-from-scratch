import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import unittest
from typing import Dict, List, Tuple, Optional
from src.data.vocab import Vocabulary


def create_synthetic_data(num_samples=10, max_seq_len=5):
    """
    Generates a simple synthetic dataset for a sequence-to-sequence task.
    The task is to learn the identity mapping of sequences.
    """
    data = []
    # Vocabulary mapping
    # Note: Pad and EOS tokens are often required for real-world tasks.
    # We will use simple integer tokens here.
    vocab = {i: i for i in range(100)} # A simple mapping for our tokens

    for i in range(num_samples):
        # Create a random source sequence
        source_seq_len = torch.randint(1, max_seq_len + 1, (1,)).item()
        source_seq = torch.randint(0, 50, (source_seq_len,)).tolist()

        # The target sequence is the same as the source sequence for simplicity
        target_seq = source_seq

        data.append((source_seq, target_seq))
    return data, vocab

class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_seq, target_seq = self.data[idx]
        return torch.tensor(source_seq), torch.tensor(target_seq)

def custom_collate_fn(batch):
    """
    Pads the sequences in a batch to the same length.
    """
    source_sequences, target_sequences = zip(*batch)

    # Pad sequences to the length of the longest sequence in the batch
    # This is necessary because transformer models require fixed-size inputs
    source_padded = pad_sequence(source_sequences, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_sequences, batch_first=True, padding_value=0)

    return source_padded, target_padded


def construct_future_mask(seq_len: int):
    """
    Construct a binary mask that contains 1's for all valid connections and 0's for all outgoing future connections.
    This mask will be applied to the attention logits in decoder self-attention such that all logits with a 0 mask
    are set to -inf.

    :param seq_len: length of the input sequence
    :return: (seq_len,seq_len) mask
    """
    subsequent_mask = torch.triu(torch.full((seq_len, seq_len), 1), diagonal=1)
    return subsequent_mask == 0


def construct_batches(
    corpus: List[Dict[str, str]],
    vocab: Vocabulary,
    batch_size: int,
    src_lang_key: str,
    tgt_lang_key: str,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
    """
    Constructs batches given a corpus.

    :param corpus: The input corpus is a list of aligned source and target sequences, packed in a dictionary.
    :param vocab: The vocabulary object.
    :param batch_size: The number of sequences in a batch
    :param src_lang_key: The source language key is a string that the source sequences are keyed under. E.g. "en"
    :param tgt_lang_key: The target language key is a string that the target sequences are keyed under. E.g. "nl"
    :param device: whether or not to move tensors to gpu
    :return: A tuple containing two dictionaries. The first represents the batches, the second the attention masks.
    """
    pad_token_id = vocab.token2index[vocab.PAD]
    batches: Dict[str, List] = {"src": [], "tgt": []}
    masks: Dict[str, List] = {"src": [], "tgt": []}
    for i in range(0, len(corpus), batch_size):
        src_batch = torch.IntTensor(
            vocab.batch_encode(
                [pair[src_lang_key] for pair in corpus[i : i + batch_size]],
                add_special_tokens=True,
                padding=True,
            )
        )
        tgt_batch = torch.IntTensor(
            vocab.batch_encode(
                [pair[tgt_lang_key] for pair in corpus[i : i + batch_size]],
                add_special_tokens=True,
                padding=True,
            )
        )

        src_padding_mask = src_batch != pad_token_id
        future_mask = construct_future_mask(tgt_batch.shape[-1])

        # Move tensors to gpu; if available
        if device is not None:
            src_batch = src_batch.to(device)  # type: ignore
            tgt_batch = tgt_batch.to(device)  # type: ignore
            src_padding_mask = src_padding_mask.to(device)
            future_mask = future_mask.to(device)
        batches["src"].append(src_batch)
        batches["tgt"].append(tgt_batch)
        masks["src"].append(src_padding_mask)
        masks["tgt"].append(future_mask)
    return batches, masks

