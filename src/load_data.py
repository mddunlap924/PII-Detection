import os
import gc
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from typing import List, Tuple, Union, Type, TypeVar
import random
import regex as re

T = TypeVar('T')


def load_individual(path_file: str,
                    *,
                    cols: List[str] = ["document", "full_text", "tokens",
                                       "trailing_whitespace", "labels"]) -> List:
    # Load the file
    data = json.load(open(path_file))
    assert list(data[0].keys()) == cols, f'Wrong Columns for: {path_file}'
    return data


def count_tokens(text: str, tokenizer: T) -> int:
    tokenized = tokenizer(text, padding=False, truncation=False)
    text_len = len(tokenized['input_ids'])
    return text_len


# Load Data
class LoadData:
    """
    Load JSON Data Files
    (Expand this class to other datasets suitable for your needs)
    """

    def __init__(self,
                 data_dir: str,
                 train_files: List[str],
                 val_file: str,
                 path_tokenizer: str,
                 *,
                 split: bool = False,
                 max_token_length: Union[bool, int] = None) -> None:
        """
        :param base_dir: Directory data files are stored
        """
        self.data_dir = Path(data_dir)
        self.train_files = train_files
        self.val_file = val_file
        self.path_tokenizer = path_tokenizer
        self.split = split
        self.max_token_length = max_token_length
        self.cols = ["document", "full_text", "tokens",
                     "trailing_whitespace", "labels"]

    def load(self, *,
             explode: bool = True,
             ds_ratio: float = 0.3,
             mask_data: bool = False,
             mask_prob: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # Validation dataset
        val = pd.read_json(self.data_dir / self.val_file)
        val['source'] = self.val_file.split('/')[-1].split('.')[0]

        # Train Dataset
        train = None
        for file in self.train_files:
            # Load individual file
            tmp = pd.read_json(self.data_dir / file)

            # Source of data
            tmp['source'] = file.split('/')[-1].split('.')[0]

            # Concatenate results
            if train is not None:
                train = pd.concat([train, tmp], ignore_index=True, axis=0)
            else:
                train = tmp
        assert all(train.columns.isin(val.columns)
                   ), 'Train and Val Columns Mismatch'

        # Count tokens
        train['num_spacy_tokens'] = train['tokens'].apply(lambda x: len(x))
        val['num_spacy_tokens'] = val['tokens'].apply(lambda x: len(x))

        # Explode data
        if explode:
            train = (train
                     .explode(["tokens", "trailing_whitespace", "labels"])
                     .reset_index(drop=True))
            val = (val
                   .explode(["tokens", "trailing_whitespace", "labels"])
                   .reset_index(drop=True))
            val['document'] = val['document'].astype(str)
        return train, val
