import numpy as np
from typing import Type, TypeVar, List, Dict, Tuple
from datasets import Dataset, features
from itertools import chain
import pandas as pd
import unicodedata
from text_unidecode import unidecode
import codecs
import re


T = TypeVar('T')


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start: error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start: error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error(
    "replace_decoding_with_cp1252",
    replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


# This be a function called 'tokenize', me hearties!
def tokenize_single(example, tokenizer, max_length, label2id):
    # We be creatin' two empty lists, 'text' and 'token_map'
    text = []
    token_map = []
    labels = []
    for idx, (t, l, ws) in enumerate(zip(example["tokens"],
                                         example['provided_labels'],
                                         example["trailing_whitespace"])):
        # t = resolve_encodings_and_normalize(text=t)
        # t= re.sub(r' {2,}', ' ', t)
        text.append(t)
        labels.extend([l] * len(t))
        token_map.extend([idx]*len(t))
        if ws:
            text.append(" ")
            labels.append('O')
            token_map.append(-1)

    labels = np.array(labels)
    text = "".join(text)

    # Now, we tokenize the concatenated 'text' and return offsets mappings
    # along with 'token_map'.
    tokenized = tokenizer(text,
                          return_offsets_mapping=True,
                          truncation=True,
                          max_length=max_length)

    # Number of tokens
    length = len(tokenized.input_ids)

    # Assign labels to each token
    token_labels = []
    for start_idx, end_idx in tokenized.offset_mapping:
        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue
        # case when token starts with whitespace
        if text[start_idx].isspace():
            if end_idx != len(text):
                start_idx += 1
        # Assign token label
        token_labels.append(label2id[labels[start_idx]])

    # We return a dictionary containin' the tokenized data and the 'token_map'.
    return {**tokenized,
            'token_map': token_map,
            'labels': token_labels,
            'length': length}


def train_val_dataset(tokenizer: Type[T],
                      label2id: dict,
                      df: pd.DataFrame,
                      max_length: int,
                      *,
                      num_proc: int = 8) -> Type[T]:
    # Prep dataframe
    df['document'] = df['document'].astype(str)

    # Create a dataset from pandas dataframe
    ds = Dataset.from_dict({
        "full_text": df["full_text"].tolist(),
        "document": df["document"].tolist(),
        "tokens": df["tokens"].tolist(),
        "trailing_whitespace": df["trailing_whitespace"].tolist(),
        "provided_labels": df["labels"].tolist(),
    })

    # Tokenize data
    ds = ds.map(tokenize_single,
                fn_kwargs={"tokenizer": tokenizer,
                           "max_length": max_length,
                           "label2id": label2id},
                num_proc=num_proc,
                # remove_columns=ds.column_names,
                )

    return ds


def tokenize(example, tokenizer, label2id, max_length, stride):
    # print(example)
    # rebuild text from tokens
    text = []
    labels = []
    token_map = []
    idx = 0
    for t, l, ws in zip(example["tokens"],
                        example["provided_labels"],
                        example["trailing_whitespace"]):
        text.append(t)
        labels.extend([l] * len(t))
        token_map.extend([idx] * len(t))
        if ws:
            text.append(" ")
            labels.append("O")
            token_map.append(-1)
        idx += 1

    # actual tokenization
    tk = tokenizer("".join(text),
                   return_offsets_mapping=True,
                   max_length=max_length,
                   stride=stride,
                   truncation=True,
                   return_overflowing_tokens=True,
                   return_length=True,
                   padding=False,
                   #    padding='max_length',
                   )
    labels = np.array(labels)
    text = "".join(text)
    num_splits = len(tk['length'])

    token_labels = []
    for input_ids, offset_map in zip(tk.input_ids, tk.offset_mapping):
        assert len(input_ids) == len(offset_map)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        token_labels_ = []
        for ii, (start_idx, end_idx) in enumerate(offset_map):
            token = tokens[ii]
            # CLS token
            if start_idx == 0 and end_idx == 0:
                token_labels_.append(label2id["O"])
                continue
            # case when token starts with whitespace
            if text[start_idx].isspace() and (start_idx < len(text) - 1):
                start_idx += 1
            token_labels_.append(label2id[labels[start_idx]])
        token_labels.append(token_labels_)
    docs = [example['document'] for _ in range(num_splits)]
    srcs = [example['source'] for _ in range(num_splits)]
    full_text = [example['full_text'] for _ in range(num_splits)]
    text = [text for _ in range(num_splits)]
    token_maps = [token_map for _ in range(num_splits)]
    tokens = [example['tokens'] for _ in range(num_splits)]
    return {**tk,
            'document': docs,
            'source': srcs,
            "labels": token_labels,
            'token_map': token_maps,
            'tokens': tokens,
            'full_text': full_text,
            'text': text}


def train_dataset(tokenizer: Type[T],
                  label2id: dict,
                  data: List[Dict],
                  max_length: int,
                  stride: int,
                  *,
                  num_proc: int = 8) -> Type[T]:
    data['document'] = data['document'].astype(str)
    data.rename(columns={'labels': 'provided_labels'}, inplace=True)
    ds = Dataset.from_pandas(data)
    ds = ds.map(tokenize,
                fn_kwargs={"tokenizer": tokenizer,
                           "label2id": label2id,
                           "max_length": max_length,
                           "stride": stride},
                num_proc=num_proc,
                remove_columns=ds.column_names,
                # batched=True,
                # with_indices=True,
                )

    def flatten_list_of_dict(batch):
        results = {}
        for key in batch.keys():
            results[key] = list(chain(*batch[key]))
        return {**results}

    ds = ds.map(
        flatten_list_of_dict,
        batched=True,
        remove_columns=ds.column_names)

    return ds


#######################################
# Masked Language Modeling
#######################################

def mlm_tokenizer(example, tokenizer):
    text = []
    for idx, (t, ws) in enumerate(zip(example["tokens"],
                                      example["trailing_whitespace"])):
        text.append(t)
        if ws:
            text.append(" ")

    # Tokenize text
    tk = tokenizer("".join(text), return_length=True)
    return tk


def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
