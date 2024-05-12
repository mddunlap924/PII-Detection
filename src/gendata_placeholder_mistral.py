import string
import re
import numpy as np
from spacy.lang.en import English
from typing import List
import re
import copy
import difflib
import pandas as pd
import random
import sys


def tokenize_with_spacy(text):
    tokenized_text = English().tokenizer(text)
    tokens = [token.text for token in tokenized_text]
    trailing_whitespace = [bool(token.whitespace_) for token in tokenized_text]
    return tokens, trailing_whitespace


def pii_total_uniques(pii_phs: List[str], text: str):
    count = 0
    for pii_ph in pii_phs:
        if pii_ph in text:
            count += 1
    return count


def pii_placeholders_cleaned(pii_phs: List[str], text: str):
    text_org = text
    text = text.strip()

    # First apply a simple replacement for each pii_phs
    for pii_ph in pii_phs + ['ID_NUM']:
        if pii_ph == 'ID_NUM':
            replace = 'IDENTIFICATION_NUM'
        else:
            replace = pii_ph
        text = text.replace('{' + pii_ph + '}', replace)

    if text[0] == '{':
        text = text[1:]

    # Regex replacement
    patterns, replacements = [], []
    for pii_ph in pii_phs:
        num_splits = len(pii_ph.split('_'))
        if (pii_ph != 'STREET_ADDRESS') and (pii_ph != 'YOUR_NAME'):
            pattern = r"\{[^{}]*(?i:PII0)[^{}]*\}"
            pattern = pattern.replace('PII0', pii_ph.split('_')[0])
            repl = pii_ph
        elif pii_ph == 'YOUR_NAME':
            pattern = r"\{[^{}]*(?i:PII0)[^{}]*\}"
            pattern = pattern.replace('PII0', pii_ph.split('_')[1])
            repl = pii_ph
        elif pii_ph == 'STREET_ADDRESS':
            pattern = r"\{[^{}]*(?i:PII0|PII1)[^{}]*\}"
            # pattern = r"\{(?=.*?(?i:PII0))(?=.*?(?i:PII1))[^{}]*\}"
            pattern = pattern.replace('PII0', pii_ph.split('_')[0])
            pattern = pattern.replace('PII1', pii_ph.split('_')[-1])
            repl = pii_ph
        patterns.append(pattern)
        replacements.append(repl)

    # Compile the patterns
    compiled_patterns = [re.compile(pattern, re.IGNORECASE)
                         for pattern in patterns]

    # Apply the patterns
    for pattern, replacement in zip(compiled_patterns, replacements):
        text = pattern.sub(replacement, text)

    # Regex pattern to match curly braces and text inside them
    pattern = r"\{[^{}]*\}"
    text = re.sub(pattern, "", text)

    # Remove title from text
    if text[0:7] == 'Title: ':
        text = text[7:]

    if 'YOUR_NAME' not in text:
        if random.random() >= 0.5:
            text = '\n' + 'YOUR_NAME' + '\n' + text
        else:
            text = text + '\n\n' + 'YOUR_NAME'
    return text


# Split model response
def split_model_response(x):
    if ('mistral' in x.model.lower()) or ('mixtral' in x.model.lower()):
        gen_text = x.generated_text.split('[/INST]')[-1]
    elif 'llama-3' in x.model.lower():
        gen_text = x.generated_text.split('>assistant<|end_header_id|>')[-1]
    else:
        sys.exit()

    # For llama3 cut off Note: The placeholders
    if 'llama-3' in x.model.lower():
        if 'Note: The place' in gen_text:
            gen_text = gen_text.split('Note: The place')[0]
    return gen_text


def token_labels(token: str, pii_phs: List[str]):
    # Identify Token Labels
    token_label = None
    for pii_ph in pii_phs:
        if pii_ph in token:
            token_label = pii_ph
    # Convert Nones to O
    if token_label is None:
        token_label = 'O'
    return token_label


def inject_pii(row: pd.DataFrame, pii: pd.Series, pii_placeholders: List[str]):
    # Identify Token Labels
    row['label'] = row.apply(lambda x: token_labels(token=x.tokens,
                                                    pii_phs=pii_placeholders),
                             axis=1)
    row['label_org'] = row.label.tolist()
    used_phs = [i for i in row.label.unique().tolist() if i != 'O']
    # Go over each pii-placeholder
    for used_ph in used_phs:

        # Tokenize pii-details
        pii_tokens_original, pii_tws_original = tokenize_with_spacy(
            text=pii[used_ph])

        idxs = row[row.label == used_ph].index.tolist()
        row['idx_org'] = row.index
        if len(idxs) > 0:
            for idx in idxs:

                # Randomly split PII into shorter segments
                prob_split = random.randrange(0, 100)

                if len(pii_tokens_original) == 2 and prob_split >= 99:
                    pii_tokens, pii_tws = [
                        pii_tokens_original[-1]], [pii_tws_original[-1]]
                elif len(pii_tokens_original) > 2 and prob_split >= 99:
                    possible_tokens = []
                    pii_tokens_red = pii_tokens_original[1:]
                    for pii_token in pii_tokens_red:
                        check1 = len(pii_token) >= 3
                        check2 = pii_tokens_red.index(
                            pii_token) / len(pii_tokens_red) <= 0.75
                        check3 = len(
                            pii_tokens_red[pii_tokens_red.index(pii_token):]) > 1
                        if check1 and check2 and check3:
                            possible_tokens.append(
                                pii_tokens_red.index(pii_token))
                    if len(possible_tokens) > 0:
                        pii_tokens = pii_tokens_red[possible_tokens[0]:]
                        pii_tws = pii_tws_original[1:][possible_tokens[0]:]
                    else:
                        pii_tokens, pii_tws = pii_tokens_original, pii_tws_original
                else:
                    pii_tokens, pii_tws = pii_tokens_original, pii_tws_original

                # Assign B- and I- labels
                pii_labels = [
                    f'B-{used_ph}'] + [f'I-{used_ph}' for jj in range(len(pii_tokens) - 1)]

                idx_ = row[row['idx_org'] == idx].index[0]
                pii_tws[-1] = row[row['idx_org'] ==
                                  idx].trailing_whitespace.iloc[0]
                tmp = pd.DataFrame({'tokens': pii_tokens,
                                    'trailing_whitespace': pii_tws,
                                    'label': pii_labels})
                del pii_tokens, pii_tws, pii_labels
                missing_cols = list(set(row.columns) - set(tmp.columns))
                sub = row[missing_cols].loc[0:len(tmp) - 1]
                sub['idx_org'] = -100
                tmp = pd.concat([tmp, sub], axis=1)
                row = pd.concat(
                    [row.iloc[:idx_], tmp, row.iloc[idx_:]], axis=0)
                row = row.drop(idx_).reset_index(drop=True)
    return row


# A function displays PII (like phone number) to help double-check the
# generated text
def verify_df(df):

    for i in range(len(df)):
        row = df.iloc[[i]]
        row = row.explode(['tokens', 'trailing_whitespace',
                          'labels']).reset_index(drop=True)
        gen_text = row['gen_response'].iloc[0]
        full_text = row['full_text'].iloc[0]

        text = []
        for t, ws in zip(row["tokens"],
                         row["trailing_whitespace"]):
            text.append(t)
            if ws:
                text.append(" ")
        text = ''.join(text)

        pii_found = row[row.labels != 'O'][['tokens', 'labels']]
        fields_requested = sorted(row.iloc[0].fields_used)
        fields_requested = [
            i.replace(
                'IDENTIFICATION_NUM',
                'ID_NUM') for i in fields_requested]
        fields_requested = [
            i.replace(
                'YOUR_NAME',
                'NAME_STUDENT') for i in fields_requested]
        fields_requested = sorted(fields_requested)
        fields_found = sorted(
            np.unique([i.split('-')[-1] for i in pii_found['labels']]))
        missing_fields = sorted(set(fields_requested) - set(fields_found))
        # Display full text and all
        print(f"======= Doc {i} =======\n")
        print(f'GENERATED TEXT:\n{gen_text}')
        print()
        # print(f'FULL TEXT:\n{full_text}')
        # print()
        print(f'FINAL TEXT:\n{text}')
        print()
        print(f'Fields Requested: {fields_requested}')
        print(f'Fields Found: {fields_found}')
        print(f'Fields Missing: {missing_fields}')
        print(f'PII:')
        for idx, r in pii_found.iterrows():
            print(f'{idx}: {r.labels} -> {r.tokens}')
        print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
