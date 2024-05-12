import string
import re
import numpy as np
from spacy.lang.en import English
import copy
import difflib

label_types = ['NAME_STUDENT', 'EMAIL', 'USERNAME', 'ID_NUM',
               'PHONE_NUM', 'URL_PERSONAL', 'STREET_ADDRESS']


def tokenize_with_spacy(text):
    tokenized_text = English().tokenizer(text)
    tokens = [token.text for token in tokenized_text]
    trailing_whitespace = [bool(token.whitespace_) for token in tokenized_text]
    return tokens, trailing_whitespace


def assign_name(names: list[str],
                tokens: list[str],
                labels: list[str],
                label_type: str):
    # Go through each token and assign name label ('NAME_STUDENT') if matched
    idxs = []
    new_labels = []
    for ii, token in enumerate(tokens):
        for pi_ in names:
            if pi_ == token.lower():
                if len(idxs) >= 1:
                    idx_diff = ii - idxs[-1]
                    prev_label = f'{label_type}' in new_labels[-1]
                    if (idx_diff == 1) and prev_label:
                        idxs.append(ii)
                        new_labels.append(f'I-{label_type}')
                    else:
                        idxs.append(ii)
                        new_labels.append(f'B-{label_type}')
                else:
                    idxs.append(ii)
                    new_labels.append(f'B-{label_type}')
    for idx, label in zip(idxs, new_labels):
        labels[idx] = label
    return labels


# Go through each token and assign street address label
def assign_phone_number(address, tokens, labels):
    # Back copy to clean up old labels
    old_labels = copy.deepcopy(labels)

    # Keep track of index for a long label
    label_index = 0
    reserve_indices = []
    sandwich_max_size = 1
    # print(f'address = {address}')
    for i, token in enumerate(tokens):
        try:
         # Order matters and sandwiches are possible
            token = str(token).lower()
            curr_idx = label_index
            curr_token = address[curr_idx]
            if (token in address):
                # print(f'Matched token {token}')
                # case where a token corresponds to the expected next token
                if len(reserve_indices) > sandwich_max_size:
                    reserve_indices = []
                    label_index = 0

                # prefix = 'B-' if curr_idx == 0 else 'I-'
                prev_labels = any(
                    ['PHONE_NUM' in label for label in labels[i - sandwich_max_size:i]])
                if (curr_idx != 0) and prev_labels:
                    prefix = 'I-'
                else:
                    prefix = 'B-'
                labels[i] = prefix + 'PHONE_NUM'

                # fill sandwiches if the next token has been found
                for k in reserve_indices:
                    labels[k] = 'I-PHONE_NUM'
                reserve_indices = []

                # Update positional pointer
                label_index += 1
                # At the end of index
                # if label_index == len(address) or (len(reserve_indices) >= 2
                # * len(address)):
                if label_index == len(address):
                    label_index = 0
                # print(f'label_index = {label_index}')
            elif token != curr_token and label_index > 0:
                # case where some surprise token has been added in the PII
                reserve_indices.append(i)
        except Exception as e:
            print(f"Error occurs at {i}-th {token} \n{e}")

    # Clean up False positive phone numbers
    idxs_to_check = [i for i, j in enumerate(labels) if j == 'B-PHONE_NUM']
    # rollback_idxs = [i for i in idxs_to_check \
    # if (labels[i + 1] != 'I-PHONE_NUM') and (tokens[i] in
    # string.punctuation)]
    rollback_idxs = [i for i in idxs_to_check
                     if (tokens[i] in string.punctuation) and
                     (tokens[i + 1] not in string.ascii_letters)]
    for rollback_idx in rollback_idxs:
        labels[rollback_idx] = old_labels[rollback_idx]

    # Clean up alone I-PHONE_NUM
    idxs_to_check = [i for i, j in enumerate(labels) if j == 'I-PHONE_NUM']
    idxs_to_check.reverse()
    # rollback_idxs = [i for i in idxs_to_check \
    # if (labels[i - 1] == 'B-PHONE_NUM') or (labels[i - 1] != 'I-PHONE_NUM')]
    rollback_idxs = [
        i for i in idxs_to_check if 'PHONE_NUM' not in labels[i - 1]]
    for rollback_idx in rollback_idxs:
        labels[rollback_idx] = old_labels[rollback_idx]
    return labels


def assign_single_url(text, label_type, tokens, labels):
    # Only B labels
    idxs = [j for j, t in enumerate(tokens) if text.lower() in t.lower()]
    if len(idxs) == 0:
        if 'https://' in text:
            text = text.replace('https://', '')
            idxs = [j for j, t in enumerate(
                tokens) if text.lower() in t.lower()]
    for idx in idxs:
        if labels[idx] == 'O':
            labels[idx] = f'B-{label_type}'
    return labels


def assign_other_single_label_types_mdd(text, label_type, tokens, labels):
    # Only B labels
    idxs = [j for j, t in enumerate(tokens) if text.lower() in t.lower()]
    if len(idxs) == 0:
        similar_text = difflib.get_close_matches(
            text.lower(), [i.lower() for i in tokens])
        if len(similar_text) != 0:
            similar_text = similar_text[0].lower()
            s = difflib.SequenceMatcher(None, text, similar_text)
            s.quick_ratio()
            if s.quick_ratio() > 0.9:
                idxs = [j for j, t in enumerate(
                    tokens) if similar_text in t.lower()]
            else:
                print('too low')
    for idx in idxs:
        if labels[idx] == 'O':
            labels[idx] = f'B-{label_type}'
    return labels


def assign_other_multiple_label_types_mdd(texts, label_type, tokens, labels):
    # Both B and I labels
    # First see if the pattern can be found by single character separators
    pattern = r'.'.join(texts)
    idxs = [
        j for j,
        t in enumerate(tokens) if bool(
            re.search(
                pattern,
                t.lower()))]
    if len(idxs) != 0:
        for idx in idxs:
            if labels[idx] == 'O':
                labels[idx] = f'B-{label_type}'
    else:
        if label_type == 'URL_PERSONAL':
            for j, t in enumerate(tokens):
                present = any([text in t.lower()
                              for text in texts]) and ('.com' in t.lower())
                if present:
                    if labels[j] == 'O':
                        labels[j] = f'B-{label_type}'
        else:
            print('figure out more')
    return labels


# Go through each token and assign street address label
def assign_other_label_types(label_type, text, tokens, labels):
    # Keep track of index for a long label
    label_index = 0
    reserve_indices = []
    sandwich_max_size = 2
    for i, token in enumerate(tokens):
        try:
         # Order matters and sandwiches are possible
            token = str(token).lower()
            curr_idx = label_index
            curr_token = text[curr_idx]
            if any(token in i for i in text):
                # print(f'Matched token {token}')
                # case where a token corresponds to the expected next token
                if len(reserve_indices) > sandwich_max_size:
                    reserve_indices = []

                prefix = 'B-' if curr_idx == 0 else 'I-'
                labels[i] = prefix + f'{label_type}'

                # fill sandwiches if the next token has been found
                for k in reserve_indices:
                    labels[k] = f'I-{label_type}'
                reserve_indices = []

                # Update positional pointer
                label_index += 1
                # At the end of index
                if label_index == len(text):
                    label_index = 0
                # print(f'label_index = {label_index}')
            elif token != curr_token and label_index > 0:
                # case where some surprise token has been added in the PII
                reserve_indices.append(i)
        except Exception as e:
            print(f"Error occurs at {i}-th {token} \n{e}")
    return labels


# Go through each token and assign street address label
def assign_street_address(address, tokens, labels):
    # Keep track of index for a long label
    label_index = 0
    reserve_indices = []
    sandwich_max_size = 2
    # print(f'address = {address}')
    for i, token in enumerate(tokens):
        try:
         # Order matters and sandwiches are possible
            token = str(token).lower()
            curr_idx = label_index
            curr_token = address[curr_idx]
            if (token in address):
                # print(f'Matched token {token}')
                # case where a token corresponds to the expected next token
                if len(reserve_indices) > sandwich_max_size:
                    reserve_indices = []
                    label_index = 0

                # prefix = 'B-' if curr_idx == 0 else 'I-'
                prev_labels = any(
                    ['STREET_ADDRESS' in label for label in labels[i - sandwich_max_size:i]])
                if (curr_idx != 0) and prev_labels:
                    prefix = 'I-'
                else:
                    prefix = 'B-'
                labels[i] = prefix + 'STREET_ADDRESS'

                # fill sandwiches if the next token has been found
                for k in reserve_indices:
                    labels[k] = 'I-STREET_ADDRESS'
                reserve_indices = []

                # Update positional pointer
                label_index += 1
                # At the end of index
                # if label_index == len(address) or (len(reserve_indices) >= 2
                # * len(address)):
                if label_index == len(address):
                    label_index = 0
                # print(f'label_index = {label_index}')
            elif token != curr_token and label_index > 0:
                # case where some surprise token has been added in the PII
                reserve_indices.append(i)
        except Exception as e:
            print(f"Error occurs at {i}-th {token} \n{e}")
    return labels


# “B-”: the beginning of an entity.
# “I-”: the next of an entity
def assign_labels(row, tokens):
    # Assign "O" to labels by default
    labels = ['O' for token in tokens]
    # Create a boolean flag list to track if a label type start the text.
    isFirst_flags = {label_type: True for label_type in label_types}
    # Go through each token and check if the label appear in the token
    # All token and label values are lower case for comparison
    for label_type in sorted(eval(row.fields_used)):

        # Spacy tokenize pii
        if label_type == 'STREET_ADDRESS':
            pii = row[label_type]
            pii = pii.translate(
                str.maketrans(
                    '',
                    '',
                    string.punctuation))  # Remove punctuations
            pii, _ = tokenize_with_spacy(pii)
            pii = [i.lower() for i in pii]

        else:
            pii, _ = tokenize_with_spacy(row[label_type])
            pii = [i.lower() for i in pii]

        # Select pii identification function
        if label_type == 'NAME_STUDENT':
            labels = assign_name(pii, tokens, labels, label_type)
        elif label_type == 'STREET_ADDRESS':
            labels = assign_street_address(pii, tokens, labels)
        elif label_type == 'PHONE_NUM':
            labels = assign_phone_number(pii, tokens, labels)

        else:
            if len(pii) == 2:
                if len(pii[1]) == 1:
                    pii = [pii[0] + pii[1]]
            if len(pii) == 1:
                if label_type == 'URL_PERSONAL':
                    labels = assign_single_url(
                        pii[0], label_type, tokens, labels)
                else:
                    labels = assign_other_single_label_types_mdd(
                        pii[0], label_type, tokens, labels)
            else:
                labels = assign_other_multiple_label_types_mdd(
                    pii, label_type, tokens, labels)
    return labels


# Map the label to token
def create_token_map(tokens, labels):
    token_map = []
    for i, label in enumerate(labels):
        if label != 'O':
            token_map.append({label: (tokens[i], i)})
    return token_map


# A function displays PII (like phone number) to help double-check the
# generated text
def verify_df(df):
    tmp = df.copy()
    tmp = tmp.reset_index(drop=True)
    for i in range(len(tmp)):
        row = tmp.iloc[i]
        full_text = row['full_text']
        tokens = row['tokens']
        token_map = row['token_map']
        pii_info = {}
        for field in row.pii_fields_requested.split(' '):
            found = field in row.pii_fields_identified.split(' ')
            pii_info[field] = f'Identified: {found} ::: {row[field]}'

        # Display full text and all
        print(f"======= Doc {i} =======\n")
        print(f'full_text: {full_text}')
        for t_dic in token_map:
            print(t_dic)
        print('pii_info:')
        for field, info in pii_info.items():
            print(f'\t{field}: {info}')
        print()
