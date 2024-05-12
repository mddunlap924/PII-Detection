import string
import pandas as pd
import multiprocessing as mp
from time import time
from copy import deepcopy
import gc
from typing import List, Dict


def retokenize_punctuation(df: pd.DataFrame) -> pd.DataFrame:
    """Strips punctuation if it is the last letter of a token and
    puts it as new token instead. This way, the formatting is (more)
    in line with the original training formatting, and punctuation after
    relevant tokens is not misstrained.

    Args:
        df: the exploded pii_dataset dataframe

    Returns:
        df - same dataframe but last letter is a new row if its punctuation.
    """
    pii_dataset_as_list = []
    # Loop over whole dataframe to check for punctuation mistake
    for i, row in df.iterrows():
        if row["tokens"][-1] in string.punctuation:
            # append token without punctuation
            pii_dataset_as_list.append(
                [
                    row["document"],
                    row["text"],
                    row["tokens"][:-1],
                    False,
                    row["labels"]
                ]
            )

            # separately append punctuation
            pii_dataset_as_list.append(
                [
                    row["document"],
                    row["text"],
                    row["tokens"][-1],
                    True,
                    "O"
                ]
            )
        else:
            # append unchanged row
            pii_dataset_as_list.append(
                [
                    row["document"],
                    row["text"],
                    row["tokens"],
                    row["trailing_whitespace"],
                    row["labels"]
                ]
            )
    # recreate dataframe from tokens
    fixed = pd.DataFrame(pii_dataset_as_list, columns=df.columns)
    # Removed empty spaces (that are somehow being created)
    fixed = fixed[fixed.tokens != ""]
    return fixed


def retokenize_punctuation_multiprocess(row: dict) -> list:
    """Strips punctuation if it is the last letter of a token and
    puts it as new token instead. This way, the formatting is (more)
    in line with the original training formatting, and punctuation after
    relevant tokens is not misstrained.

    Args:
        df: the exploded pii_dataset dataframe

    Returns:
        df - same dataframe but last letter is a new row if its punctuation.
    """
    pii_dataset_as_list = []
    # Loop over whole dataframe to check for punctuation mistake
    if row["tokens"][-1] in string.punctuation:
        # append token without punctuation
        pii_dataset_as_list.append(
            [
                row["document"],
                row["text"],
                row["tokens"][:-1],
                False,
                row["labels"],
            ]
        )

        # separately append punctuation
        pii_dataset_as_list.append(
            [
                row["document"],
                row["text"],
                row["tokens"][-1],
                True,
                "O",
            ]
        )
    else:
        # append unchanged row
        pii_dataset_as_list.append(
            [
                row["document"],
                row["text"],
                row["tokens"],
                row["trailing_whitespace"],
                row["labels"],
            ]
        )
    return pii_dataset_as_list


def flatten_records(vec: list) -> list:
    flat_results = []
    for r in vec:
        if len(r) > 1:
            flat_results.extend(r)
        else:
            flat_results.append(r[0])
    return flat_results


def multiprocess_tokens(records: List[Dict], *, N: int = 3) -> pd.DataFrame:
    tmp = None
    pool = mp.Pool(processes=8)
    for ii in range(0, N):
        st_loop = time()
        if tmp is None:
            results = pool.map(retokenize_punctuation_multiprocess, records)
        else:
            results = pool.map(
                retokenize_punctuation_multiprocess,
                tmp.to_dict(
                    orient='records'))
        results = flatten_records(vec=results)
        tmp = pd.DataFrame(results,
                           columns=['document', 'text', 'tokens',
                                    'trailing_whitespace', 'labels'])
        tmp['trailing_whitespace'] = tmp['trailing_whitespace'].astype(
            'object')
        tmp = tmp.drop(tmp[tmp['tokens'] == ''].index).reset_index(
            drop=True).copy()
        print(
            f'Completed Loop {ii + 1} of {len(range(1, 4))}: {(time() - st_loop) / 60:.2} [min]')
    pool.close()
    return tmp


def find_indices(lst, element):
    return [index for index, value in enumerate(lst) if value == element]


def convert_to_tokenlevel(para, label):
    labs = eval(label)
    tok = para.split()
    labb = ['O']*len(tok)
    ws = [True]*len(tok)
    ws[-1] = False
    for i in (labs):
        for j in labs[i]:
            k = j.split()
            b_flag = True
            for m in k:
                if b_flag:
                    indices = find_indices(tok, m)
                    for ind in indices:
                        labb[ind] = 'B-'+i
                    b_flag = False
                else:
                    indices = find_indices(tok, m)
                    for ind in indices:
                        labb[ind] = 'I-'+i
    return {
        "tokens": tok,
        "labels": labb,
        "trailing_whitespace": ws
    }
