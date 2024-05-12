import os
import sys
from pathlib import Path
import pandas as pd
import random
import re
from typing import List
import string
from gendata_placeholder_mistral import (split_model_response,
                                         pii_total_uniques,
                                         pii_placeholders_cleaned,
                                         tokenize_with_spacy,
                                         token_labels,
                                         inject_pii,
                                         verify_df)
random.seed(42)


if __name__ == '__main__':
    # Inputs
    save_path = Path(os.getenv('DATA_DIR')) / \
        'mdd-gen/llama3_placeholder_2.3K_v0.json'
    pii_data_path = Path(os.getenv('GEN_DIR')) / 'pii_syn_data_v4.csv'
    SPLIT_PERCENT = 1.0
    THRESHOLD = 0.70
    DOC_PREFIX = 'llama3-syn-v0'
    DEBUG = False

    # Base dir
    path_data = Path(os.getenv('GEN_DIR'))

    # Load data
    df = pd.concat([
        pd.read_csv(path_data / 'placeholder/gen_placeholder_Meta-Llama-3-8B-Instruct_N1500_151.csv',
                    encoding='UTF-8'),
        pd.read_csv(path_data / 'placeholder/gen_placeholder_Meta-Llama-3-8B-Instruct_N1500_252.csv',
                    encoding='UTF-8'),
    ],
        axis=0)

    df = df.dropna(subset=['generated_text']).reset_index(drop=True)
    # Reduce datasize for development
    if DEBUG:
        df = df.copy().iloc[0:5, :]

    # Parse LLM response from entire generated text (prompt + response)
    df['gen_response'] = df.apply(lambda x: split_model_response(x=x), axis=1)

    # Unique pii_placeholders
    df.rename(columns={'fields_used': 'fields_used_str'}, inplace=True)
    df['fields_used'] = [i.split(', ') for i in df.fields_used_str.tolist()]
    pii_placeholders = list(df['fields_used'].explode().unique())

    # Clean up messy placeholder names between curly braces
    df['full_text'] = df.apply(lambda x: pii_placeholders_cleaned(
        pii_phs=x.fields_used, text=x.gen_response), axis=1)

    # Count number of pii-placeholders inserted by LLM
    df['num_pii_fields_requested'] = df.fields_used.apply(lambda x: len(x))
    df['num_pii_fields_identified'] = df.apply(lambda x: pii_total_uniques(
        pii_phs=x.fields_used, text=x.full_text), axis=1)
    df['pii_ratio'] = df['num_pii_fields_identified'] / \
        df['num_pii_fields_requested']

    # Drop samples below a pii-ratio
    df = df[df.pii_ratio >= THRESHOLD].reset_index(drop=True)
    print(f'Num. Samples: {len(df):,}')

    # Spacy tokenizer and trailing whitespaces
    df['tokens'], df['trailing_whitespace'] = zip(
        *df.full_text.apply(tokenize_with_spacy))

    # Load PII Data
    df_pii = pd.read_csv(pii_data_path)
    df_pii.rename(columns={'NAME': 'YOUR_NAME',
                           'ID_NUM': 'IDENTIFICATION_NUM'},
                  inplace=True)
    df_pii = df_pii[pii_placeholders]

    # Insert PII into Full Text
    df_final = None
    for ii in range(len(df)):
        gen, pii = df.iloc[[ii]], df_pii.iloc[ii]
        gen = gen.reset_index(drop=True)
        gen_explode = gen.copy().explode(
            ['tokens', 'trailing_whitespace']).reset_index(drop=True)
        # Incorporate PII into placeholders
        gen_pii = inject_pii(
            row=gen_explode,
            pii=pii,
            pii_placeholders=pii_placeholders)

        # Apply competition label names
        gen_pii['label'] = gen_pii.label.str.replace(
            '-YOUR_NAME', '-NAME_STUDENT')
        gen_pii['label'] = gen_pii.label.str.replace(
            '-IDENTIFICATION_NUM', '-ID_NUM')

        # Create full text with pii filled-in
        text = []
        for t, ws in zip(gen_pii["tokens"], gen_pii["trailing_whitespace"]):
            text.append(t)
            if ws:
                text.append(" ")
        text = ''.join(text)

        # Aggregate results
        cols_agg = ['tokens', 'trailing_whitespace', 'label']
        tmp = (gen_pii.groupby('file_name')
               .agg({"tokens": lambda x: x.tolist(),
                     "trailing_whitespace": lambda x: x.tolist(),
                    "label": lambda x: x.tolist()})
               .reset_index())

        # Assign new full text
        tmp['full_text'] = text

        # Combine results
        cols = list(set(gen.columns) - set(tmp.columns))
        new_gen = pd.concat([gen[cols], tmp], axis=1)

        # Concatenate into final dataframe
        if df_final is None:
            df_final = new_gen
        else:
            df_final = pd.concat([df_final, new_gen],
                                 axis=0).reset_index(drop=True)
        if ii % 50 == 0:
            print(f'Completed {ii} of {len(df):,}')

    # Document ID
    df_final['document'] = [DOC_PREFIX + f'_{i}' for i in range(len(df_final))]

    # Reduce to only required columns
    df_final.rename(columns={'label': 'labels'}, inplace=True)

    # View results
    if DEBUG:
        verify_df(df=df_final.copy())
    print(f'df_final.shape: {df_final.shape}')
    df_final = df_final[['document', 'full_text',
                         'tokens', 'trailing_whitespace', 'labels']]
    print(f'df_final.shape: {df_final.shape}')

    # Save to disk
    df_final.to_json(save_path)
    print(f'Saved at:\n{save_path}')
print('End of Script - Completed')
