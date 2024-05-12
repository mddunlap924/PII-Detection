# Setup Env. Variables
import gc
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'True'

# Do NOT log models to WandB
os.environ["WANDB_LOG_MODEL"] = "false"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"


from transformers import (AutoModelForMaskedLM,
                          AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
from transformers import AutoTokenizer, Trainer, TrainingArguments
from tokenizers import AddedToken
from datasets import Dataset
from pathlib import Path
import argparse
import math


# Custom (cx) modules
from utils import (load_cfg,
                   debugger_is_active,
                   seed_everything)
from load_data import LoadData
from create_datasets import mlm_tokenizer, group_texts


ALL_LABELS = ['B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM',
              'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME',
              'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM',
              'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O']

if __name__ == '__main__':

    # Determine if running in debug mode
    # If in debug manually point to CFG file
    is_debugger = debugger_is_active()

    # Construct the argument parser and parse the arguments
    if is_debugger:
        args = argparse.Namespace()
        args.dir = os.getenv('BASE_DIR') + '/cfgs/mlm'
        args.name = 'cfg1.yaml'
    else:
        arg_desc = '''This program points to input parameters for model training'''
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=arg_desc)
        parser.add_argument("-cfg_dir",
                            "--dir",
                            required=True,
                            help="Base Dir. for the YAML config. file")
        parser.add_argument("-cfg_filename",
                            "--name",
                            required=True,
                            help="File name of YAML config. file")
        args = parser.parse_args()
        print(args)

    # Load the configuration file
    CFG = load_cfg(base_dir=Path(args.dir),
                   filename=args.name)
    CFG.paths.base_dir = os.getenv('BASE_DIR')
    CFG.paths.data_dir = os.getenv('DATA_DIR')
    CFG.paths.save_dir = os.getenv('MODEL_DIR')

    # Seed everything
    seed_everything(seed=CFG.seed)

    # Load data
    df_train, df_val = (LoadData(data_dir=CFG.paths.data_dir,
                                 train_files=CFG.paths.data.train,
                                 val_file=CFG.paths.data.val,
                                 path_tokenizer=str(
                                     Path(os.getenv('MODEL_DIR')) / CFG.model.name),
                                 split=False,
                                 max_token_length=CFG.tokenizer.max_token_length)
                        .load(explode=False))

    # Get labels
    data = df_train.to_dict(orient='records') + \
        df_val.to_dict(orient='records')
    # ALL_LABELS = sorted(list(set(chain(*[x["labels"] for x in data]))))
    label2id = {l: i for i, l in enumerate(ALL_LABELS)}
    id2label = {v: k for k, v in label2id.items()}
    del data
    _ = gc.collect()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(os.getenv('MODEL_DIR')) / CFG.model.name),
        use_fast=CFG.tokenizer.use_fast,
        do_lower_case=CFG.tokenizer.do_lower)
    # Add tokens
    if CFG.tokenizer.add_tokens is not None:
        for token_to_add in CFG.tokenizer.add_tokens:
            token_to_add = bytes(
                token_to_add, 'utf-8').decode('unicode_escape')
            tokenizer.add_tokens(AddedToken(token_to_add, normalized=False))

    # Debug smaller dataset
    if CFG.debug:
        df_train = df_train.sample(
            n=500, random_state=42).reset_index(
            drop=True)
        df_val = df_val.sample(n=250, random_state=42).reset_index(drop=True)

    # Training dataset
    cols = ['tokens', 'trailing_whitespace', 'full_text']
    ds_train = Dataset.from_pandas(df_train[cols])
    ds_train = ds_train.map(mlm_tokenizer,
                            fn_kwargs={'tokenizer': tokenizer},
                            num_proc=8,
                            remove_columns=ds_train.column_names)
    total_len = sum([i[0] for i in ds_train['length']])
    print(f'1st Stage Total Len: {total_len:,}')

    # Concatenate texts and split into blocks of text
    ds_train = ds_train.map(
        group_texts,
        fn_kwargs={'block_size': CFG.block_size},
        batched=True,
        # batch_size=1,
        num_proc=8,
    )
    total_len = sum([len(i) for i in ds_train['input_ids']])
    max_len = max([len(i) for i in ds_train['input_ids']])
    print('TRAIN:')
    print(f'2nd Stage Total Len: {total_len:,}')
    print(f'Max Len.: {max_len:,}')
    print(f'Num. Rows: {ds_train.num_rows:,}')
    # Validation dataset
    ds_val = Dataset.from_pandas(df_val[cols])
    ds_val = ds_val.map(mlm_tokenizer,
                        fn_kwargs={'tokenizer': tokenizer},
                        num_proc=8,
                        remove_columns=ds_val.column_names)

    # Concatenate texts and split into blocks of text
    ds_val = ds_val.map(
        group_texts,
        fn_kwargs={'block_size': CFG.block_size},
        batched=True,
        num_proc=8,
    )
    total_len = sum([len(i) for i in ds_val['input_ids']])
    max_len = max([len(i) for i in ds_val['input_ids']])
    print('VAL.:')
    print(f'2nd Stage Total Len: {total_len:,}')
    print(f'Max Len.: {max_len:,}')
    print(f'Num. Rows: {ds_train.num_rows:,}')

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=CFG.mlm_probability)

    # MLM Model
    model = AutoModelForMaskedLM.from_pretrained(
        str(Path(CFG.paths.save_dir) / CFG.model.name))

    # Training Arguments
    output_dir = str(Path(CFG.paths.save_dir) / 'mlm-tuned')
    eval_steps = 300
    print(output_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
        fp16=CFG.train_args.fp16,
        overwrite_output_dir=True,
        num_train_epochs=CFG.train_args.num_train_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy='steps',
        save_steps=eval_steps,
        eval_steps=eval_steps,
        save_total_limit=2,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        load_best_model_at_end=True,
        prediction_loss_only=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds_train,
        eval_dataset=ds_val)

    trainer.train()
    print(output_dir)

    trainer.save_model(output_dir=output_dir + '/final4')
    tokenizer.save_pretrained(save_directory=output_dir + '/final4')

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    print('End of Training')
