# Setup Env. Variables
import gc
import sys
import os

os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['TOKENIZERS_PARALLELISM'] = 'True'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'True'

# Do NOT log models to WandB
os.environ["WANDB_LOG_MODEL"] = "false"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"


from pathlib import Path
import json
import argparse
from itertools import chain
from functools import partial
import math
import shutil
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from datasets import Dataset, features, concatenate_datasets
import wandb
from scipy.special import softmax
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from tokenizers import AddedToken
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
import copy


# Custom (cx) modules
from cxmetrics import train_metrics
from cxmetrics import compute_metrics
from utils import (load_cfg,
                   debugger_is_active)
from load_data import LoadData
import create_datasets


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class CustomTrainer(Trainer):
    def __init__(
            self,
            focal_loss_info: SimpleNamespace,
            *args,
            class_weights=None,
            **kwargs):
        super().__init__(*args, **kwargs)
        # Assuming class_weights is a Tensor of weights for each class
        self.class_weights = class_weights
        self.focal_loss_info = focal_loss_info

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels
        labels = inputs.pop("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        # Reshape for loss calculation
        if self.focal_loss_info.apply:
            loss_fct = FocalLoss(alpha=5, gamma=2, reduction='mean')
            loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                            labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
            if self.label_smoother is not None and "labels" in inputs:
                loss = self.label_smoother(outputs, inputs)
            else:
                loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                                labels.view(-1))
        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':

    # Determine if running in debug mode
    # If in debug manually point to CFG file
    is_debugger = debugger_is_active()

    # Construct the argument parser and parse the arguments
    if is_debugger:
        args = argparse.Namespace()
        args.dir = os.getenv('BASE_DIR') + '/cfgs/training'
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

    # Load data
    df_train, df_val = (LoadData(data_dir=CFG.paths.data_dir,
                                 train_files=CFG.paths.data.train,
                                 val_file=CFG.paths.data.val,
                                 path_tokenizer=str(
                                     Path(os.getenv('MODEL_DIR')) / CFG.model.name),
                                 split=CFG.data.split,
                                 max_token_length=CFG.tokenizer.max_token_length)
                        .load(explode=False,
                              ds_ratio=CFG.data.ds_factor,
                              mask_data=CFG.mask_data.apply,
                              mask_prob=CFG.mask_data.prob))

    # Get labels
    data = df_train.to_dict(orient='records') + \
        df_val.to_dict(orient='records')
    all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
    label2id = {l: i for i, l in enumerate(all_labels)}
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
            n=1500, random_state=42).reset_index(
            drop=True)
        # df_val = df_val.sample(n=250, random_state=42).reset_index(drop=True)

    # Datasets
    ds_train = create_datasets.train_dataset(
        tokenizer=tokenizer,
        label2id=label2id,
        data=df_train.copy(),
        max_length=CFG.tokenizer.max_token_length,
        stride=CFG.tokenizer.stride,
        num_proc=8)
    ds_val = create_datasets.train_dataset(
        tokenizer=tokenizer,
        label2id=label2id,
        data=df_val.copy(),
        max_length=CFG.tokenizer.max_token_length,
        stride=CFG.tokenizer.stride,
        num_proc=8)

    # Shuffle data
    ds_train = ds_train.shuffle(42)
    ds_val = ds_val.shuffle(42)

    # Model
    model = AutoModelForTokenClassification.from_pretrained(
        str(Path(os.getenv('MODEL_DIR')) / CFG.model.name),
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Resize model token embeddings if tokens were added
    if CFG.tokenizer.add_tokens is not None:
        model.resize_token_embeddings(
            len(tokenizer),
            # pad_to_multiple_of=CFG.tokenizer.pad_to_multiple_of,
        )

    # Freeze layers
    if CFG.model.freeze.apply:
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False if CFG.model.freeze.apply else True
        for layer in model.deberta.encoder.layer[:CFG.model.freeze.num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    # Collator
    collator = DataCollatorForTokenClassification(
        tokenizer,
        pad_to_multiple_of=CFG.tokenizer.pad_to_multiple_of,
    )

    # Calculate num train steps
    num_steps = CFG.train_args.num_train_epochs * len(ds_train)
    num_steps = num_steps / CFG.train_args.per_device_train_batch_size
    num_steps = num_steps / CFG.train_args.gradient_accumulation_steps
    print(f'My Calculated NUM_STEPS: {num_steps:,.2f}')

    # Step per epoch to eval every 0.2 epochs
    eval_steps = int(math.ceil((num_steps / CFG.train_args.num_train_epochs) *
                               CFG.train_args.eval_epoch_fraction))
    print(f'My Calculated eval_steps: {eval_steps:,}')

    # Setup WandB
    if CFG.debug:
        os.environ['WANDB_MODE'] = 'disabled'
        run = wandb.init(project='PII')
        run.name = 'junk-debug'
    else:
        os.environ['WANDB_MODE'] = CFG.wandb.mode
        wandb.login(key=os.getenv('wandb_api_key'))
        run = wandb.init(project='PII')

    # Directory to save results
    output_dir = Path(os.getenv('SAVE_DIR')) / f'{run.name}'
    if not output_dir.exists():
        output_dir.mkdir(parents=False, exist_ok=True)
    if run.name == 'junk-debug':
        os.system(f'rm -rf {str(output_dir)}/*')
    # Send copy of cfg to output directory
    shutil.copyfile(str(Path(args.dir) / args.name),
                    str(output_dir / args.name))

    # Trainer Arguments
    args = TrainingArguments(
        output_dir=str(output_dir),
        fp16=CFG.train_args.fp16,
        learning_rate=CFG.train_args.learning_rate,
        num_train_epochs=CFG.train_args.num_train_epochs,
        per_device_train_batch_size=CFG.train_args.per_device_train_batch_size,
        gradient_accumulation_steps=CFG.train_args.gradient_accumulation_steps,
        per_device_eval_batch_size=CFG.train_args.per_device_train_batch_size,
        report_to="wandb",
        evaluation_strategy="steps",
        save_total_limit=2,
        logging_steps=eval_steps,
        save_steps=eval_steps,
        lr_scheduler_type=CFG.train_args.lr_scheduler_type,
        metric_for_best_model=CFG.train_args.metric_for_best_model,
        greater_is_better=CFG.train_args.greater_is_better,
        warmup_ratio=CFG.train_args.warmup_ratio,
        weight_decay=CFG.train_args.weight_decay,
        load_best_model_at_end=True,
    )

    # Calculate class weights based on your dataset
    if CFG.class_weights.apply:
        train_labels = list(chain.from_iterable(
            [i['labels'] for i in ds_train]))
        val_labels = list(chain.from_iterable([i['labels'] for i in ds_val]))
        class_weights = compute_class_weight('balanced',
                                             classes=np.sort(
                                                 np.unique(train_labels)),
                                             y=train_labels + val_labels)
        if CFG.class_weights.approach == 'absolute':
            class_weights = torch.tensor(
                class_weights).to(torch.float32).to('cuda')
        elif CFG.class_weights.approach == 'mean':
            # class_weights[:12] = np.median(class_weights[:-1]) * CFG.class_weights.multiplier
            class_weights[:-1] = np.median(class_weights[:-1]) * \
                CFG.class_weights.multiplier
            class_weights = torch.tensor(
                class_weights).to(torch.float32).to('cuda')
        elif CFG.class_weights.approach == 'fixed':
            class_weights[:-1] = 19200.0
            # class_weights[12] = 0.07697
            class_weights = torch.tensor(
                class_weights).to(torch.float32).to('cuda')
        else:
            print('error in class weight')
            sys.exit()
    else:
        class_weights = None

    # Initialize Trainer with custom class weights
    if CFG.class_weights.apply or CFG.focal_loss.apply:
        trainer = CustomTrainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=collator,
            tokenizer=tokenizer,
            compute_metrics=partial(train_metrics, all_labels=all_labels),
            class_weights=class_weights,
            focal_loss_info=CFG.focal_loss,
        )
    else:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=collator,
            tokenizer=tokenizer,
            compute_metrics=partial(train_metrics, all_labels=all_labels),
        )
    trainer.train()

    ############################################
    # F5 Score on Validation Dataset
    # Adjust Threshold
    ############################################

    # Predict on val dataset
    predictions = trainer.predict(ds_val).predictions
    weighted_preds = softmax(predictions, axis=-1) * 1.0
    preds = weighted_preds.argmax(-1)
    # preds_without_O = weighted_preds[:, :, :12].argmax(-1)
    # O_preds = weighted_preds[:, :, 12]
    preds_without_O = weighted_preds[:, :, :-1].argmax(-1)
    O_preds = weighted_preds[:, :, -1]

    # Test various threshold levels
    f5_scores = {}
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]:
        preds_final = np.where(O_preds < threshold, preds_without_O, preds)
        # Prepare to plunder the data for valuable triplets!
        triplets = []
        document, token, label, token_str = [], [], [], []
        # For each prediction, token mapping, offsets, tokens, and document in
        # the dataset
        for p, row in zip(preds_final, ds_val):
            token_map = row['token_map']
            offsets = row['offset_mapping']
            tokens = row['tokens']
            doc = row['document']

            # Iterate through each token prediction and its corresponding
            # offsets
            for token_pred, (start_idx, end_idx) in zip(p, offsets):
                label_pred = id2label[token_pred]  # Predicted label from token

                # If start and end indices sum to zero, continue to the next
                # iteration
                if start_idx + end_idx == 0:
                    continue

                # If the token mapping at the start index is -1, increment
                # start index
                if token_map[start_idx] == -1:
                    start_idx += 1

                # Ignore leading whitespace tokens ("\n\n")
                while start_idx < len(
                        token_map) and tokens[token_map[start_idx]].isspace():
                    start_idx += 1

                # If start index exceeds the length of token mapping, break the
                # loop
                if start_idx >= len(token_map):
                    break

                token_id = token_map[start_idx]   # Token ID at start index

                # Ignore "O" predictions and whitespace tokens
                if label_pred != "O" and token_id != -1:
                    # Form a triplet
                    triplet = (doc, token_id)  # Form a triplet

                    # If the triplet is not in the list of triplets, add it
                    if triplet not in triplets:
                        document.append(doc)
                        token.append(token_id)
                        label.append(label_pred)
                        # token_str.append(tokens[token_id])
                        token_str.append(tokens[token_id])
                        triplets.append(triplet)

        # Prediction dataframe
        df_pred = pd.DataFrame({"document": document,
                                "token": token,
                                "label": label,
                                "token_str": token_str})
        # Score val data
        df_ref = df_val.copy()
        df_ref['document'] = df_ref['document'].astype(str)
        df_ref = df_ref[df_ref['document'].isin(
            ds_val['document'])].reset_index(drop=True)
        df_ref = (df_ref.explode(['tokens', 'labels', 'trailing_whitespace'])
                  .reset_index(drop=True)
                  .rename(columns={'labels': 'label'}))
        df_ref['token'] = df_ref.groupby('document').cumcount()
        df_ref = df_ref[df_ref['label'] != 'O'].copy()
        df_ref = df_ref.reset_index().rename(columns={'index': 'row_id'})
        df_ref = df_ref[['row_id', 'document', 'token', 'label']].copy()
        m = compute_metrics(df_pred, df_ref)
        print(f'Threshold: {threshold}; F5: {m["ents_f5"]:.4f}')
        f5_scores[f'f5_{threshold}'] = m['ents_f5']

    # Best threshold for F5
    best_threshold = -1.0
    best_f5 = -1.0
    for name, key in f5_scores.items():
        if key > best_f5:
            best_f5 = key
            best_threshold = float(name.split('f5_')[-1])
    print(f'Best F5: {best_f5:.4f}; Threshold: {best_threshold}')

    ############################################
    # Log Metrics to WandB
    ############################################
    # Trainer optimal checkpoint steps
    best_ckpt = trainer.state.best_model_checkpoint
    best_val_metric = trainer.state.best_metric
    print(f'Best CKPT: {best_ckpt}')
    print(f'best_val_metric: {best_val_metric}')

    # Log F5 score for holdout
    run.log({'best_ckpt': best_ckpt,
             'best_val_metric': best_val_metric})
    run.log(f5_scores)
    run.log({'best_f5': best_f5, 'best_threshold': best_threshold})

    # Num. steps for best checkpoint
    log_hist = copy.deepcopy(trainer.state.log_history)
    metric_name = f'eval_{CFG.train_args.metric_for_best_model}'
    optimal_steps = None
    for log_ in log_hist:
        for key, value in log_.items():
            if key == metric_name and value == best_val_metric:
                optimal_steps = log_['step']
    assert optimal_steps is not None, 'Error in Finding Optimal Steps'
    print(f'Optimal Steps: {optimal_steps:,}')
    print(f'trainer.state.max_steps: {trainer.state.max_steps:,}')
    del log_hist, metric_name, log_, key, value
    _ = gc.collect()

    # Final wandb log
    run.log({'optimal_steps_post': optimal_steps,
             'class_weights_approach': CFG.class_weights.approach,
             'dataset_name': '; '.join(CFG.paths.data.train),
             'model_name': CFG.model.name,
             'max_steps_post': trainer.state.max_steps})

    # Close wandb logger
    wandb.finish()

    ############################################
    # Clean up memory
    ############################################
    del run, trainer, model
    torch.cuda.empty_cache()
    _ = gc.collect()

    ############################################
    # Train on All Data
    ############################################
    # Create directory for saving all_data training
    output_all_dir = output_dir / 'all_data'
    output_tokenizer_dir = output_dir / 'tokenizer'
    if not output_all_dir.exists():
        output_all_dir.mkdir(parents=False, exist_ok=True)
        output_tokenizer_dir.mkdir(parents=False, exist_ok=True)

    # Combine train and val datasets
    ds_all = concatenate_datasets([ds_train, ds_val])
    ds_all = ds_all.shuffle(42)

    # Model
    model = AutoModelForTokenClassification.from_pretrained(
        str(Path(os.getenv('MODEL_DIR')) / CFG.model.name),
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    # Resize model token embeddings if tokens were added
    if CFG.tokenizer.add_tokens is not None:
        model.resize_token_embeddings(
            len(tokenizer),
            pad_to_multiple_of=CFG.tokenizer.pad_to_multiple_of,
        )

    # Trainer Arguments
    args = TrainingArguments(
    output_dir=str(output_all_dir),
    fp16=CFG.train_args.fp16,
    learning_rate=CFG.train_args.learning_rate,
    per_device_train_batch_size=CFG.train_args.per_device_train_batch_size,
    gradient_accumulation_steps=CFG.train_args.gradient_accumulation_steps,
    report_to="none",
    lr_scheduler_type='cosine',
    warmup_ratio=CFG.train_args.warmup_ratio,
    weight_decay=CFG.train_args.weight_decay,
    max_steps=optimal_steps,
    evaluation_strategy="no",
    save_total_limit=1,
    )

    # Initialize Trainer with custom class weights
    if not CFG.class_weights.apply and not CFG.focal_loss.apply:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_all,
            data_collator=collator,
            tokenizer=tokenizer,
            compute_metrics=partial(train_metrics, all_labels=all_labels),
        )
    else:
        trainer = CustomTrainer(
            model=model,
            args=args,
            train_dataset=ds_all,
            data_collator=collator,
            tokenizer=tokenizer,
            compute_metrics=partial(train_metrics, all_labels=all_labels),
            class_weights=class_weights,
            focal_loss_info=CFG.focal_loss,
        )
    trainer.train()

    # Save the trainer
    trainer.save_model(output_dir=output_all_dir)
    tokenizer.save_pretrained(save_directory=output_tokenizer_dir)

print('End of Script - Complete')
