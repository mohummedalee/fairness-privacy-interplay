import pdb
import argparse
import warnings; warnings.simplefilter(action='ignore', category=FutureWarning)
import os, sys
sys.path.append("/work/fairness-privacy")  # root dir
from typing import *

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import get_scheduler
import datasets

import wandb; WANDB_PROJECT="fairness-privacy"


def tokenize(batch, tokenizer, maxlen):
    tokenized = tokenizer(batch['text'], truncation=True, padding="max_length", max_length=maxlen)    
    return {**tokenized}


def evaluate_model(
        model: AutoModelForSequenceClassification,
        val_dataloader: DataLoader):    
    device = model.device
    model.eval()  # switch to evaluation mode
    val_loss, val_acc = 0, 0
    for batch in val_dataloader:        
        # includes input_ids, attention_mask, labels etc.
        batch_topass = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'labels': batch['label'].to(device)
        }
        outputs = model(**batch_topass)
        loss = outputs.loss
        logits = outputs.logits
        batch_size = batch_topass['input_ids'].shape[0]
        val_loss += loss.item() * batch_size
        # TODO: compare with GT predictions (batch_topass['labels'])

        preds = torch.argmax(logits, axis=-1)
        batch_acc = torch.sum(preds.cpu() == batch['label']).item()
        val_acc += batch_acc

    model.train()  # revert to training mode
    val_loss /= len(val_dataloader.dataset)
    val_acc /= len(val_dataloader.dataset)
    return val_loss, val_acc

# -> if use with Trainer required, something like this should be written
# def compute_metrics(eval_preds, train_preds=None):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


def train_custom_loop(
        model: AutoModelForSequenceClassification,
        args: argparse.Namespace,
        train_data: datasets.Dataset,
        val_data: datasets.Dataset
    ) -> AutoModelForSequenceClassification:
    """
    Finetunes classification model using a custom PyTorch training loop.

    Parameters:
    model: base model to fine-tune
    args: parsed arguments from command line
    train_data: tokenized train split of the data as a Dataset object
    val_data: tokenized val split of the data as a Dataset object    

    Returns:
    fine-tuned model that can be saved
    """
    # set up experiment tracking
    wandb.init(project=WANDB_PROJECT, job_type='train-custom', config=vars(args))

    num_epochs = args.epochs
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    num_training_steps = num_epochs * len(train_dataloader)  # 152,607 for twitter-AAE-sentiment    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # begin training
    model.to(device).train()  # set to training mode
    progress_bar = tqdm(range(num_training_steps))
    steps = 0  # steps taken so far
    best_val_acc = -float('inf')
    best_model = model
    for epoch in range(num_epochs):
        train_loss, pts_seen, train_correct = 0, 0, 0
        for batch in train_dataloader:
            # includes input_ids, attention_mask, labels etc.
            batch_topass = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['label'].to(device)
            }
            outputs = model(**batch_topass)
            logits = outputs.logits
            # count denominator of mean loss / accuracy as you go
            pts_seen += batch_topass['input_ids'].shape[0]            
                        
            loss = outputs.loss
            # haven't found it in source, but am convinced loss.item() is average loss of batch and not the sum
            train_loss += loss.item() * batch_topass['input_ids'].shape[0]
            loss.backward()  # compute gradients (based on `labels` passed to model)
            
            # make predictions for tracking train accuracy
            preds = torch.argmax(logits, axis=-1)
            batch_acc = torch.sum(preds.cpu() == batch['label']).item()
            train_correct += batch_acc

            optimizer.step()  # gradient update based on current learning rate
            scheduler.step()
            optimizer.zero_grad()  # clear out gradients, compute new ones for next batch
            # note: possibility to use gradient accumulation steps if larger batches needed
            progress_bar.update(1)
            steps += 1

            # track training metrics if args ask for it
            if args.tracking == "steps" and steps % args.tracking_interval == 0:
                progress_bar.set_description(f'Logged @ step {steps}')                
                eval_loss, eval_acc = evaluate_model(model, val_dataloader)
                train_loss /= pts_seen
                train_correct /= pts_seen
                metrics = {
                    'training_loss': train_loss,
                    'training_acc': train_correct,
                    'eval_loss': eval_loss,
                    'eval_acc': eval_acc
                }
                wandb.log(metrics)
                pts_seen = 0; train_correct = 0; train_loss = 0

        # check if i have the best model at the end of every epoch
        epoch_loss, epoch_acc = evaluate_model(model, val_dataloader)
        if epoch_acc > best_val_acc:
            best_model = model
            best_val_acc = epoch_acc

    return best_model

def train_helper(
        args: argparse.Namespace,
        train_data: datasets.Dataset,
        val_data: datasets.Dataset
    ):
    """
    Finetunes a sequence classification model on the given dataset

    Parameters:
    args: parsed arguments from command line
    train_data: train split of the data as a Dataset object
    val_data: val split of the data as a Dataset object

    Returns:
    None -- saves model before quitting
    """
    # tokenize train and validation data
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_data_tok = train_data.map(tokenize,
        num_proc=3,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "maxlen": args.tokenizer_maxlen}
    ).with_format("torch")
    val_data_tok = val_data.map(tokenize,
        num_proc=3,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "maxlen": args.tokenizer_maxlen}
    ).with_format("torch")
    
    # instantiate model and fine-tune based on training mode
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)    
    if args.train_mode == 'custom':
        model_ft = train_custom_loop(model, args, train_data_tok, val_data_tok)
        model_ft.save_pretrained(args.model_out_path)
    else:
        raise Exception(f'training mode `{args.train_mode}` not implemented')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune a model for given classification task.')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--train-mode', type=str, choices=['trainer', 'custom', 'private'], required=True,
                        help="`trainer` (use HF Trainer), `custom` (custom PyTorch training loop), or `private` (use private-transformers). If private, `privacy_args` also required")
    parser.add_argument('--model-path', type=str, default="FacebookAI/roberta-base", required=False)
    parser.add_argument('--model-out-path', type=str, required=True)
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--tokenizer-maxlen', type=int, required=False, default=128)
    parser.add_argument('--epochs', type=int, required=False, default=3)
    parser.add_argument('--batch-size', type=int, required=False, default=32)
    parser.add_argument('--weight-decay', type=float, required=False, default=0.01)
    parser.add_argument('--lr', type=float, required=False, default=1e-5)    
    parser.add_argument('--lr-scheduler', type=str, required=False, default="cosine")
    parser.add_argument('--warmup-ratio', type=float, required=False, default=0.1)
    parser.add_argument('--tracking', type=str, choices=['steps', 'epochs'], required=False, default='steps')
    parser.add_argument('--tracking-interval', type=int, required=False, default=10000)
    args = parser.parse_args()

    dataset = datasets.load_from_disk(args.data_path)
    train_helper(args, dataset['train'], dataset['validation'])