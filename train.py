import pdb
import argparse
import warnings; warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
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
        val_dataloader: DataLoader,
        train_dataloader: Optional[DataLoader] = None
    ):
    model.eval()
    device = model.device
    # TODO: make sure this device variable is accurate
    val_loss = 0
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
        val_loss += loss.item()
        # TODO: compare with GT predictions (batch_topass['labels'])


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
    Finetunes using a custom PyTorch training loop

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
    # TODO: can track training loss here?
    for epoch in range(num_epochs):
        train_loss, pts_seen = 0, 0
        for batch in train_dataloader:
            # includes input_ids, attention_mask, labels etc.
            batch_topass = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['label'].to(device)
            }  # things like dialect and text are not passed to the model
            pts_seen += args.batch_size
            outputs = model(**batch_topass)  # unpack dict and pass as kwargs
            
            # normally, i'd have to compute the loss with a custom loss_fn
            # but in HF, it's part of model output for convenience
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()  # compute gradients (based on `labels` passed to model)            

            optimizer.step()  # gradient update based on current training rate
            scheduler.step()
            optimizer.zero_grad()  # clear out gradients, compute new ones for next batch
            # TODO: read about gradient accumulation steps -- maybe i should incorporate it
            progress_bar.update(1)
            steps += 1

            # track training metrics if args ask for it
            if args.tracking and steps % args.tracking_interval == 0:
                progress_bar.set_description(f'Tracked metrics at step {steps}')
                eval_preds, train_preds = evaluate_model(model, val_data, train_data)
                metrics = compute_metrics(model, train_data, val_data)  # could also be re-used with a Trainer
                wandb.log(metrics)

    return model

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
        train_custom_loop(model, args, train_data_tok, val_data_tok)
    else:
        raise Exception(f'training mode `{args.train_mode}` not implemented')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune a model for given classification task')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--train-mode', type=str, choices=['trainer', 'custom', 'private'], required=True,
                        help="`trainer` (use HF Trainer), `custom` (custom PyTorch training loop), or `private` (use private-transformers). If private, `privacy_args` also required")
    parser.add_argument('--model-path', type=str, default="FacebookAI/roberta-base", required=False)
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--tokenizer-maxlen', type=int, required=False, default=128)
    parser.add_argument('--epochs', type=int, required=False, default=3)
    parser.add_argument('--batch-size', type=int, required=False, default=32)
    parser.add_argument('--weight-decay', type=float, required=False, default=0.01)
    parser.add_argument('--lr', type=float, required=False, default=1e-5)    
    parser.add_argument('--lr-scheduler', type=str, required=False, default="cosine")
    parser.add_argument('--warmup-ratio', type=float, required=False, default=0.1)
    parser.add_argument('--tracking', type=bool, required=False, default=True)
    parser.add_argument('--tracking-interval', type=int, required=False, default=1000)
    args = parser.parse_args()

    dataset = datasets.load_from_disk(args.data_path)
    train_helper(args, dataset['train'], dataset['validation'])