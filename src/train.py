import os, sys
ROOT_DIR = "/work/fairness-privacy/"
sys.path.append(ROOT_DIR)

import pdb
import argparse
import warnings; warnings.simplefilter(action='ignore', category=FutureWarning)
from typing import *

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import datasets
import functools
import evaluate
from evaluate import EvaluationModule
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction
from transformers import get_scheduler

from private_transformers import PrivacyEngine
# private-transformers also has sub-classed versions of Trainer and TrainingArguments
private_transformers_path = ROOT_DIR + "private-transformers/"
sys.path.append(private_transformers_path)
from examples.classification.src.trainer import Trainer
from examples.classification.src.compiled_args import PrivacyArguments, TrainingArguments

import wandb; WANDB_PROJECT="fairness-privacy"


def compute_metrics_fn(acc_metric: EvaluationModule, eval_out: EvalPrediction):
    logits, labels = eval_out
    preds = np.argmax(logits, axis=-1)
    return acc_metric.compute(predictions=preds, references=labels)


def tokenize(batch, tokenizer, maxlen):
    tokenized = tokenizer(batch['text'], truncation=True, padding="max_length", max_length=maxlen)    
    return {**tokenized}


def load_model_and_tokenizer(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def train_private(
        args: argparse.Namespace,
        train_data: datasets.Dataset,
        val_data: datasets.Dataset
    ) -> AutoModelForSequenceClassification:
    """
    Finetunes classification model using a differential privacy (private-transformers, built on Opacus).

    Parameters:
    model: base model to fine-tune
    args: parsed arguments from command line
    train_data: tokenized train split of the data as a Dataset object
    val_data: tokenized val split of the data as a Dataset object    

    Returns:
    fine-tuned model that can be saved
    """
    wandb.init(project=WANDB_PROJECT, job_type='train-custom', config=vars(args))
    
    gradient_accumulation_steps = args.batch_size // args.per_device_train_batch_size
    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        evaluation_strategy=args.tracking,
        eval_steps=args.tracking_interval,
        report_to="wandb",
        logging_steps=args.tracking_interval,
        output_dir=args.model_out_path,
        n_gpu=torch.cuda.device_count()
    )    

    privacy_args = PrivacyArguments(
        target_epsilon=args.priv_epsilon,
        per_example_max_grad_norm=args.priv_max_grad_norm
        # TODO: accounting_mode, clipping_mode, target_delta
    )    

    acc_metric = evaluate.load_metric("accuracy")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        privacy_args=privacy_args,
        train_dataset=train_data,
        eval_dataset=val_data,        
        compute_metrics=functools.partial(compute_metrics_fn, acc_metric),
    )
    
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # apply weight decay to these
        {
            'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            'weight_decay': training_args.weight_decay
        },
        # but not to these
        {
            'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = trainer.optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate
    )
    num_training_steps = training_args.num_train_epochs * len(train_data)
    scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps
    )        
    # TODO: attach optimizer to privacy engine etc.
    
    total_train_batch_size = training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size
    privacy_engine = PrivacyEngine(
        module=model,
        batch_size=total_train_batch_size,
        sample_size=len(train_data),
        epochs=training_args.num_train_epochs,
        max_grad_norm=privacy_args.per_example_max_grad_norm,
        noise_multiplier=privacy_args.noise_multiplier,
        target_epsilon=privacy_args.target_epsilon,
        target_delta=privacy_args.target_delta,
        accounting_mode=privacy_args.accounting_mode,
        clipping_mode=privacy_args.clipping_mode,
        skip_checks=True,
    )
    # Originally, it could have been null.
    privacy_args.noise_multiplier = privacy_engine.noise_multiplier
    privacy_args.target_delta = privacy_engine.target_delta

    print('privacy_args: ')
    print(json.dumps(privacy_args.__dict__, indent=4))
    privacy_engine.attach(optimizer)
    # AND TRAIN...
    trainer.train(model_path=None)

    return trainer.model


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
    elif args.train_mode == 'private':
        assert args.priv_epsilon is not None and args.priv_max_grad_norm is not None, "`priv_max_grad_norm` and `priv_epsilon` required for private training"
        model_ft = train_private(model, args, train_data_tok, val_data_tok)
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
    parser.add_argument('--batch-size', type=int, required=False, default=64)
    parser.add_argument('--per-device-train-batch-size', type=int, required=False, default=16)
    parser.add_argument('--per-device-eval-batch-size', type=int, required=False, default=16)
    parser.add_argument('--weight-decay', type=float, required=False, default=0.01)
    parser.add_argument('--lr', type=float, required=False, default=1e-5)    
    parser.add_argument('--lr-scheduler', type=str, required=False, default="cosine")
    parser.add_argument('--warmup-ratio', type=float, required=False, default=0.1)
    parser.add_argument('--tracking', type=str, choices=['steps', 'epoch'], required=False, default='steps')
    parser.add_argument('--tracking-interval', type=int, required=False, default=10000)    

    # privacy arguments
    parser.add_argument('--priv-epsilon', type=float, required=False)
    parser.add_argument('--priv-max-grad-norm', type=float, required=False)
    # TODO: accounting_mode, clipping_mode, target_delta
    args = parser.parse_args()

    dataset = datasets.load_from_disk(args.data_path)
    train_helper(args, dataset['train'], dataset['validation'])