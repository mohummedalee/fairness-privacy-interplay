import os, sys, json, pdb
ROOT_DIR = "/work/fairness-privacy/"
sys.path.append(ROOT_DIR)
os.environ["WANDB_DISABLED"] = "true"

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

# import wandb; WANDB_PROJECT="fairness-privacy"


def compute_metrics_fn(acc_metric: EvaluationModule, eval_out: EvalPrediction):
    logits, labels = eval_out
    preds = np.argmax(logits, axis=-1)
    return acc_metric.compute(predictions=preds, references=labels)


def tokenize(batch, tokenizer, maxlen):
    tokenized = tokenizer(batch['text'], truncation=True, padding="max_length", max_length=maxlen)    
    return {**tokenized}


def load_model_and_tokenizer(model_name, clf_labels=2):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=clf_labels)
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
    # wandb.init(project=WANDB_PROJECT, job_type='train-private', config=vars(args))
            
    # set up training and privacy arguments    
    gradient_accumulation_steps = args.batch_size // args.per_device_train_batch_size
    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        evaluation_strategy=args.tracking,
        do_eval=args.do_eval,
        eval_steps=args.tracking_interval,
        logging_steps=args.tracking_interval,
        output_dir=args.model_out_path
    )
    print('GPUS:', training_args.n_gpu)
    privacy_args = PrivacyArguments(
        target_epsilon=args.priv_epsilon,
        per_example_max_grad_norm=args.priv_max_grad_norm,
        non_private="no"
        # use default `accounting_mode`, `clipping_mode`, target_delta
    )    
    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # set up optimizer with recipe from lxuechen    
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
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate
    )    

    # set up trainer, lr scheduler, evaluation metrics etc.
    acc_metric = evaluate.load("accuracy")    
    num_training_steps = training_args.num_train_epochs * len(train_data)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=(optimizer, lr_scheduler),
        privacy_args=privacy_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        # compute_metrics can be extended by passing more metrics
        compute_metrics=functools.partial(compute_metrics_fn, acc_metric),
    )    
    
    privacy_engine = PrivacyEngine(
        module=model,
        batch_size=args.batch_size,  # the full batch size, not per device
        sample_size=len(train_data),
        epochs=training_args.num_train_epochs,
        max_grad_norm=privacy_args.per_example_max_grad_norm,
        noise_multiplier=privacy_args.noise_multiplier,
        target_epsilon=privacy_args.target_epsilon,
        target_delta=privacy_args.target_delta,
        accounting_mode=privacy_args.accounting_mode,
        clipping_mode=privacy_args.clipping_mode,
        skip_checks=True        
    )
    # Originally, it could have been null.
    privacy_args.noise_multiplier = privacy_engine.noise_multiplier
    privacy_args.target_delta = privacy_engine.target_delta    

    print('privacy_args: ')
    print(json.dumps(privacy_args.__dict__, indent=4))
    privacy_engine.attach(optimizer)
    # AND TRAIN...
    trainer.train(model_path=None, dev_objective="eval_accuracy")

    return trainer.model


def train_non_private(
        args: argparse.Namespace,
        train_data: datasets.Dataset,
        val_data: datasets.Dataset
    ) -> AutoModelForSequenceClassification:
    """
    Finetunes classification model using a transformer without differential privacy.

    Parameters:
    model: base model to fine-tune
    args: parsed arguments from command line
    train_data: tokenized train split of the data as a Dataset object
    val_data: tokenized val split of the data as a Dataset object    

    Returns:
    fine-tuned model that can be saved
    """
    # wandb.init(project=WANDB_PROJECT, job_type='train-non-private', config=vars(args))
            
    # set up training and privacy arguments    
    gradient_accumulation_steps = args.batch_size // args.per_device_train_batch_size
    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        evaluation_strategy=args.tracking,
        do_eval=args.do_eval,
        eval_steps=args.tracking_interval,
        # report_to="wandb",
        # _n_gpu=torch.cuda.device_count(),
        logging_steps=args.tracking_interval,
        output_dir=args.model_out_path,
        log_level="info"
    )
    print('GPUS:', training_args.n_gpu)

    privacy_args = PrivacyArguments(
        non_private="yes"
        # use default `accounting_mode`, `clipping_mode`, target_delta
    )    
    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # set up optimizer with recipe from lxuechen    
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
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate
    )    

    # set up trainer, lr scheduler, evaluation metrics etc.
    acc_metric = evaluate.load("accuracy")    
    num_training_steps = training_args.num_train_epochs * len(train_data)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        privacy_args=privacy_args,
        optimizers=(optimizer, lr_scheduler),
        train_dataset=train_data,
        eval_dataset=val_data,
        # compute_metrics can be extended by passing more metrics
        compute_metrics=functools.partial(compute_metrics_fn, acc_metric),
    )    
    
    # privacy_engine = PrivacyEngine(
    #     module=model,
    #     batch_size=args.batch_size,  # the full batch size, not per device
    #     sample_size=len(train_data),
    #     epochs=training_args.num_train_epochs,
    #     max_grad_norm=privacy_args.per_example_max_grad_norm,
    #     noise_multiplier=privacy_args.noise_multiplier,
    #     target_epsilon=privacy_args.target_epsilon,
    #     target_delta=privacy_args.target_delta,
    #     accounting_mode=privacy_args.accounting_mode,
    #     clipping_mode=privacy_args.clipping_mode,
    #     skip_checks=True        
    # )
    # Originally, it could have been null.
    # privacy_args.noise_multiplier = privacy_engine.noise_multiplier
    # privacy_args.target_delta = privacy_engine.target_delta    

    # print('privacy_args: ')
    # print(json.dumps(privacy_args.__dict__, indent=4))
    # privacy_engine.attach(optimizer)
    # AND TRAIN...
    trainer.train(model_path=None, dev_objective="eval_accuracy")

    return trainer.model


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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.train_mode == 'nonprivate':
        model_ft = train_non_private(args, train_data_tok, val_data_tok)
        model_ft.save_pretrained(args.model_out_path)
    elif args.train_mode == 'private':
        assert (args.priv_epsilon is not None and args.priv_max_grad_norm is not None), \
              "--priv-max-grad-norm and --priv-epsilon required for private training"
        model_ft = train_private(args, train_data_tok, val_data_tok)
        model_ft.save_pretrained(args.model_out_path)        
    else:
        raise Exception(f'training mode `{args.train_mode}` not implemented')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune a model for given classification task.')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--train-mode', type=str, choices=['nonprivate', 'private'], required=True,
                        help="whether to use private or non-private fine-tuning. If private, `--priv-epsilon` and `--priv-max-grad-norm` also required")
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
    parser.add_argument('--do-eval', default=False, action='store_true')
    parser.add_argument('--tracking-interval', type=int, required=False, default=10000)    

    # --- PRIVACY ARGUMENTS ---
    parser.add_argument('--priv-epsilon', type=float, required=False)
    parser.add_argument('--priv-max-grad-norm', type=float, required=False)
    parser.add_argument('--priv-clipping-mode', choices=['default', 'ghost'], default='default', required=False)
    # using default accounting_mode and target_delta
    args = parser.parse_args()

    dataset = datasets.load_from_disk(args.data_path)
    train_helper(args, dataset['train'], dataset['validation'])