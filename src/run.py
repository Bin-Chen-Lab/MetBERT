__author__ = "Omkar Kulkarni"

import argparse
import logging
import numpy as np
import time
import random
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import logging
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from sklearn.utils import class_weight
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from itertools import compress
from models import *
from dataloader import *
from train_val import *
from met import *
from utiles import *

# Tried but did not work
# tried incorporating bucketiterator with dataloaders but max lenth is 512 and mainly used
# all the time in our case sbo it would not work

# things updated:
# udpated cleaning strategy
# updated split from 70 15 15 to 80 10 10
# shuffled and stratified at splitting
# new optimizer with additional params
# changwes defaults for opti params and scgeuler
# added truncation strategies


def main(hparams) -> None:

    torch.manual_seed(hparams.seed)
    
    # prioritizing on cpu here
    if hparams.device == 'cpu':
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No Device found')

    if hparams.model_type == 'clinical':
        model_path = 'emilyalsentzer/Bio_Discharge_Summary_BERT'
        tokenizer = BertTokenizer.from_pretrained(model_path, truncation=True)
    elif hparams.model_type == 'pubmedbert':
        model_path = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        tokenizer = BertTokenizer.from_pretrained(model_path, truncation=True)
    else:
        pass #// add others such as bert, bluebert and biobert

    # reading in data
    dtrain = pd.read_csv('../data/train.csv')
    dtest = pd.read_csv('../data/test.csv')

    training_set = CustomDataset(dtrain.TEXT.values, dtrain['LABEL'].values, tokenizer, hparams.maxlen, hparams.approach)
    test_set = CustomDataset(dtest.TEXT.values, dtest['LABEL'].values, tokenizer, hparams.maxlen, hparams.approach)

    if hparams.balance_sampler:
        _, counts = np.unique(dtrain['LABEL'].values, return_counts=True)
        weights = 1.0 / torch.tensor(counts, dtype=torch.float)
        sample_weights = weights[dtrain['LABEL'].values]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_params = {'batch_size': hparams.train_batch,
                        'num_workers': 0,
                        'sampler': sampler
                        }
    else:
        train_params = {'batch_size': hparams.train_batch,
                        'num_workers': 0,
                        'shuffle': True
                        }

    test_params = {'batch_size': hparams.valid_batch,
                    'shuffle': False,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    test_loader = DataLoader(test_set, **test_params)

    if hparams.evaluation == 'yes':
        dval = pd.read_csv('../data/val.csv')
        val_set = CustomDataset(dval.TEXT.values, dval['LABEL'].values, tokenizer, hparams.maxlen, hparams.approach)
        val_loader = DataLoader(val_set, **test_params)

    # either load class from huggingace or write you own in model.py
    # latter will give you more freedom in terms of customizing weights and stuff
    # model = MetBERT(model_path, hparams.dropout_rate)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict = True, num_labels = 2)
    model.to(device)
    # model.load_state_dict(torch.load(model_name))

    if hparams.train_last == 'yes':
        for param in model.parameters():
            param.requires_grad = False

    if hparams.opti_params == 'yes':
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay))], 'weight_decay': 0.0},
            # {'params': model.l4.weight, "lr": lr2, 'weight_decay': 0.01},
            # {'params': model.l4.bias, "lr": lr2, 'weight_decay': 0.0},
            ]
        optimizer = AdamW(optimizer_parameters, lr=hparams.LEARNING_RATE, eps=hparams.epsilon)
    else:
        optimizer = AdamW(model.parameters(), lr=hparams.LEARNING_RATE, eps=hparams.epsilon)

    if hparams.scheduler == 'yes':
        total_steps = len(training_loader) * hparams.EPOCHS
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps, # 0 is default
                                                    num_training_steps=total_steps)
    # model name to keep track of hyperparametrs used
    model_name = '' +  hparams.model_type +  '_' + str(hparams.EPOCHS) + 'ep_' + \
                str(hparams.LEARNING_RATE) + 'lr_' + str(hparams.train_batch) + \
                str(hparams.valid_batch) + 'bt_' + hparams.balance_sampler + 'sampler_' + \
                hparams.opti_params + 'extparams_' + hparams.scheduler + 'sch_' + hparams.train_last + 'tlast_' + \
                hparams.approach + 'approach_' + '_can.pt'
                
    model = train(model, optimizer, training_loader,
                    hparams.EPOCHS, device, hparams.clip_norm, model_name, val_loader, hparams.scheduler, hparams.evaluation)

    print("Testing Begins!")
    preds, labels, probs = bert_predict(model, test_loader, device)
    all_metrics(preds, labels, probs, model_name)
    print("Saving Missclassified Predictions!")
    dtest2 = dtest
    dtest2['Y_PREDICT'] = preds
    dtest2['Y_PROBA'] = probs
    dtest2 = dtest2.rename(columns = {'LABEL': 'Y_TRUE'})
    idx = dtest2['Y_PREDICT'] != dtest2['Y_TRUE']
    res = list(compress(range(len(idx)), idx))
    misclassified_dtest = dtest2.iloc[res]
    misclassified_dtest.to_csv(str(model_name) + '_misclassified.csv', index = False)
    print('Done')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fine tuned Transformer Classifier",
        add_help=True,
    )

    parser.add_argument("--seed", type=int, default=25, help="Training seed.")

    parser.add_argument(
        "--model_type",
        default='pubmedbert',
        type=str,
        help="Path of the pre-trained model in Huggingface library",
        choices=["discharge", "pubmedbert", "bert", "bluebert", "biobert"],
    )

    parser.add_argument(
        "--balance_loss_function",
        default='no',
        type=str,
        help="Balancing the dataset by providing weights to the Loss function"
    )

    parser.add_argument(
        "--balance_sampler",
        default='no',
        type=str,
        help="Adding weighted sampler "
    )

    parser.add_argument(
        "--EPOCHS",
        default=1,
        type=int,
        help="Number of training epochs for model",
    )

    parser.add_argument(
        "--maxlen",
        default=512,
        type=int,
        help="Max Length",
    )

    parser.add_argument(
        "--LEARNING_RATE",
        default=2e-5,
        type=float,
        help="Learning Rate",
    )

    parser.add_argument(
        "--epsilon",
        default=1e-8,
        type=float,
        help="Epsilon Rate for AdamW",
    )

    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout Rate",
    )

    parser.add_argument(
        "--opti_params",
        default='no',
        type=str,
        help="Additional tweaks in Adam optimizer",
    )

    parser.add_argument(
        "--approach",
        default='head',
        type=str,
        help="Truncation Approach",
        choices=["head", "mixed", "tail"]
    )

    parser.add_argument(
        "--scheduler",
        default='no',
        type=str,
        help="Using Scheduler or not",
    )

    parser.add_argument(
        "--train_last",
        default='no',
        type=str,
        help="Only train last layer",
    )

    parser.add_argument(
        "--train_batch",
        default=32,
        type=int,
        help="Training Batch size",
    )

    parser.add_argument(
        "--valid_batch",
        default=16,
        type=int ,
        help="Validation/Testing Batch size",
    )

    parser.add_argument(
        "--no_workers",
        default=0,
        type=int ,
        help="Number of workers for the dataloader",
    )

    parser.add_argument(
        "--device",
        default='cpu',
        type=str,
        help="Choose gpu vs cpu for training",
    )

    parser.add_argument(
        "--clip_norm",
        default='yes',
        type=str,
        help="Choose to clip gradients",
    )

    parser.add_argument(
        "--evaluation",
        default='no',
        type=str,
        help="Training or validation",
        choices=["yes", "no"]
    )

    hparams = parser.parse_args()

    main(hparams)