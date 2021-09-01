from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
import numpy as np
import time
import random
import pandas as pd
import torch

class CustomDataset(Dataset):

    """
    A class which inhertis pytorch dataset class
    used for preparing data compatible with BERT models
    ...

    Attributes
    ----------
    text : str
        discharge summary texts
    label : int
        label associated with dicharge summary
    tokenizer : object
        tokenizer initialted based on pretrained model type
    max_len : int
        maximum lenght of sequcne; for BERT based mdoel it is 512
    approach : str
        truncation approaches to use
        for more refer to this :

    Methods
    -------
    __getitem__
        gets dicharge summary sentences associated with index and it's corresponding labels
        and preprares them to be fed to BERT model
    Sidenote:
        WE rae not using token_type_ids here as most oft he text size is > 512 

    """

    def __init__(self, text, label, tokenizer, max_len, approach):
        self.tokenizer = tokenizer
        self.text = text
        self.label = label
        self.max_len = max_len
        self.trunc = approach

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = ' '.join(text.split())

        if self.trunc == 'head':

            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_attention_mask = True,
                truncation = True,
                return_tensors='pt'
            )

            ids = inputs['input_ids']
            ids = torch.squeeze(ids, 0)
            mask = inputs['attention_mask']
            mask = torch.squeeze(mask, 0)


        elif self.trunc == 'mixed':

            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_attention_mask = True,
                return_tensors='pt')

            ids = torch.cat([inputs["input_ids"].squeeze()[:256], inputs["input_ids"].squeeze()[-256:]])
            mask = torch.cat([inputs["attention_mask"].squeeze()[:256], inputs["attention_mask"].squeeze()[-256:]])


        elif self.trunc == 'tail':

            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_attention_mask = True,
                return_tensors='pt')

            ids = torch.cat([inputs["input_ids"].squeeze()[:1], inputs["input_ids"].squeeze()[-511:]])
            mask = torch.cat([inputs["attention_mask"].squeeze()[:1], inputs["attention_mask"].squeeze()[-511:]])

        else:
            raise NotImplementedError


        return {
            'ids': ids,
            'mask': mask,
            'targets': torch.tensor(self.label[index], dtype=torch.long)
        }

#  class_weights = class_weight.compute_class_weight('balanced',np.unique(dtrain['LABEL']), dtrain['LABEL'])
#     class_weights = torch.tensor(class_weights,dtype=torch.float)
#     loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)