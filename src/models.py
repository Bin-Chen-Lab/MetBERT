import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel

class MetBERT(nn.Module):
    """
    A class which inhertis pytorch model class
    used for building custom layers on top of pre-trained bert models
    ...

    Attributes
    ----------
    dropout : int
        dropout threshold 
    classifier : object
        classifer layer with output units equal to number of classes

    """

    def __init__(self, model_path, drpout):
        super(BERTClass, self).__init__()
        self.bert = BertModel.from_pretrained(model_path, return_dict = True) # or automodel
        self.dropout = torch.nn.Dropout(drpout)
        self.classifier = torch.nn.Linear(768, 2)
        # nn.init.normal_(self.classifier.weight, std=0.02)
        # nn.init.xavier_normal_(self.classifier.weight)


    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):

        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:] # only if the output attention is true

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs

