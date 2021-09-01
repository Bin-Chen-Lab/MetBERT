import os
import numpy as np

import sklearn
import sklearn.ensemble
import sklearn.metrics
from scipy.special import softmax

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn

from captum.attr import IntegratedGradients, LayerIntegratedGradients, InputXGradient
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import matplotlib.pyplot as plt


# Models and tokenizers loading 
model_path = '' # same as model_type in run.py
tokenizer = BertTokenizer.from_pretrained(model_path, truncation=True)
pretrain_path = 'path to fine-tuned .pt model file'
model = BertForSequenceClassification.from_pretrained(model_path, return_dict = True, num_labels = 2)
model.load_state_dict(torch.load(pretrain_path))


def compute_bert_outputs(model_bert, embedding_output, attention_mask=None, head_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
    else:
        head_mask = [None] * model_bert.config.num_hidden_layers

    encoder_outputs = model_bert.encoder(embedding_output,
                                        extended_attention_mask,
                                        head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = model_bert.pooler(sequence_output)
    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)   



class BertModelWrapper(nn.Module):
    
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, embeddings):        
        outputs = compute_bert_outputs(self.model.bert, embeddings)
        pooled_output = outputs[1]
        
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)       # [16, 768] --> [16, 2] assuming no of labels is 2
        z = torch.softmax(logits, dim=1)[:, 1].unsqueeze(1) # [16, 2] --> [16, 1]
        return z


def interpret_sentence(model_wrapper, sentence, label=1):

    model_wrapper.eval()
    model_wrapper.zero_grad()
    
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True, max_length=512, padding='max_length', truncation=True)])
    input_embedding = model_wrapper.model.bert.embeddings(input_ids)
    
    # predict
    pred = model_wrapper(input_embedding).item()
    pred_ind = round(pred)

    # compute attributions and approximation delta using integrated gradients
    attributions_ig, delta = ig.attribute(input_embedding, n_steps=100, return_convergence_delta=True)

    print('pred: ', pred_ind, '(', '%.2f' % pred, ')', ', delta: ', abs(delta))

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].numpy().tolist())    
    attributions = add_attributions_to_visualizer(attributions_ig, tokens, pred, pred_ind, label, delta, vis_data_records_ig)
    
    return tokens[1:-1], attributions[1:-1]  


def add_attributions_to_visualizer(attributions, tokens, pred, pred_ind, label, delta, vis_data_records):
    # mean attributions
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().numpy()
    
    # l2 norm
    # attributions = attributions.norm(p=1, dim=-1).squeeze(0)
    # attributions = attributions.detach().numpy()
    
    attributions[0]=0
    attributions[-1]=0
    attributions /= max(attributions)
    
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions[1:-1],
                            pred,
                            pred_ind,
                            label,
                            "label",
                            attributions.sum(),       
                            tokens[1:-1],
                            delta))
    
    return attributions

wrapper = BertModelWrapper(model)

ig = IntegratedGradients(wrapper)

_ig = []

sentence = '' # text you want to try out
label = # actual label

tokens, attributions = interpret_sentence(wrapper, sentence, label=label)

mm1 = sorted(tuple(zip(tokens, attributions)), key=lambda x: x[1], reverse=True)
mm2 = sorted(tuple(zip(tokens, attributions)), key=lambda x: x[1])

print(f'tokens with highest attribution scores {mm1}')
print(f'tokens with lowest attribution scores {mm2}')
print('\n')

_ = visualization.visualize_text(_ig)