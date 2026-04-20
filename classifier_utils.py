# Adapted from TELEClass: https://github.com/yzhan238/TELEClass
# See original repo for more details

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from joblib import Parallel, delayed
import os
import math
from math import ceil
from tqdm import tqdm
import pickle
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModel
import argparse
import json



class ClassModel(nn.Module):
    def __init__(self, encoder_name, enc_dim, class_embeddings):
        super(ClassModel, self).__init__()

        self.doc_encoder = AutoModel.from_pretrained(encoder_name)
        self.doc_dim = enc_dim

        self.num_classes, self.label_dim = class_embeddings.size()
        self.label_embedding_weights=nn.Parameter(class_embeddings, requires_grad=True)

        self.interaction = LBM(self.doc_dim, self.label_dim, n_classes=self.num_classes, bias=False)

    def forward(self, input_ids, attention_mask):
        doc_tensor = self.doc_encoder(input_ids, attention_mask=attention_mask)[1]
        scores = self.interaction(doc_tensor, self.label_embedding_weights)
        return scores

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


""" Interaction Layers
"""
class LBM(nn.Module):
    def __init__(self, l_dim, r_dim, n_classes=None, bias=True):
        super(LBM, self).__init__()
        self.weight = Parameter(torch.Tensor(l_dim, r_dim))
        self.use_bias = bias
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(n_classes))

        bound = 1.0 / math.sqrt(l_dim)
        init.uniform_(self.weight, -bound, bound)
        if self.use_bias:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, e1, e2):
        """
        e1: tensor of size (batch_size, l_dim)
        e2: tensor of size (n_classes, r_dim)
        return: tensor of size (batch_size, n_classes)
        """
        scores = torch.matmul(torch.matmul(e1, self.weight), e2.T)
        if self.use_bias:
            scores = scores + self.bias
        return scores


def encode(docs, tokenizer, max_len):
    encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, 
                                               max_length=max_len, padding='max_length',
                                                return_attention_mask=True, truncation=True, 
                                               return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks


def create_infer_dataset(docs, tokenizer, max_len=512, num_cpus=20):
    print(f"Converting texts into tensors.")
    chunk_size = ceil(len(docs) / num_cpus)
    chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
    results = Parallel(n_jobs=num_cpus)(delayed(encode)(docs=chunk, tokenizer=tokenizer, max_len=max_len) for chunk in chunks)
    input_ids = torch.cat([result[0] for result in results])
    attention_masks = torch.cat([result[1] for result in results])
    return {"input_ids": input_ids, "attention_masks": attention_masks}