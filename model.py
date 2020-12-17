from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertModel
BERTLayerNorm = torch.nn.LayerNorm

class BertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained(config)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels * 36)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, starts=None, n_class=1):
        seq_length = input_ids.size(2)
        _, pooled_output = self.bert(input_ids.view(-1,seq_length),
                                     token_type_ids = token_type_ids.view(-1,seq_length),
                                     attention_mask = attention_mask.view(-1,seq_length))
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = logits.view(-1, 36)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, 36)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class BertForSequenceClassificationTrigger(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(config)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels * 36)

        self.fc_start = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, starts=None, n_class=1):
        seq_length = input_ids.size(2)
        opt, pooled_output = self.bert(input_ids.view(-1,seq_length),
                                     token_type_ids = token_type_ids.view(-1,seq_length),
                                     attention_mask = attention_mask.view(-1,seq_length))
        opt = self.dropout(opt)
        pooled_output = self.dropout(pooled_output)

        trigger_start_logits = self.fc_start(opt).view(-1, 512)
        loss_fct = CrossEntropyLoss()
        loss_trigger = loss_fct(trigger_start_logits, starts.view(-1))

        trigger_soft_max = torch.softmax(trigger_start_logits, -1)

        #pooled_output = pooled_output + trigger_soft_max.unsqueeze(1).bmm(opt).squeeze(1)
        pooled_output = trigger_soft_max.unsqueeze(1).bmm(opt).squeeze(1)

        logits = self.classifier(pooled_output)
        logits = logits.view(-1, 36)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, 36)
            loss = loss_fct(logits, labels)
            loss = loss + 0.01 * loss_trigger
            return loss, logits
        else:
            return logits