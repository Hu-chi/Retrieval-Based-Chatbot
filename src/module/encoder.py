from typing import Callable

import torch
import torch.nn as nn
from transformers.modeling_bert import BertModel


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def get_sentence_embedding(self, *args, **kwargs):
        raise Exception

    def resize_token_embeddings(self, *args, **kwargs):
        raise Exception


class BasicBertEncoder(Encoder):
    def __init__(
            self,
            param_path='bert-base-uncased',
            aggregation: [Callable, str] = 'cls'
    ):
        super(BasicBertEncoder, self).__init__()
        self._encoder = BertModel.from_pretrained(param_path)
        if isinstance(aggregation, str):
            if aggregation == 'cls':
                self.aggregation_layer = lambda x: x[:, 0]
            elif aggregation == 'mean':
                self.aggregation_layer = lambda x: torch.sum(x, dim=1)
            else:
                raise Exception(
                    "Aggregation Layer doesn't support %s!" % aggregation)
        elif isinstance(aggregation, Callable):
            self.aggregation_layer = aggregation
        else:
            raise Exception("Aggregation Layer doesn't support this!")

    def get_sentence_embedding(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor = None,
    ):
        embedding = \
            self._encoder(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids)[0]
        return self.aggregation_layer(embedding)

    def resize_token_embeddings(self, token_size):
        self._encoder.resize_token_embeddings(token_size)
