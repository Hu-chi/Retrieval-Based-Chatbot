import random

import numpy as np
import torch
from transformers import BertTokenizerFast


def setup_seed(seed):  # 设置随机数种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_bert_tokenizer(checkpoint_path, add_tokens):
    model_tokenizer = BertTokenizerFast.from_pretrained(checkpoint_path)
    model_tokenizer.add_tokens(add_tokens)
    return model_tokenizer
