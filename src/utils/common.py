import random

import numpy as np
import torch
import torch.nn as nn
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


def model_and_optimizer_save(
        epoch_id,
        model: nn.Module,
        optimizer,
        checkpoint_path: str
):
    torch.save({
        'epoch': epoch_id,
        'state_dict': model.state_dict(),
        'best_loss': loss_v,
        'optimizer': optimizer.state_dict()
    }, checkpoint_path)


def get_saved_model_and_optimizer(
        model: nn.Module,
        optimizer,
        checkpoint_path: str
):
    model_CKPT = torch.load(checkpoint_path)
    model.load_state_dict(model_CKPT['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer
