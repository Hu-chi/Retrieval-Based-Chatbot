import logging
import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast

from dataset.douban_dataset import DoubanPairDataset
from model.bi_encoder import BiEncoder
from module.encoder import BasicBertEncoder
from utils.common import setup_seed
from utils.data_process import load_data_to_device, collate_bi_info
from utils.metric import calculate_candidates_ranking, logits_recall_at_k, mean_average_precision_fn, precision_at_one, \
    logits_mrr


# from apex.parallel import DistributedDataParallel as DDP


def train(model, optimizer, criterion, dataloader, valid_dataloader, hparam):
    model.train()
    step = 0
    total_loss = 0
    with tqdm(dataloader) as t:
        for data in t:
            step += 1
            encoding = load_data_to_device(data, hparam.device)
            labels = encoding['labels'].reshape(-1, 1)
            score = model(encoding)
            loss = criterion(score, labels)

            loss.backward()
            total_loss += loss.cpu().detach().item()

            if step % hparam.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_postfix(loss=(total_loss / step))

            if step % hparam.valid_steps == 0:
                acc = valid(model, valid_dataloader, hparam)
                model.train()
                print("Valid dataset acc:", acc)
                if acc > hparam.best_acc:
                    hparam.best_acc = acc
                    torch.save(model.state_dict(), os.path.join(hparam.model_checkpoint, "douban_pair_acc_%.5f" % acc))


def valid(model, dataloader, hparam):
    model.eval()
    total_num = 0
    total_true = 0.0
    with tqdm(dataloader) as t:
        for data in t:
            encoding = load_data_to_device(data, device=hparam.device)
            labels = encoding['labels']
            score = model(encoding).squeeze(1)
            total_true += torch.sum((score > 0) == labels).detach().cpu().item()
            total_num += score.size(0)
            t.set_postfix(acc=(total_true / total_num))
    return total_true / total_num


def test(model, dataloader, hparam):
    model.eval()
    step = 0.0
    total_num = 0
    total_true = 0.0

    recall_k = 0
    precision_1 = 0
    mean_reciprocal_rank = 0
    mean_average_precision = 0

    with tqdm(dataloader) as t:
        for data in t:
            encoding = load_data_to_device(data, hparam.device)
            labels = encoding['labels']
            score = model(encoding).squeeze(1)

            total_true += torch.sum((score > 0) & (labels > 0)).detach().cpu().item()
            total_num += torch.sum((labels > 0)).detach().cpu().item()

            score = score.cpu().detach().numpy()
            labels = labels.cpu().numpy()

            rank_by_pred, pos_index, _ = calculate_candidates_ranking(score, labels, 10)

            recall_k += logits_recall_at_k(pos_index)
            precision_1 += precision_at_one(rank_by_pred)
            mean_reciprocal_rank += logits_mrr(pos_index)
            mean_average_precision += mean_average_precision_fn(pos_index)

            step += sum(sum(pred) > 0 for pred in rank_by_pred)
            if total_num != 0:
                t.set_postfix(true_positive=total_true / total_num)

    recall_k /= step
    precision_1 /= step
    mean_reciprocal_rank /= step
    mean_average_precision /= step

    print(recall_k, mean_reciprocal_rank, precision_1, mean_average_precision, sep='fuck\n')
    return recall_k, mean_reciprocal_rank, precision_1, mean_average_precision


def get_bert_tokenizer(checkpoint_path):
    model_tokenizer = BertTokenizerFast.from_pretrained(checkpoint_path)
    model_tokenizer.add_tokens(['[EOT]'])
    return model_tokenizer


def build_biencoder_model(encoder_checkpoint_path, model_tokenizer, device, aggregation='cls'):
    context_encoder = BasicBertEncoder(param_path=encoder_checkpoint_path, aggregation=aggregation)
    response_encoder = BasicBertEncoder(param_path=encoder_checkpoint_path, aggregation=aggregation)
    bi_encoder = BiEncoder(context_encoder, response_encoder)
    bi_encoder.resize_token_embeddings(len(model_tokenizer))
    if args.distributed:
        bi_encoder.cuda(args.local_rank)
    else:
        bi_encoder.to(device)
    return bi_encoder


def get_simple_dataloader(hparam):
    # train_dataset = DoubanPairDataset(tokenizer, hparam.cache_dir, hparam.data_dir, "douban_pair", "train", True)
    valid_dataset = DoubanPairDataset(tokenizer, hparam.cache_dir, hparam.data_dir, "douban_pair", "dev", True)
    test_dataset = DoubanPairDataset(tokenizer, hparam.cache_dir, hparam.data_dir, "douban_pair", "test", True)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if hparam.local_rank!=-1 else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas=hparam.world_size, rank=hparam.local_rank
    ) if hparam.local_rank != -1 else None
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hparam.world_size, rank=hparam.local_rank
    ) if hparam.local_rank != -1 else None

    # train_dataloader = DataLoader(train_dataset, batch_size=hparam.train_batch_size, shuffle=True,
    #                               collate_fn=collate_bi_info, sampler=train_sampler)
    valid_dataloader = DataLoader(valid_dataset, batch_size=hparam.valid_batch_size, shuffle=False,
                                  collate_fn=collate_bi_info, sampler=valid_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=hparam.test_batch_size, shuffle=False,
                                 collate_fn=collate_bi_info, sampler=test_sampler)
    return valid_dataloader, valid_dataloader, test_dataloader


if __name__ == '__main__':

    logger = logging.getLogger(__file__)
    parser = ArgumentParser()
    parser.add_argument("--encoder", type=str, default="bert", help="Select encoder for context and response: [bert]")
    parser.add_argument("--pretrain_checkpoint", type=str, default=None, help="Path of pretrain encoder checkpoint")

    parser.add_argument("--data_dir", type=str, default="", help="Path or url of the dataset. ")
    parser.add_argument("--cache_dir", type=str, default="", help="Path or url of the dataset cache. ")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path or url of the dataset cache. ")

    parser.add_argument('--log_file', '-log_file', type=str, default="", help="Output logs to a file under this path")

    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=10, help="Batch size for test")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=3,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--valid_steps", type=int, default=5000, help="Exec validation every X steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--world_size", type=int, default=4,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--seed", type=int, default=None, help="set seed for model to get fixed param")

    # parser.add_argument("--fp16", type=str, default="",
    #                     help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")

    # parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    # parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm up steps")
    # parser.add_argument("--scheduler", type=str, default="noam", choices=['noam', 'linear'], help="method of optim")

    args = parser.parse_args()
    args.best_acc = 0
    args.distributed = (args.local_rank != -1)

    if args.seed is not None:
        setup_seed(args.seed)

    if args.encoder == 'bert':
        tokenizer = get_bert_tokenizer(args.pretrain_checkpoint)
        two_tower_model = build_biencoder_model(args.pretrain_checkpoint, model_tokenizer=tokenizer, device=args.device)
    else:
        raise Exception

    likelihood_criterion = nn.BCEWithLogitsLoss().to(args.device)
    if args.distributed and args.device == 'cuda':
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.local_rank
        )
        likelihood_criterion.cuda(args.local_rank)
        two_tower_model = nn.parallel.DistributedDataParallel(two_tower_model,
                                                              device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)

    optimizer = AdamW([{'params': two_tower_model.parameters(), 'initial_lr': args.lr}], lr=args.lr)
    douban_train_dataloader, douban_valid_dataloader, douban_test_dataloader = get_simple_dataloader(args)

    if args.eval_before_start:
        print(valid(two_tower_model, douban_valid_dataloader, args))
        test(two_tower_model, douban_test_dataloader, args)

    for i in range(args.n_epochs):
        train(two_tower_model, optimizer, likelihood_criterion, douban_train_dataloader, douban_valid_dataloader, args)
        print(valid(two_tower_model, douban_valid_dataloader, args))
        test(two_tower_model, douban_test_dataloader, args)
