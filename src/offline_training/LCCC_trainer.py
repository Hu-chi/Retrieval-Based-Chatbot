import logging
import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.adamw import AdamW
from tqdm import tqdm

from utils.common import setup_seed, get_bert_tokenizer
from utils.data_process import load_data_to_device, get_simple_dataloader
# from apex.parallel import DistributedDataParallel as DDP
from utils.modle_builder import build_biencoder_model


def train(model, optimizer, criterion, dataloader, valid_dataloader, hparam):
    model.train()
    step = 0
    total_loss = 0
    with tqdm(dataloader) as t:
        for data in t:
            step += 1
            encoding = load_data_to_device(data, hparam.device)
            score = model.forward(encoding, cross_dot_product=True)
            labels = torch.eye(score.size(0)).to(hparam.device)
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
                    torch.save(model.state_dict(),
                               os.path.join(hparam.model_checkpoint,
                                            "LCCC_base_pair_acc_%.5f" % acc))


def valid(model, dataloader, hparam):
    model.eval()
    total_num = 0
    total_true = 0.0
    with tqdm(dataloader) as t:
        for data in t:
            encoding = load_data_to_device(data, device=hparam.device)
            score = model.forward(encoding, cross_dot_product=True)
            labels = torch.eye(score.size(0)).to(hparam.device)
            total_true += torch.sum((score > 0) == labels).detach().cpu().item()
            total_num += score.size(0) ** 2
            t.set_postfix(acc=(total_true / total_num))
    return total_true / total_num


if __name__ == '__main__':

    logger = logging.getLogger(__file__)
    parser = ArgumentParser()
    parser.add_argument("--encoder", type=str, default="bert",
                        help="Select encoder for context and response: [bert]")
    parser.add_argument("--pretrain_checkpoint", type=str, default=None,
                        help="Path of pretrain encoder checkpoint")

    parser.add_argument("--data_dir", type=str, default="",
                        help="Path or url of the dataset. ")
    parser.add_argument("--cache_dir", type=str, default="",
                        help="Path or url of the dataset cache. ")
    parser.add_argument("--model_checkpoint", type=str, default="",
                        help="Path or url of the dataset cache. ")

    parser.add_argument('--log_file', '-log_file', type=str, default="",
                        help="Output logs to a file under this path")

    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4,
                        help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=10,
                        help="Batch size for test")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with evaluation before training")
    parser.add_argument("--valid_steps", type=int, default=5000,
                        help="Exec validation every X steps")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training "
                             "(-1: not distributed)")
    parser.add_argument("--world_size", type=int, default=4,
                        help="num process for training")
    parser.add_argument("--seed", type=int, default=None,
                        help="set seed for model to get fixed param")

    parser.add_argument("--best_acc", type=float, default=0,
                        help="set seed for model to get fixed param")
    parser.add_argument("--num_worker", type=int, default=0,
                        help="num worker for dataloader")
    parser.add_argument("--task_name", type=str, default="LCCC_base_pair",
                        help="num worker for dataloader")

    parser.add_argument("--epoch_id", type=int, default=0,
                        help="start epoch id")
    parser.add_argument("--max_norm", type=float, default=None,
                        help="Clipping gradient norm")

    args = parser.parse_args()
    args.distributed = (args.local_rank != -1)

    if args.seed is not None:
        setup_seed(args.seed)

    likelihood_criterion = nn.BCEWithLogitsLoss().to(args.device)

    if args.encoder == 'bert':
        tokenizer = get_bert_tokenizer(
            args.pretrain_checkpoint,
            add_tokens=['[EOT]']
        )
        two_tower_model = build_biencoder_model(
            args=args,
            model_tokenizer=tokenizer,
            aggregation='cls'
        )
        if args.best_acc > 0:
            two_tower_model.load_state_dict(torch.load(os.path.join(
                args.model_checkpoint,
                "%s_acc_%.5f" % (args.task_name, args.best_acc)
            )))
    else:
        raise Exception

    if args.distributed and args.device == 'cuda':
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.local_rank
        )
        likelihood_criterion.cuda(args.local_rank)
        two_tower_model = nn.parallel.DistributedDataParallel(
            two_tower_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    optimizer = AdamW(
        [{'params': two_tower_model.parameters(), 'initial_lr': args.lr}],
        lr=args.lr
    )
    train_dataloader, valid_dataloader, test_dataloader = \
        get_simple_dataloader(
            args, tokenizer=tokenizer, task_name=args.task_name
        )

    if args.eval_before_start:
        print(valid(two_tower_model, valid_dataloader, args))
        print(valid(two_tower_model, test_dataloader, args))

    for i in range(args.n_epochs):
        train(two_tower_model, optimizer, likelihood_criterion,
              train_dataloader, valid_dataloader, args)
        print(valid(two_tower_model, valid_dataloader, args))
        print(valid(two_tower_model, test_dataloader, args))
