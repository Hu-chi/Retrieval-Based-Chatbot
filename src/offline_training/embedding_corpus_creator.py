import os
from argparse import ArgumentParser

import faiss
import torch
from tqdm import tqdm

from utils.common import setup_seed, get_bert_tokenizer
from utils.data_process import load_data_to_device, get_simple_dataloader
from utils.modle_builder import build_biencoder_model

if __name__ == '__main__':
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

    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4,
                        help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=10,
                        help="Batch size for test")

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=None,
                        help="set seed for model to get fixed param")

    parser.add_argument("--best_acc", type=float, default=0,
                        help="set seed for model to get fixed param")
    parser.add_argument("--num_worker", type=int, default=0,
                        help="num worker for dataloader")
    parser.add_argument("--task_name", type=str, default="LCCC_base_pair",
                        help="num worker for dataloader")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training "
                             "(-1: not distributed)")

    args = parser.parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

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

    test_dataloader = \
        get_simple_dataloader(
            args,
            tokenizer=tokenizer,
            task_name=args.task_name,
            type='valid'
        )

    response_embedding_matrix = torch.empty(0)
    context_embedding_matrix = torch.empty(0)
    two_tower_model.eval()

    with tqdm(test_dataloader) as t:
        for data in t:
            encoding = load_data_to_device(data, args.device)

            response_embedding = two_tower_model.get_embedding(
                encoding['responses_input_ids'],
                encoding['responses_input_masks'],
                mode='response',
                norm=True
            ).detach().cpu()

            context_embedding = two_tower_model.get_embedding(
                encoding['context_input_ids'],
                encoding['context_input_masks'],
                mode='context',
                norm=True
            ).detach().cpu()

            response_embedding_matrix = torch.cat(
                [response_embedding_matrix, response_embedding], dim=0
            )

            context_embedding_matrix = torch.cat(
                [context_embedding_matrix, context_embedding], dim=0
            )

    index = faiss.IndexFlatL2(768)
    index.add(response_embedding_matrix.numpy())

    for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        acc = 0
        D, I_r = index.search(response_embedding_matrix.numpy(), 1)
        true_idx_list = [idxs[0] for idxs in I_r]
        D, I_c = index.search(context_embedding_matrix.numpy(), k)

        for i, true_idx in enumerate(true_idx_list):
            acc += true_idx in I_c[i]
        print("%d Acc: %f" % (k, acc / len(true_idx_list)))
