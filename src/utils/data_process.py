import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader

from dataset.LCCC_dataset import LCCCPairDataset
from dataset.douban_dataset import DoubanPairDataset


def load_data_to_device(data, device):
    for key in data.keys():
        data[key] = data[key].to(device)
    return data


def collate_bi_info(batch, padding_value=0, batch_first=True):
    concat_context_input_ids = pad_sequence(
        [torch.tensor(instance["context"], dtype=torch.long) for instance in
         batch],
        batch_first=batch_first, padding_value=padding_value
    )
    response_input_ids = pad_sequence(
        [torch.tensor(instance["response"], dtype=torch.long) for instance in
         batch],
        batch_first=batch_first, padding_value=padding_value
    )
    data = dict()
    data["context_input_ids"] = concat_context_input_ids
    data["responses_input_ids"] = response_input_ids
    data["context_input_masks"] = concat_context_input_ids != padding_value
    data["responses_input_masks"] = response_input_ids != padding_value
    if "label" in batch[0].keys():
        data["labels"] = torch.Tensor([instance["label"] for instance in batch])
    return data


def get_simple_dataloader(hparam, tokenizer, task_name="douban_pair"):
    if task_name == 'douban_pair':
        OptionDataset = DoubanPairDataset
    elif task_name == 'LCCC_base_pair':
        OptionDataset = LCCCPairDataset
    else:
        raise Exception(
            "Task %s doesn't have supported its dataset." % task_name
        )
    train_dataset = OptionDataset(tokenizer, hparam.cache_dir,
                                  hparam.data_dir, task_name, "train", True)
    valid_dataset = OptionDataset(tokenizer, hparam.cache_dir,
                                  hparam.data_dir, task_name, "valid", True)
    test_dataset = OptionDataset(tokenizer, hparam.cache_dir,
                                 hparam.data_dir, task_name, "test", True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hparam.world_size, rank=hparam.local_rank
    ) if hparam.local_rank != -1 else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas=hparam.world_size, rank=hparam.local_rank
    ) if hparam.local_rank != -1 else None
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hparam.world_size, rank=hparam.local_rank
    ) if hparam.local_rank != -1 else None

    train_dataloader = DataLoader(
        train_dataset, batch_size=hparam.train_batch_size,
        shuffle=False if hparam.distributed else True,
        collate_fn=collate_bi_info,
        sampler=train_sampler,
        num_workers=hparam.num_worker
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=hparam.valid_batch_size,
        shuffle=False,
        collate_fn=collate_bi_info,
        sampler=valid_sampler,
        num_workers=hparam.num_worker
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hparam.test_batch_size,
        shuffle=False,
        collate_fn=collate_bi_info,
        sampler=test_sampler,
        num_workers=hparam.num_worker
    )
    return train_dataloader, valid_dataloader, test_dataloader
