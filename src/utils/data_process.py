import torch
from torch.nn.utils.rnn import pad_sequence


def load_data_to_device(data, device):
    for key in data.keys():
        data[key] = data[key].to(device)
    return data


def collate_bi_info(batch, padding_value=0, batch_first=True):
    concat_context_input_ids = pad_sequence(
        [torch.tensor(instance["context"], dtype=torch.long) for instance in batch],
        batch_first=batch_first, padding_value=padding_value
    )
    response_input_ids = pad_sequence(
        [torch.tensor(instance["response"], dtype=torch.long) for instance in batch],
        batch_first=batch_first, padding_value=padding_value
    )
    data = dict()
    data["context_input_ids"] = concat_context_input_ids
    data["responses_input_ids"] = response_input_ids
    data["context_input_masks"] = concat_context_input_ids != padding_value
    data["responses_input_masks"] = response_input_ids != padding_value
    data["labels"] = torch.Tensor([instance["label"] for instance in batch])
    return data
