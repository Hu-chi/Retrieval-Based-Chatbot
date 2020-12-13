import os
import pickle

from torch.utils.data.dataset import Dataset


class PairUtils(object):
    def __init__(self, txt_path, tokenizer):
        super(PairUtils, self).__init__()
        self.txt_path = txt_path
        self.tokenizer = tokenizer

    def read_raw_file(self, *args, **kwargs):
        raise Exception

    def make_examples_pkl(self, *args, **kwargs):
        raise Exception


class PairDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            pair_utils,
            cache_dir: str = None,
            data_dir: str = None,
            task_name=None,
            split: str = None,
            use_EOT: bool = False,
            max_sequence_len: int = 512,
            data_file_type: str = 'txt'
    ):
        super(PairDataset, self).__init__()
        self.tokenizer = tokenizer
        self.input_examples = []
        self.split = split
        self.use_EOT = use_EOT
        self.max_sequence_len = max_sequence_len
        utterance_len_dict = dict()
        cache_file_path = os.path.join(cache_dir,
                                       "%s_%s.pkl" % (task_name, split))
        if not os.path.exists(cache_file_path):
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            raw_data_file_path = os.path.join(data_dir,
                                              "%s.%s" % (split, data_file_type))
            util_helper = pair_utils(raw_data_file_path, self.tokenizer)
            raw_data = util_helper.read_raw_file()
            util_helper.make_examples_pkl(raw_data, cache_file_path)

        with open(cache_file_path, "rb") as pkl_handle:
            while True:
                try:
                    example = pickle.load(pkl_handle)
                    num_examples = len(example["utterances"])
                    if num_examples in utterance_len_dict.keys():
                        utterance_len_dict[num_examples] += 1
                    else:
                        utterance_len_dict[num_examples] = 1
                    self.input_examples.append(example)
                    if len(self.input_examples) % 100000 == 0:
                        print("%d examples has been loaded!" % len(
                            self.input_examples))
                except EOFError:
                    break

        self.num_input_examples = len(self.input_examples)

        print("{utterances num: count} :", utterance_len_dict)
        print("total %s examples" % split, self.num_input_examples)

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, index):
        raise Exception

    def annotate_bi_sentence(self, example):
        dialog_context = []
        if self.use_EOT:
            for utt in example["utterances"]:
                dialog_context.extend(utt + ["[EOT]"])
            response = example["response"] + ["[EOT]"]
        else:
            for utt in example["utterances"]:
                dialog_context.extend(utt)
            response = example["response"]

        dialog_context, response = self.max_len_trim_seq(dialog_context,
                                                         response)

        dialog_context = ["[CLS]"] + dialog_context + ["[SEP]"]
        response = response + ["[SEP]"]

        dialog_context = self.tokenizer.convert_tokens_to_ids(dialog_context)
        response = self.tokenizer.convert_tokens_to_ids(response)

        return dialog_context, response

    def max_len_trim_seq(self, dialog_context, response):

        while len(dialog_context) + len(response) > self.max_sequence_len - 3:
            if len(dialog_context) > len(response):
                dialog_context.pop(0)  # from the front
            else:
                response.pop()

        return dialog_context, response
