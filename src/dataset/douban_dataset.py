import os
import pickle

from torch.utils.data import Dataset
from tqdm import tqdm


class DoubanPairUtils(object):
    def __init__(self, txt_path, tokenizer):
        self.txt_path = txt_path
        self.tokenizer = tokenizer

    def read_raw_file(self):
        with open(self.txt_path, "r", encoding="utf8") as fr_handle:
            data = [line.strip() for line in fr_handle if len(line.strip()) > 0]

        return data

    def make_examples_pkl(self, data, pkl_path):
        with open(pkl_path, "wb") as pkl_handle:
            for dialog in tqdm(data):
                dialog_data = dialog.split("\t")
                label = dialog_data[0]
                utterances = [self.tokenizer.tokenize(utt) for utt in dialog_data[1:-1]]
                context_len = [len(utt) for utt in utterances]
                response = self.tokenizer.tokenize(dialog_data[-1])

                pickle.dump(
                    dict(
                        utterances=utterances,
                        response=response,
                        label=int(label),
                        seq_lengths=(context_len, len(response))
                    ),
                    pkl_handle
                )

        print(pkl_path, " save completes!")


class DoubanPairDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            cache_dir: str = None,
            data_dir: str = None,
            task_name='douban_pair',
            split: str = 'train',
            use_EOT=False,
            max_sequence_len=512
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.input_examples = []
        self.split = split
        self.use_EOT = use_EOT
        self.max_sequence_len = max_sequence_len
        utterance_len_dict = dict()
        cache_file_path = os.path.join(cache_dir, "%s_%s.pkl" % (task_name, split))
        if not os.path.exists(cache_file_path):
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            raw_data_file_path = os.path.join(data_dir, "%s.txt" % split)
            util_helper = DoubanPairUtils(raw_data_file_path, self.tokenizer)
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
                        print("%d examples has been loaded!" % len(self.input_examples))
                except EOFError:
                    break

        self.num_input_examples = len(self.input_examples)

        print("{utterances num: count} :", utterance_len_dict)
        print("total %s examples" % split, self.num_input_examples)

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, index):
        # Get Input Examples
        """
        InputExamples
          self.utterances = utterances
          self.response = response
          self.label
        """
        curr_example = self.input_examples[index]
        context, response = self._annotate_bi_sentence(curr_example)
        current_feature = dict()
        current_feature["context"] = context
        current_feature["response"] = response
        current_feature["label"] = float(curr_example["label"])

        return current_feature

    def _annotate_bi_sentence(self, example):
        dialog_context = []
        if self.use_EOT:
            for utt in example["utterances"]:
                dialog_context.extend(utt + ["[EOT]"])
            response = example["response"] + ["[EOT]"]
        else:
            for utt in example["utterances"]:
                dialog_context.extend(utt)
            response = example["response"]

        dialog_context, response = self._max_len_trim_seq(dialog_context, response)

        dialog_context = ["[CLS]"] + dialog_context + ["[SEP]"]
        response = response + ["[SEP]"]

        dialog_context = self.tokenizer.convert_tokens_to_ids(dialog_context)
        response = self.tokenizer.convert_tokens_to_ids(response)

        return dialog_context, response

    def _max_len_trim_seq(self, dialog_context, response):

        while len(dialog_context) + len(response) > self.max_sequence_len - 3:
            if len(dialog_context) > len(response):
                dialog_context.pop(0)  # from the front
            else:
                response.pop()

        return dialog_context, response
