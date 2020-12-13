import json
import pickle

from tqdm import tqdm

from dataset.base_dataset import PairUtils, PairDataset


class LCCCPairUtils(PairUtils):
    def __init__(self, txt_path, tokenizer):
        super(LCCCPairUtils, self).__init__(txt_path, tokenizer)

    def read_raw_file(self):
        with open(self.txt_path, "r", encoding="utf8") as fr_handle:
            data = json.load(fr_handle)

        return data

    def make_examples_pkl(self, data, pkl_path):
        with open(pkl_path, "wb") as pkl_handle:
            for dialog in tqdm(data):
                utterances = [self.tokenizer.tokenize(utt) for utt in
                              dialog[:-1]]
                response = self.tokenizer.tokenize(dialog[-1])
                pickle.dump(dict(utterances=utterances, response=response),
                            pkl_handle)

        print(pkl_path, " save completes!")


class LCCCPairDataset(PairDataset):
    def __init__(
            self,
            tokenizer,
            cache_dir: str = None,
            data_dir: str = None,
            task_name='LCCC_pair',
            split: str = 'train',
            use_EOT=False,
            max_sequence_len=512,
            data_file_type='json'
    ):
        super(LCCCPairDataset, self).__init__(
            tokenizer=tokenizer,
            pair_utils=LCCCPairUtils,
            cache_dir=cache_dir,
            data_dir=data_dir,
            task_name=task_name,
            split=split,
            use_EOT=use_EOT,
            max_sequence_len=max_sequence_len,
            data_file_type=data_file_type
        )

    def __getitem__(self, index):
        # Get Input Examples
        """
        InputExamples
          self.utterances = utterances
          self.response = response
          self.label
        """
        curr_example = self.input_examples[index]
        context, response = self.annotate_bi_sentence(curr_example)
        current_feature = dict()
        current_feature["context"] = context
        current_feature["response"] = response

        return current_feature
