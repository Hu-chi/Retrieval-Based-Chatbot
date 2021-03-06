import pickle

from tqdm import tqdm

from dataset.base_dataset import PairUtils, PairDataset


class DoubanPairUtils(PairUtils):
    def __init__(self, txt_path, tokenizer):
        super(DoubanPairUtils, self).__init__(txt_path=txt_path,
                                              tokenizer=tokenizer)

    def read_raw_file(self):
        with open(self.txt_path, "r", encoding="utf8") as fr_handle:
            data = [line.strip() for line in fr_handle if len(line.strip()) > 0]

        return data

    def make_examples_pkl(self, data, pkl_path):
        with open(pkl_path, "wb") as pkl_handle:
            for dialog in tqdm(data):
                dialog_data = dialog.split("\t")
                label = dialog_data[0]
                utterances = [self.tokenizer.tokenize(utt) for utt in
                              dialog_data[1:-1]]
                response = self.tokenizer.tokenize(dialog_data[-1])

                pickle.dump(
                    dict(
                        utterances=utterances,
                        response=response,
                        label=int(label),
                    ),
                    pkl_handle
                )

        print(pkl_path, " save completes!")


class DoubanPairDataset(PairDataset):
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
        super(DoubanPairDataset, self).__init__(
            tokenizer=tokenizer,
            pair_utils=DoubanPairUtils,
            cache_dir=cache_dir,
            data_dir=data_dir,
            task_name=task_name,
            split=split,
            use_EOT=use_EOT,
            max_sequence_len=max_sequence_len
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
        current_feature["label"] = float(curr_example["label"])

        return current_feature
