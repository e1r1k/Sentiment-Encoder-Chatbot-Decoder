import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

class QADataset(Dataset):
    def __init__(
        self, config, tokenizer, 
    ):
        self.dataset = load_from_disk(config.dataset)
        n_subset = int(config.model_train_fraction * len(self.dataset))
        self.dataset= self.dataset.select(range(n_subset))
        print(
            f"Loaded dataset of size {len(self.dataset)} with columns {self.dataset.column_names}"
        )

        self.tokenizer = tokenizer
        self.max_length = config.max_len

        # Special token IDs (you can use these IDs in the __getitem__ method)
        self.pad_id = self.tokenizer.token_to_id(config.pad_token)
        self.sep_id = self.tokenizer.token_to_id(config.sep_token)
        self.end_id = self.tokenizer.token_to_id(config.end_token)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question, answer = self.dataset[idx]["question"], self.dataset[idx]["answer"]

        question = self.tokenizer.encode(question)
        answer = self.tokenizer.encode(answer)

        concat_qa = (question.ids + [self.sep_id] + answer.ids)[:self.max_length-1] + [self.end_id]
        sequence = concat_qa + [self.pad_id] * (self.max_length + 1 - len(concat_qa))

        source = torch.tensor(sequence[:-1])
        target = torch.tensor(sequence[1:])
        target[target == self.pad_id] = -100

        return {
            "source_sequence": source, 
            "target_sequence": target,
            "key_padding_mask": source == self.pad_id,
        }


if __name__ == "__main__":
    from config import config
    from tokenizers import Tokenizer

    # Sanity check the dataset class
    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    idx = 1
    config.max_len = 64 # For testing purposes
    dataset = QADataset(config, tokenizer)

    source, target, key_padding_mask = dataset[idx].values()

    print("Source sequence shape:", source.shape)
    print("Target sequence shape:", target.shape)
    print("Key padding mask shape:", key_padding_mask.shape)

    print("Source sequence:", source)
    print("Target sequence:", target)
    print("Key padding mask:", key_padding_mask)

    decoded_source = tokenizer.decode(source.tolist(), skip_special_tokens=False)
    decoded_target = tokenizer.decode(target[target != -100].tolist(), skip_special_tokens=False)
    print("Decoded source sequence:", decoded_source)
    print("Decoded target sequence:", decoded_target)

