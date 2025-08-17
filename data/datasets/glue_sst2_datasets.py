from datasets import load_dataset
import torch
from transformers import BertTokenizer
from torch.utils.data import random_split, Subset
from typing import Any


class GlueSST2Datasets:
    def __init__(self, data_config: dict, generator: torch.Generator):
        dataset = load_dataset("glue", "sst2")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        def tokenize_fn(example):
            return tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=128)

        tokenized: Any = dataset.map(tokenize_fn, batched=True)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        data_fraction: float = data_config["fraction"]
        full_train = tokenized["train"]
        total_size = int(len(full_train) * data_fraction)
        self._train = Subset(full_train, range(total_size))

        # 検証用データの分割（valid/test）
        valid_test = tokenized["validation"]
        valid_size = int(len(valid_test) * data_config["split"]["validation"])
        test_size = len(valid_test) - valid_size

        self._valid, self._test = random_split(valid_test, [valid_size, test_size], generator=generator)

    @property
    def datasets(self):
        return self._train, self._valid, self._test
