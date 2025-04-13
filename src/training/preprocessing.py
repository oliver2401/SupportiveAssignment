import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



def format_example(example):
    formatted_text = f"Question: {example['question']} Answer: {example['answer']} <|endoftext|>"
    return {'text': formatted_text}

def split_dataset_from_df(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> DatasetDict:

    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-7:
        raise ValueError(
            f"train_ratio + validation_ratio + test_ratio must sum to 1.0; got {total_ratio}"
        )
    dataset = Dataset.from_pandas(df)

    train_testvalid = dataset.train_test_split(
        test_size=(1.0 - train_ratio),
        seed=seed
    )

    test_valid = train_testvalid["test"].train_test_split(
        test_size=test_ratio / (test_ratio + validation_ratio),
        seed=seed
    )

    split_datasets = DatasetDict({
        "train": train_testvalid["train"],
        "test": test_valid["test"],
        "validation": test_valid["train"]
    })

    split_datasets = split_datasets.map(format_example)

    return split_datasets


tokenizer = AutoTokenizer.from_pretrained('t5-small')
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):

    inputs = tokenizer(
        examples['text'],
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    labels = np.array(inputs['input_ids'], dtype=np.int64)
    labels[labels == tokenizer.pad_token_id] = -100
    inputs['labels'] = labels.tolist()
    return inputs