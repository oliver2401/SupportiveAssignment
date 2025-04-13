import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,Trainer, TrainingArguments, GPT2LMHeadModel, GPT2TokenizerFast
from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, TrainerCallback, T5Config
import pandas as pd
import numpy as np



#device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

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

#tokenizer = AutoTokenizer.from_pretrained('gpt2') # t5-small
tokenizer = AutoTokenizer.from_pretrained('t5-small') 
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    inputs = tokenizer(examples['text'], max_length=256, truncation=True, padding="max_length")
    # Use numpy to efficiently replace pad_token_id with -100
    labels = np.array(inputs['input_ids'], dtype=np.int64)
    labels[labels == tokenizer.pad_token_id] = -100
    inputs['labels'] = labels.tolist()
    return inputs



df = pd.read_csv("data/mle_screening_dataset.csv")


splits = split_dataset_from_df(df, train_ratio=0.65, validation_ratio=0.25, test_ratio=0.1)

tokenized_datasets = {}

tokenized_datasets['train']= splits['train'].map(tokenize_function, batched=True, remove_columns=['text', 'question', 'answer'])
tokenized_datasets['validation']= splits['validation'].map(tokenize_function, batched=True, remove_columns=['text', 'question', 'answer'])
tokenized_datasets['test']= splits['test'].map(tokenize_function, batched=True, remove_columns=['text', 'question', 'answer'])


########
model = AutoModelForSeq2SeqLM.from_pretrained('t5-small').to(device) 
#model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model.config.pad_token_id = tokenizer.pad_token_id
training_args = TrainingArguments(
    output_dir="./gpt2-medquad-finetuned",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    fp16=torch.cuda.is_available(),  # Enable mixed precision if CUDA is available
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True, 
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

trainer.train()

#model.save_pretrained("./gpt2-medquad-finetuned")
#tokenizer.save_pretrained("./gpt2-medquad-finetuned")
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")


### Evaluation
results_train = trainer.evaluate(tokenized_datasets['train'])
results_val = trainer.evaluate(tokenized_datasets['validation'])
results_test = trainer.evaluate(tokenized_datasets['test'])

results_df = pd.DataFrame({
    "Dataset": ["Training", "Validation", "Testing"],
    "epoch":[results_train['epoch'], results_val['epoch'], results_test['epoch']],
    "Loss": [results_train['eval_loss'], results_val['eval_loss'], results_test['eval_loss']],
     "eval_runtime": [results_train['eval_runtime'], results_val['eval_runtime'], results_test['eval_runtime']],
      "eval_samples_per_second": [results_train['eval_samples_per_second'], results_val['eval_samples_per_second'], results_test['eval_samples_per_second']],
       "eval_steps_per_second": [results_train['eval_steps_per_second'], results_val['eval_steps_per_second'], results_test['eval_steps_per_second']]

})

