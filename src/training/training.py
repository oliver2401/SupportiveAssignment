import torch
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, GPT2LMHeadModel, GPT2TokenizerFast)

def train_model(tokenized_train, tokenized_val, device="cpu"):

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        eval_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )

    trainer.train()

    return trainer, model