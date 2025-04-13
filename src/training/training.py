import torch
from transformers import (AutoModelForSeq2SeqLM, Trainer, TrainingArguments)

def train_model(tokenized_train, tokenized_val, device="cpu"):

    model = AutoModelForSeq2SeqLM.from_pretrained('t5-small').to(device)
    model.config.pad_token_id = model.config.eos_token_id

    training_args = TrainingArguments(
        output_dir="./gpt2-medquad-finetuned",
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