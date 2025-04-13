import pandas as pd

from src.training.data_cleaning import clean_data
from src.training.preprocessing import split_dataset_from_df, tokenize_function, tokenizer, device
from src.training.training import train_model
from src.training.evaluation import run_evaluation, calculate_bleu, calculate_bert_scores, calculate_rouge_scores
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main():
    # 1. Load the CSV
    df = pd.read_csv("data/mle_screening_dataset.csv")

    # 2. Clean the data
    df = clean_data(df)

    # 3. Split into train/val/test and tokenize
    splits = split_dataset_from_df(df, train_ratio=0.65, validation_ratio=0.25, test_ratio=0.1)

    tokenized_datasets = {}
    tokenized_datasets['train'] = splits['train'].map(
        tokenize_function, batched=True, remove_columns=['text', 'question', 'answer']
    )
    tokenized_datasets['validation'] = splits['validation'].map(
        tokenize_function, batched=True, remove_columns=['text', 'question', 'answer']
    )
    tokenized_datasets['test'] = splits['test'].map(
        tokenize_function, batched=True, remove_columns=['text', 'question', 'answer']
    )

    # 4. Train
    trainer, model = train_model(
        tokenized_train=tokenized_datasets['train'],
        tokenized_val=tokenized_datasets['validation'],
        device=device
    )

    # Save your final model/ tokenizer
    model.save_pretrained("medical_chatbot_model")
    tokenizer.save_pretrained("medical_chatbot_tokenizer")

    # 5. Evaluation
    run_evaluation(dataset=splits, model_path="medical_chatbot_model", tokenizer_path="medical_chatbot_tokenizer", device=device)

if __name__ == "__main__": 
    main()
