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
model.save_pretrained("medical_chatbot_model")
tokenizer.save_pretrained("medical_chatbot_tokenizer")


### Evaluation
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score


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


tokenizer = AutoTokenizer.from_pretrained('medical_chatbot_tokenizer')
model = AutoModelForSeq2SeqLM.from_pretrained('medical_chatbot_model')




def prepare_input(tokenizer, input_text):
    prompt = f"Question: {input_text} Answer:"
    encoded_input = tokenizer.encode(prompt, return_tensors='pt')
    return encoded_input.to(device)

def generate_text(model, tokenizer, encoded_input):
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            encoded_input,
            max_length=512,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def calculate_bleu(model, tokenizer,dataset):
    total_bleu_score = 0
    for i, entry in enumerate(dataset):
        # print(i,entry)
        input_text = entry['question']
        reference_text = entry['answer']  # Reference texts need to be in a list of lists

        # Prepare and generate text
        encoded_input = prepare_input(tokenizer, input_text).to(model.device)
        generated_text = generate_text(model, tokenizer, encoded_input)

        # Extract the answer part from generated text
        if 'answer:' in generated_text:
            output = generated_text.split("answer:")[1].strip()
        else:
            output = generated_text
        bleu_score = sacrebleu.corpus_bleu([output], [reference_text])
        total_bleu_score += bleu_score.score
        # print(f"Example {i+1}, BLEU score: {bleu_score.score}")

    # Calculate average BLEU score
    average_bleu_score = total_bleu_score / len(dataset)
    return average_bleu_score


def calculate_rouge_scores(model, tokenizer, dataset):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []

    for entry in dataset:
        input_text = entry['question']
        reference_text = entry['answer']
        
        encoded_input = prepare_input(tokenizer, input_text).to(model.device)
        generated_text = generate_text(model, tokenizer, encoded_input)
        
        scores = scorer.score(reference_text, generated_text)
        rouge_scores.append(scores)

    average_scores = {
        'rouge1': np.mean([score['rouge1'].fmeasure for score in rouge_scores]),
        'rouge2': np.mean([score['rouge2'].fmeasure for score in rouge_scores]),
        'rougeL': np.mean([score['rougeL'].fmeasure for score in rouge_scores])
    }

    return average_scores

def calculate_bert_scores(model, tokenizer,dataset):

    predictions = []
    references = []
    
    # Generate predictions for each question in the dataset
    for entry in dataset:
        input_text = entry['question']
        reference_text = entry['answer']
        
        # Prepare and generate text
        encoded_input = prepare_input(tokenizer, input_text).to(model.device)
        generated_text = generate_text(model, tokenizer, encoded_input)
        
        # Store the generated and reference texts for batch scoring
        predictions.append(generated_text)
        references.append(reference_text)
    
    # Calculate BERTScores
    P, R, F1 = score(predictions, references, lang="en", rescale_with_baseline=True)
    
    # Compute average scores
    average_scores = {
        'Precision': P.mean().item(),
        'Recall': R.mean().item(),
        'F1 Score': F1.mean().item()
    }
    
    return average_scores


evu_dataset = splits['test']
average_bleu_score = calculate_bleu(model, tokenizer, evu_dataset) ## attention mask, eos pad
average_rouge_scores = calculate_rouge_scores(model, tokenizer, evu_dataset)
bertscore_results = calculate_bert_scores(model, tokenizer, evu_dataset)

print(f"Average BLEU score: {average_bleu_score}")
print("Average ROUGE scores:", average_rouge_scores)
print("BERTScore Results:", bertscore_results)