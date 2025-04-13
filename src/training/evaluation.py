#import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score

###############################################################################
#         HELPER FUNCTIONS FOR GENERATION AND PREPARATION                     #
###############################################################################

def prepare_input(tokenizer, input_text, device="cpu"):
    prompt = f"Question: {input_text} Answer:"
    encoded_input = tokenizer.encode(prompt, return_tensors='pt')
    return encoded_input.to(device)

def generate_text(model, tokenizer, encoded_input, device="cpu"):
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

###############################################################################
#                  METRIC CALCULATION FUNCTIONS                               #
###############################################################################

def calculate_bleu(model, tokenizer, dataset, device="cpu"):
    total_bleu_score = 0
    for i, entry in enumerate(dataset):
        input_text = entry['question']
        reference_text = entry['answer']

        encoded_input = prepare_input(tokenizer, input_text, device=device)
        generated_text = generate_text(model, tokenizer, encoded_input, device=device)

        if 'answer:' in generated_text.lower():
            parts = generated_text.lower().split("answer:")
            output = parts[-1].strip()
        else:
            output = generated_text
        bleu_score = sacrebleu.corpus_bleu([output], [reference_text])
        total_bleu_score += bleu_score.score

    average_bleu_score = total_bleu_score / len(dataset)
    return average_bleu_score

def calculate_rouge_scores(model, tokenizer, dataset, device="cpu"):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []

    for entry in dataset:
        input_text = entry['question']
        reference_text = entry['answer']

        encoded_input = prepare_input(tokenizer, input_text, device=device)
        generated_text = generate_text(model, tokenizer, encoded_input, device=device)

        scores = scorer.score(reference_text, generated_text)
        rouge_scores.append(scores)

    average_scores = {
        'rouge1': np.mean([score['rouge1'].fmeasure for score in rouge_scores]),
        'rouge2': np.mean([score['rouge2'].fmeasure for score in rouge_scores]),
        'rougeL': np.mean([score['rougeL'].fmeasure for score in rouge_scores])
    }
    return average_scores

def calculate_bert_scores(model, tokenizer, dataset, device="cpu"):
    predictions = []
    references = []

    for entry in dataset:
        input_text = entry['question']
        reference_text = entry['answer']

        encoded_input = prepare_input(tokenizer, input_text, device=device)
        generated_text = generate_text(model, tokenizer, encoded_input, device=device)

        predictions.append(generated_text)
        references.append(reference_text)

    P, R, F1 = score(predictions, references, lang="en", rescale_with_baseline=True)

    average_scores = {
        'Precision': P.mean().item(),
        'Recall': R.mean().item(),
        'F1 Score': F1.mean().item()
    }
    return average_scores

###############################################################################
#                     MAIN EVALUATION WRAPPER                                 #
###############################################################################

def run_evaluation(trainer, tokenized_datasets, model_path="medical_chatbot_model", tokenizer_path="medical_chatbot_tokenizer", device="cpu"):

    results_train = trainer.evaluate(tokenized_datasets['train'])
    results_val   = trainer.evaluate(tokenized_datasets['validation'])
    results_test  = trainer.evaluate(tokenized_datasets['test'])

    results_df = pd.DataFrame({
        "Dataset": ["Training", "Validation", "Testing"],
        "epoch": [
            results_train.get('epoch', None),
            results_val.get('epoch', None),
            results_test.get('epoch', None)
        ],
        "Loss": [
            results_train['eval_loss'],
            results_val['eval_loss'],
            results_test['eval_loss']
        ],
        "eval_runtime": [
            results_train['eval_runtime'],
            results_val['eval_runtime'],
            results_test['eval_runtime']
        ],
        "eval_samples_per_second": [
            results_train['eval_samples_per_second'],
            results_val['eval_samples_per_second'],
            results_test['eval_samples_per_second']
        ],
        "eval_steps_per_second": [
            results_train['eval_steps_per_second'],
            results_val['eval_steps_per_second'],
            results_test['eval_steps_per_second']
        ]
    })

    # Reload the final model & tokenizer from disk for generation
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    # Evaluate on the test set with custom metrics
    test_set = tokenized_datasets['test']  # hugging face dataset
    average_bleu_score = calculate_bleu(model, tokenizer, test_set, device=device)
    average_rouge_scores = calculate_rouge_scores(model, tokenizer, test_set, device=device)
    bertscore_results = calculate_bert_scores(model, tokenizer, test_set, device=device)

    # Print out results
    print("===== Built-in Trainer Metrics =====")
    print(results_df)
    print("===== Custom Metrics (Test Set) =====")
    print(f"Average BLEU score: {average_bleu_score}")
    print("Average ROUGE scores:", average_rouge_scores)
    print("BERTScore Results:", bertscore_results)

    return results_df, average_bleu_score, average_rouge_scores, bertscore_results