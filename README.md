# SupportiveAssignment
Contains code for Supportive code challenge



# Medical Assistant Bot

## Overview

This project is a basic medical question-answering system that uses a pre-trained language model fine-tuned on a custom dataset. The dataset contains medical information about various diseases and their treatments/preventions.

## Approach
	• **Data Preprocessing**: We load the dataset (mle_screening_dataset.csv) and apply basic cleaning (e.g., dropping nulls). We split the data into training and test sets.
	• **Model Selection**: We use a pre-trained Transformer model (e.g., T-5) from Hugging Face, specifically for question-answering tasks.
	• **Training**: We fine-tune the model on the training set. This step can be customized further to improve results (e.g., optimizing hyperparameters, employing advanced data augmentation, etc.).
	• **Evaluation**: We evaluate using a simple accuracy measure, checking how many predicted answers match the ground truth. More robust metrics like Exact Match (EM) or F1 are recommended for QA tasks.
	
    **Potential Improvements**:
	    • Use more advanced QA models (e.g., BERT-Large, RoBERTa, etc.).
	    • Enhance the preprocessing to handle complex queries.
	    • Use more sophisticated QA metrics or human evaluation for medical relevance.

## Assumptions
	• We assume that every row in the dataset has a context, a question, and an answer.
	• We assume the dataset is already fairly clean or minimal cleaning is required.

## Strengths and Weaknesses
	• Strengths:
	• Quick to set up with Hugging Face pipelines.
	• Transfer learning benefits from large pre-trained models.
	• Weaknesses:
	• QA accuracy heavily depends on the dataset size and quality.
	• Model might not handle out-of-distribution medical queries well.

## How to Run
	1.	Clone the repo and navigate into it:

    ```
    git clone git@github.com:oliver2401/SupportiveAssignment.git
    cd medical-assistant-bot
    ```

    2.	Build Docker Image:

    ```
    docker build -t medical-bot .
    ```

    3.	Run Container:

    ```
    docker run -it medical-bot
    ```

    This will start the process of installing requirements, training, and eventually you can run the inference script.

	4.	(Optional) If you want to run locally without Docker, install the dependencies:

    ```
    pip install -r requirements.txt
    ```

    Then run:
    ```
    python -m src.data_preprocessing
    python -m src.model_training
    python -m src.inference
    ```


## Results for Bleu/Rouge scores:

```
>>> print(f"Average BLEU score: {average_bleu_score}")
Average BLEU score: 0.0005052506433694588
>>> print("Average ROUGE scores:", average_rouge_scores)
Average ROUGE scores: {'rouge1': np.float64(0.13002570632525373), 'rouge2': np.float64(0.05252692549022517), 'rougeL': np.float64(0.1148791558235285)}
>>> print("BERTScore Results:", bertscore_results)
BERTScore Results: {'Precision': 0.10485164821147919, 'Recall': -0.17924591898918152, 'F1 Score': -0.042341481894254684}
```