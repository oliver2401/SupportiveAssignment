import re
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    question_words = ['what', 'who', 'why', 'when', 'where', 'how', 'is', 'are', 'does', 'do', 'can', 'will', 'shall']

    df['question'] = df['question'].str.lower()

    df = df[df['question'].str.split().str[0].isin(question_words)]

    # Check for duplicate rows in the dataset
    duplicates = df.duplicated()
    print(f"Number of duplicate rows: {duplicates.sum()}")

    df = df.drop_duplicates()

    df.reset_index(drop=True, inplace=True)

    df = df.drop_duplicates(subset='question', keep='first').reset_index(drop=True)
    df = df.drop_duplicates(subset='answer', keep='first').reset_index(drop=True)

    df = df.dropna(subset=['question', 'answer']).reset_index(drop=True)

    df['question'] = df['question'].fillna('').astype(str)
    df['answer'] = df['answer'].fillna('').astype(str)

    def clean_text(text):
        text = re.sub(r"\(.*?\)", "", text) 
        text = re.sub(r'\s+', ' ', text.strip().lower())
        return text

    df['question'] = df['question'].apply(clean_text)
    df['answer'] = df['answer'].apply(clean_text)

    df['question'] = df['question'].str.lower().str.strip().apply(lambda x: re.sub(r'\s+', ' ', x))
    df['answer'] = df['answer'].str.lower().str.strip().apply(lambda x: re.sub(r'\s+', ' ', x))

    return df