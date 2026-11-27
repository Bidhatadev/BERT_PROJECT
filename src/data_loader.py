import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from config import MODEL_NAME, DATA_PATH, MAX_LENGTH


def load_data():

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["text", "label"])

    train_df, test_df = train_test_split(
        df,
        test_size=0.34,
        random_state=42,
        stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_data(data):

        return tokenizer(
            data["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    train_encodings = tokenize_data(train_df)
    test_encodings = tokenize_data(test_df)

    return (
        train_encodings,
        train_df["label"].tolist(),
        test_encodings,
        test_df["label"].tolist(),
        tokenizer
    )