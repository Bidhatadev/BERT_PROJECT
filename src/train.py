import torch

from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from config import *
from data_loader import load_data


class ReviewDataset(Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        item = {
            key: torch.tensor(val[index])
            for key, val in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item


def train_model():

    train_x, train_y, test_x, test_y, tokenizer = load_data()

    train_dataset = ReviewDataset(train_x, train_y)
    test_dataset = ReviewDataset(test_x, test_y)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer
    )

    trainer.train()

    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    print("BERT fine-tuning completed!")


if __name__ == "__main__":
    train_model()