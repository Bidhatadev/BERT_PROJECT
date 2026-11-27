
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_PATH


# Load trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


# Sentiment and rating mapping
sentiment_mapping = {
    0: {
        "rating": "1-2 Stars",
        "sentiment": "Negative"
    },
    1: {
        "rating": "3 Stars",
        "sentiment": "Neutral"
    },
    2: {
        "rating": "4-5 Stars",
        "sentiment": "Positive"
    }
}


# Prediction function
def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    result = sentiment_mapping[prediction]

    print("\n========== BERT PREDICTION ==========")
    print("Customer Review :", text)
    print("Rating          :", result["rating"])
    print("Sentiment       :", result["sentiment"])
    print("====================================")


# Main program
if __name__ == "__main__":

    print("\nBERT CUSTOMER REVIEW SENTIMENT PREDICTION")

    while True:

        review = input("\nEnter customer review (or type exit): ")

        if review.lower().strip() == "exit":
            print("Program closed.")
            break

        if not review.strip():
            print("Please enter a review.")
            continue

        predict_sentiment(review)
