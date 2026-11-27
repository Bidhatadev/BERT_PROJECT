
# BERT Sentiment Analysis

## About the Project
This project uses BERT (Bidirectional Encoder Representations from Transformers) to perform sentiment analysis on customer reviews.

The model analyzes text reviews and predicts a rating and its corresponding sentiment.

## Features
- BERT-based sentiment analysis
- Text review prediction
- Rating prediction (1–5)
- Sentiment classification
- Python implementation

## Project Structure

```text
BERT_PROJECT/
│
├── data/
│   └── reviews.csv
│
├── models/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── predict.py
│   └── train.py
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Technologies Used
- Python
- BERT
- Transformers
- PyTorch
- Pandas
- Scikit-learn

## Installation

Clone the repository:

```bash
git clone https://github.com/Bidhatadev/BERT_PROJECT.git
```

Navigate to the project:

```bash
cd BERT_PROJECT
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Train the model:

```bash
python src/train.py
```

Run prediction:

```bash
python src/predict.py
```

## Example Prediction

Input:
"This food was amazing and delicious!"

Output:
- Rating: 5
- Sentiment: Very Satisfied

## Author
Bidhata Dev
