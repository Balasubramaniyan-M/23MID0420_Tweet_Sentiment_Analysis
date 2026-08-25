# 23MID0420 — Tweet Sentiment Analysis

## MDI3003 Advanced Predictive Analytics — Laboratory Experiment 05

**Student:** Balasubramaniyan M  
**Registration Number:** 23MID0420  
**Semester:** Fall 2026–2027

---

## Project Overview

**Product and Brand Sentiment Prediction from Tweet Data Using Classical NLP and Machine Learning**

This project develops, compares, evaluates, and documents sentiment classifiers that predict the sentiment (positive, neutral, negative) expressed in tweets toward US airlines. The task is a supervised multiclass text classification problem using the Twitter US Airline Sentiment dataset.

## Quick Start

### Prerequisites
- Python 3.9+
- pip

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"

# Download dataset
# Place Tweets.csv from Kaggle into data/
```

### Run
Open and execute `notebooks/23MID0420_Lab05_TweetSentiment.ipynb` from top to bottom.

## Project Structure

```
├── data/                    # Dataset directory (Tweets.csv goes here)
├── notebooks/               # Main executable Jupyter notebook
├── src/                     # Reusable Python source modules
│   ├── utils.py             # Configuration and constants
│   ├── data_loader.py       # Data loading and splitting
│   ├── preprocessing.py     # Tweet text preprocessing
│   ├── baselines.py         # Dummy and VADER baselines
│   ├── models.py            # ML pipelines (LR, SVC, NB)
│   ├── evaluation.py        # Metrics and evaluation
│   ├── error_analysis.py    # Error inspection
│   ├── entity_analysis.py   # Airline entity analysis
│   └── visualization.py     # All plotting functions
├── outputs/                 # Generated outputs
│   ├── artifacts/           # Reproducibility manifests
│   ├── results/             # CSV results and predictions
│   ├── figures/             # Generated visualizations
│   └── models/              # Saved model pipeline
├── reports/                 # Technical report and docs
└── submission/              # Submission-ready artifacts
```

## Models

| # | Model | Type | Feature |
|---|---|---|---|
| 1 | DummyClassifier | Baseline | Most-frequent |
| 2 | VADER | Baseline | Lexicon-based |
| 3 | Logistic Regression | Learned | TF-IDF (1,2)-gram |
| 4 | LinearSVC | Learned | TF-IDF (1,2)-gram |
| 5 | MultinomialNB | Learned | TF-IDF (1,2)-gram |

## Reproducibility

- **Seed:** 42
- **Split:** Stratified 70/10/20 (train/val/test)
- **CV:** 5-fold StratifiedKFold
- **Selection:** Macro F1 on training-only CV
- **Leakage prevention:** All TF-IDF inside sklearn Pipelines

## License

MIT License — see [LICENSE](LICENSE)

## Academic Integrity

AI-assisted tools were used for code scaffolding and documentation. All experimental results are genuine outputs from executed code. No metrics or results have been fabricated.

---
*MDI3003 — Advanced Predictive Analytics, Fall 2026–2027*
