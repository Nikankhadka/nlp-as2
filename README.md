# ABSA Restaurant Project

## What this project does
This project turns the SemEval-2014 restaurant review XML data into a clean, runnable ABSA workflow.

It does four practical things:

1. parses the raw XML files into CSV tables
2. creates EDA outputs and summary charts
3. runs aspect-term to category mapping experiments
4. runs aspect-term sentiment classification

This version is set up for the restaurant dataset only and is focused on the full train and gold test splits.

---

## Project structure

```text
absa_project/
├── data/
│   ├── raw/
│   │   ├── Restaurants_Train_v2.xml
│   │   └── Restaurants_Test_Gold.xml
│   └── processed/
├── outputs/
│   ├── eda/
│   └── models/
├── reports/
├── src/
│   ├── config.py
│   ├── prepare_data.py
│   ├── eda.py
│   ├── aspect_generation.py
│   ├── sentiment_model.py
│   └── run_pipeline.py
└── README.md
```

---

## Input files you need

The project expects these two XML files in `data/raw/`:

- `Restaurants_Train_v2.xml`
- `Restaurants_Test_Gold.xml`

These are the SemEval-2014 restaurant train and gold test files.

### What is included right now
This package already includes both files in the correct folder.

### If you need to replace or re-add them later
Put the files back into `data/raw/` using the exact same filenames.

---

## How to run the project

Open a terminal in the `absa_project` folder.

### Run everything end to end

```bash
python src/run_pipeline.py
```

This runs the full sequence:

1. data preparation
2. EDA
3. aspect generation evaluation
4. sentiment modelling

---

## What each script does

### `src/config.py`
Stores project paths and the fixed category and polarity label order.

### `src/prepare_data.py`
Reads the raw XML files and creates clean CSV outputs.

Outputs include:

- sentence-level tables
- aspect-term tables
- aspect-category tables
- dataset summary JSON

### `src/eda.py`
Creates exploratory analysis outputs.

Outputs include:

- dataset overview CSV
- polarity distribution tables
- category distribution tables
- top aspect-term frequency table
- EDA charts as PNG files
- a small JSON file with headline findings

### `src/aspect_generation.py`
Evaluates two ways of mapping aspect terms into restaurant categories.

Methods:

- rule-based keyword baseline
- hybrid method using train-set co-occurrence signals plus fallback rules

Outputs include:

- baseline predictions
- hybrid predictions
- learned term-category lexicon
- summary JSON with evaluation metrics

### `src/sentiment_model.py`
Builds a sentiment baseline using TF-IDF features and logistic regression.

It predicts the polarity of each gold aspect term from the sentence context.

Outputs include:

- majority-class baseline summary
- model summary JSON
- classification report CSV
- confusion matrix CSV
- test predictions CSV

### `src/run_pipeline.py`
Runs all major scripts in order.

---

## What outputs are generated

### In `data/processed/`
You get clean CSV files such as:

- `train_sentences.csv`
- `train_aspect_terms.csv`
- `train_aspect_categories.csv`
- `test_sentences.csv`
- `test_aspect_terms.csv`
- `test_aspect_categories.csv`
- `dataset_summary.json`

### In `outputs/eda/`
You get:

- dataset summary tables
- top-term frequency table
- polarity and category distribution tables
- charts such as:
  - train polarity distribution
  - category distribution
  - sentence length histogram
  - aspects-per-sentence histogram

### In `outputs/models/`
You get:

- aspect generation evaluation files
- learned term-category lexicon
- sentiment classification report
- confusion matrix
- sentiment test predictions

---

## What is working now

These parts are working:

- full XML parsing
- clean CSV export
- EDA output generation
- rule-based and hybrid aspect-category mapping evaluation
- full-dataset aspect-level sentiment baseline

---

## What is partial

These parts are intentionally limited:

- aspect generation is evaluated on single-category sentences only because the dataset does not directly link each aspect term to a category in multi-category sentences
- sentiment modelling currently uses classic ML with gold aspect terms, not a joint extraction model
- the pipeline is focused on sentence-level ABSA, not full-review aggregation across multiple sentences

---

## What still depends on future work

These improvements would require extra work beyond the current package:

- joint aspect extraction and sentiment prediction from raw text alone
- transformer-based models or instruction-tuned ABSA models
- better handling of the rare `conflict` class
- domain adaptation to more recent restaurant reviews
- production-style review dashboard or deployment layer

---

## Limitations and prerequisites

### Prerequisites
This project expects a Python environment with common data science packages such as:

- pandas
- matplotlib
- scikit-learn

### Important limitations

- The dataset is old and English-only
- Labels are sentence-level, not full-review level
- The `anecdotes/miscellaneous` class is broad and noisy
- Positive sentiment is the majority class
- Conflict examples are very rare

---

## Recommended workflow for a beginner

1. run `python src/run_pipeline.py`
2. open `data/processed/dataset_summary.json`
3. inspect the files in `outputs/eda/`
4. inspect `outputs/models/aspect_generation_summary.json`
5. inspect `outputs/models/sentiment_summary.json`
6. read the files in `reports/`

That order will help you understand the project from raw data to final report material.
# nlp-as2
