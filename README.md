# ABSA Restaurant Project

## What this project does
This project turns the SemEval-2014 restaurant review XML data into a simple aspect-based sentiment analysis (ABSA) workflow.

It does four main things:

1. parses the raw XML files into CSV tables
2. creates EDA tables and charts
3. runs aspect-term to category mapping experiments
4. runs aspect-term sentiment classification

This repo is focused on the restaurant dataset only and uses the full train split plus the gold test split.

## Quick Start

### 1. Open the project folder
Run all commands from the project root:

```bash
cd /path/to/absa_project
```

### 2. Create and activate a virtual environment
Using a virtual environment is the easiest way for a junior developer to avoid package conflicts.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If your machine uses `python` instead of `python3`, that is also fine.

### 3. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
python3 src/run_pipeline.py
```

This runs the full sequence:

1. data preparation
2. EDA
3. aspect generation evaluation
4. sentiment modelling

### 5. Check the outputs
When the run completes, look at:

- `data/processed/`
- `outputs/eda/`
- `outputs/models/`

## Input files
The project expects these two XML files in `data/raw/`:

- `Restaurants_Train_v2.xml`
- `Restaurants_Test_Gold.xml`

These are already included in this repo, so you do not need to download anything extra to run the current project.

If you replace them later, keep the same filenames and folder location.

## Project Structure

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
├── requirements.txt
└── README.md
```

## What each script does

### `src/config.py`
Defines shared paths and the fixed label order used across the project.

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
- a JSON file with headline findings

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
Runs all major scripts in order so you do not need to execute each file manually.

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
- polarity distribution tables
- category distribution tables
- charts such as train polarity distribution, category distribution, sentence length histogram, and aspects-per-sentence histogram

### In `outputs/models/`
You get:

- aspect generation evaluation files
- learned term-category lexicon
- sentiment classification report
- confusion matrix
- sentiment test predictions

## Recommended Order For A Beginner

1. run `python3 src/run_pipeline.py`
2. open `data/processed/dataset_summary.json`
3. inspect the files in `outputs/eda/`
4. inspect `outputs/models/aspect_generation_summary.json`
5. inspect `outputs/models/sentiment_summary.json`
6. read the files in `reports/`

That order helps you understand the project from raw data to outputs.

## Troubleshooting

### `python: command not found`
Use `python3` instead:

```bash
python3 src/run_pipeline.py
```

### `ModuleNotFoundError`
Your dependencies are not installed yet. Activate the virtual environment and run:

```bash
python3 -m pip install -r requirements.txt
```

### Import errors when running scripts
Run the command from the project root folder, not from inside `src/`.

## Current Scope And Limitations

What is working now:

- full XML parsing
- clean CSV export
- EDA output generation
- rule-based and hybrid aspect-category mapping evaluation
- aspect-level sentiment baseline using classic ML

What is intentionally limited:

- aspect generation is evaluated on single-category sentences only because the dataset does not directly link each aspect term to a category in multi-category sentences
- sentiment modelling uses gold aspect terms, not joint extraction from raw text
- the pipeline is focused on sentence-level ABSA, not full-review aggregation

What would require future work:

- joint aspect extraction and sentiment prediction from raw text alone
- transformer-based or instruction-tuned ABSA models
- better handling of the rare `conflict` class
- domain adaptation to newer restaurant reviews
- a production dashboard or deployment layer
