# ABSA Restaurant Project

## What this project does
This project turns the SemEval-2014 restaurant review XML data into a simple aspect-based sentiment analysis (ABSA) workflow.

It does five main things:

1. parses the raw XML files into CSV tables
2. creates EDA tables and charts
3. runs a raw aspect-term extraction baseline
4. runs aspect-term to category mapping experiments
5. runs aspect-term sentiment classification

This repo is focused on the restaurant dataset only and uses the full train split plus the gold test split.

## Quick Start

Run all commands from the project root:

```bash
cd /path/to/absa_project
python3 -m pip install -r requirements.txt
python3 src/run_pipeline.py
```

If your machine uses `python` instead of `python3`, use:

```bash
python -m pip install -r requirements.txt
python src/run_pipeline.py
```

### Colab / notebook version

```python
!pip install -r requirements.txt
!python src/run_pipeline.py
```

### What the pipeline runs
The full pipeline runs these steps in order:

1. data preparation
2. EDA
3. aspect extraction baseline
4. aspect generation evaluation
5. sentiment modelling

### Check the outputs
When the run completes, look at:

- `data/processed/`
- `outputs/eda/`
- `outputs/models/`

## Input files
The project expects these two XML files in `data/raw/`:

- `Restaurants_Train_v2.xml`
- `Restaurants_Test_Gold.xml`

These files are already included in this repo, so you do not need to download anything extra to run the current project.

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
│   ├── aspect_extraction.py
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

### `src/eda.py`
Creates exploratory analysis outputs.

### `src/aspect_extraction.py`
Builds a simple raw aspect-term extraction baseline and evaluates it with exact normalized matching.

### `src/aspect_generation.py`
Evaluates two ways of mapping aspect terms into restaurant categories.

### `src/sentiment_model.py`
Compares balanced linear sentiment baselines using gold aspect terms and selects the best model by macro-F1.

### `src/run_pipeline.py`
Runs all major scripts in order so you do not need to execute each file manually.

## Recommended Order For A Beginner

1. run `python3 -m pip install -r requirements.txt`
2. run `python3 src/run_pipeline.py`
3. open `data/processed/dataset_summary.json`
4. inspect the files in `outputs/eda/`
5. inspect `outputs/models/aspect_extraction_summary.json`
6. inspect `outputs/models/aspect_generation_summary.json`
7. inspect `outputs/models/sentiment_summary.json`

## Evaluation notes

- `macro_f1` is the primary metric for model selection because the dataset is imbalanced.
- Raw aspect extraction is evaluated with exact normalized match. This is a strict metric and may underestimate partially correct matches.
- Aspect extraction and aspect-category mapping are separate tasks in this repo.
- Sentiment is still using gold aspect terms unless changed later.

## Troubleshooting

### `python: command not found`
Use `python3` instead.

### `ModuleNotFoundError`
Install the dependencies first:

```bash
python3 -m pip install -r requirements.txt
```

### Import errors when running scripts
Run the command from the project root folder, not from inside `src/`.
