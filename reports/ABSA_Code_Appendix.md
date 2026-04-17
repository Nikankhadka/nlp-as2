# ABSA Code Appendix

## Appendix purpose
This appendix explains the code structure and practical logic used in the restaurant ABSA project. It is written to help a marker or beginner reader understand what each script does, how the data moves through the pipeline, and which parts are complete or limited.

## 1. Files and responsibilities

### `src/config.py`
Stores shared project paths and the fixed label order for categories and polarity classes.

### `src/prepare_data.py`
Reads the SemEval restaurant XML files and exports three clean CSV table types for both train and test:

- sentence tables
- aspect-term tables
- aspect-category tables

It also writes `dataset_summary.json` so the main dataset counts are easy to inspect.

### `src/eda.py`
Builds the exploratory analysis outputs used in the report:

- dataset overview table
- polarity distribution table
- category distribution table
- top aspect-term table
- PNG charts
- `eda_findings.json`

### `src/aspect_generation.py`
Evaluates aspect-term to category mapping with two approaches:

- a readable rule-based baseline
- a hybrid mapper that combines train-set evidence with fallback rules

### `src/sentiment_model.py`
Trains a TF-IDF plus logistic regression baseline for aspect-term sentiment classification. It uses the gold aspect terms already provided in the dataset and predicts their polarity from marked sentence context.

### `src/run_pipeline.py`
Runs the full project in order:

1. data preparation
2. EDA
3. aspect generation evaluation
4. sentiment modelling

## 2. Data flow

### Step 1. Raw XML
The raw data is stored in:

- `data/raw/Restaurants_Train_v2.xml`
- `data/raw/Restaurants_Test_Gold.xml`

### Step 2. Processed tables
`prepare_data.py` converts the XML into structured CSV files:

- `train_sentences.csv`
- `train_aspect_terms.csv`
- `train_aspect_categories.csv`
- `test_sentences.csv`
- `test_aspect_terms.csv`
- `test_aspect_categories.csv`
- `dataset_summary.json`

### Step 3. Analysis outputs
`eda.py`, `aspect_generation.py`, and `sentiment_model.py` create the outputs used for analysis and reporting under `outputs/eda/` and `outputs/models/`.

## 3. Aspect generation logic

### Baseline
The rule-based baseline uses seed keywords and simple fallbacks. Example patterns include:

- `staff`, `waiter`, `delivery` -> service
- `price`, `cost`, `bill` -> price
- `atmosphere`, `decor`, `music` -> ambience
- common food terms -> food

This baseline is easy to read and explain, which makes it useful as a reference point.

### Improved hybrid mapper
The hybrid mapper adds train-set evidence:

- normalized term-level category tendencies
- headword-level fallback
- seed keyword fallback
- regex fallback for obvious category words

This keeps the code simple while making better use of the labelled data.

### Important evaluation note
The dataset does not directly align each aspect term to a category in multi-category sentences. To avoid inventing labels, evaluation is restricted to single-category sentences only.

## 4. Sentiment modelling logic

The sentiment script uses a lightweight classical baseline instead of a larger neural model. This is a deliberate design choice. The goal is to keep the full project reproducible, easy to inspect, and realistic for coursework.

Main steps:

1. load train and test aspect-term tables
2. mark the aspect term inside the sentence
3. combine the marked context with the normalized term
4. build TF-IDF features
5. train logistic regression with class balancing
6. evaluate on the official gold test split

Saved outputs include:

- classification report
- confusion matrix
- test predictions
- summary JSON

## 5. Generated outputs worth checking first

### Core summaries
- `data/processed/dataset_summary.json`
- `outputs/eda/eda_findings.json`
- `outputs/models/aspect_generation_summary.json`
- `outputs/models/sentiment_summary.json`

### Useful tables
- `outputs/eda/dataset_overview.csv`
- `outputs/eda/top_train_aspect_terms.csv`
- `outputs/models/learned_term_category_lexicon.csv`
- `outputs/models/sentiment_classification_report.csv`

### Useful charts
- `outputs/eda/train_term_polarity_distribution.png`
- `outputs/eda/train_category_distribution.png`
- `outputs/eda/train_sentence_length_histogram.png`
- `outputs/eda/train_aspects_per_sentence_histogram.png`

## 6. Current code status

### Fully working
- full XML parsing
- processed CSV export
- EDA generation
- aspect-generation evaluation
- sentiment baseline training and evaluation

### Partial
- aspect-category evaluation only covers single-category sentences
- sentiment model assumes gold aspect terms are already known

### Not yet implemented
- joint extraction from raw sentences without gold aspect terms
- transformer ABSA modelling
- review-level aggregation beyond sentence scope

## 7. Reproducibility note
The pipeline is deterministic in file structure and output generation because it reads the same XML files and writes fixed output locations.

## 8. Command to reproduce the current outputs

```bash
python src/run_pipeline.py
```
