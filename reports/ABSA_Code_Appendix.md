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

### `src/aspect_extraction.py`
Builds a simple raw aspect-term extraction baseline from sentence text and compares predicted terms against the gold aspect terms with exact normalized matching.

### `src/aspect_generation.py`
Evaluates aspect-term to category mapping with two approaches:

- a readable rule-based baseline
- a hybrid mapper that combines smoothed train-set evidence with fallback rules

### `src/sentiment_model.py`
Compares balanced linear baselines for aspect-term sentiment classification. It still uses the gold aspect terms already provided in the dataset and predicts their polarity from marked sentence context.

### `src/run_pipeline.py`
Runs the full project in order:

1. data preparation
2. EDA
3. raw aspect extraction baseline
4. aspect generation evaluation
5. sentiment modelling

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
`eda.py`, `aspect_extraction.py`, `aspect_generation.py`, and `sentiment_model.py` create the outputs used for analysis and reporting under `outputs/eda/` and `outputs/models/`.

## 3. Raw aspect extraction logic

The extraction baseline predicts aspect terms directly from raw sentence text before any category mapping step.

Main logic:

- match short aspect terms seen often enough in the training lexicon
- allow short multi-word matches when they repeat in training
- add conservative noun-like headword fallbacks for frequent aspect heads
- deduplicate normalized predictions within each sentence

Evaluation uses exact normalized matching against the gold aspect terms.

Important note:

- exact normalized match is a strict metric
- it may underestimate partial matches where the predicted phrase overlaps the gold term but is not an exact normalized match

## 4. Aspect generation logic

### Baseline
The rule-based baseline uses seed keywords and simple fallbacks. Example patterns include:

- `staff`, `waiter`, `delivery` -> service
- `price`, `cost`, `bill` -> price
- `atmosphere`, `decor`, `music` -> ambience
- common food terms -> food

This baseline is easy to read and explain, which makes it useful as a reference point.

### Improved hybrid mapper
The hybrid mapper adds train-set evidence:

- smoothed support-aware term-level category scoring
- minimum-support checks before trusting learned scores
- headword-level fallback
- seed keyword fallback
- regex fallback for obvious category words
- low-confidence majority-category fallback when no stronger evidence is available

This keeps the code simple while making better use of the labelled data.

### Important evaluation note
The dataset does not directly align each aspect term to a category in multi-category sentences. To avoid inventing labels, evaluation is restricted to single-category sentences only.

## 5. Sentiment modelling logic

The sentiment script compares several lightweight classical baselines instead of a larger neural model. This is a deliberate design choice. The goal is to keep the full project reproducible, easy to inspect, and realistic for coursework.

Main steps:

1. load train and test aspect-term tables
2. mark the aspect term inside the sentence
3. combine the marked context with the normalized term
4. build candidate TF-IDF feature sets
5. train balanced linear models
6. select the best model by macro-F1
7. evaluate the selected model on the official gold test split

Important note:

- sentiment is still using gold aspect terms unless changed later

Saved outputs include:

- model comparison table
- classification report
- confusion matrix
- test predictions
- summary JSON

## 6. Generated outputs worth checking first

### Core summaries
- `data/processed/dataset_summary.json`
- `outputs/eda/eda_findings.json`
- `outputs/models/aspect_extraction_summary.json`
- `outputs/models/aspect_generation_summary.json`
- `outputs/models/sentiment_summary.json`

### Useful tables
- `outputs/eda/dataset_overview.csv`
- `outputs/eda/top_train_aspect_terms.csv`
- `outputs/models/aspect_extraction_predictions.csv`
- `outputs/models/learned_term_category_lexicon.csv`
- `outputs/models/sentiment_model_comparison.csv`
- `outputs/models/sentiment_classification_report.csv`

### Useful charts
- `outputs/eda/train_term_polarity_distribution.png`
- `outputs/eda/train_category_distribution.png`
- `outputs/eda/train_sentence_length_histogram.png`
- `outputs/eda/train_aspects_per_sentence_histogram.png`

## 7. Current code status

### Fully working
- full XML parsing
- processed CSV export
- EDA generation
- raw aspect extraction baseline
- aspect-generation evaluation
- sentiment baseline training and evaluation

### Partial
- aspect-category evaluation only covers single-category sentences
- sentiment model assumes gold aspect terms are already known

### Not yet implemented
- transformer ABSA modelling
- review-level aggregation beyond sentence scope

## 8. Reproducibility note
The pipeline is deterministic in file structure and output generation because it reads the same XML files and writes fixed output locations.

## 9. Command to reproduce the current outputs

```bash
python src/run_pipeline.py
```
