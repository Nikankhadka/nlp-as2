# Phase 1: Coding and Analysis Summary

## 1. Step-by-step plan followed

This is the exact order used in the rebuild:

1. inspect what files were actually available in the workspace
2. recover the full restaurant dataset files
3. parse the XML into sentence-level, aspect-term, and aspect-category tables
4. run EDA and save charts and summary tables
5. evaluate the current rule-based aspect-category mapping
6. add a stronger hybrid aspect-category mapping approach
7. train and evaluate a stronger aspect-level sentiment baseline
8. separate what is fully working from what is partial
9. write README and report materials only after the analysis outputs existed

## 2. Dataset suitability evaluation

### Overall judgement
The SemEval-2014 restaurant dataset is **partially suitable** for the practical restaurant review intelligence problem.

It is strong enough for:

- ABSA prototyping
- controlled evaluation
- category-level restaurant insight summaries
- comparing simple and stronger modelling approaches

It is not fully suitable for a production-style restaurant intelligence system because several parts of the real business problem are missing.

### Why it is suitable
The dataset is useful because it gives sentence text plus two kinds of labels:

- explicit aspect terms such as `food`, `staff`, `prices`, `atmosphere`
- coarse aspect categories such as `food`, `service`, `price`, `ambience`, and `anecdotes/miscellaneous`

That makes it suitable for testing whether a pipeline can:

- identify what restaurant issue is being discussed
- estimate the sentiment attached to that issue
- roll those outputs into a simple operational summary

The full restaurant split loaded in this package contains:

- 3,041 train sentences
- 800 test sentences
- 3,693 train aspect-term labels
- 1,134 test aspect-term labels
- 3,713 train aspect-category labels
- 1,025 test aspect-category labels

### Why it is only partially suitable
There are four important mismatches between the dataset and the real project goal.

1. The data is sentence-level, not full-review intelligence.
2. Category labels are coarse.
3. Aspect terms are explicit only.
4. The dataset is old and narrow.

### Final suitability statement
For this project, the SemEval-2014 restaurant dataset is appropriate as a benchmark and prototype dataset, but limited as a full representation of real restaurant review intelligence.

## 3. Data preparation

### What files are available now
The current project contains the full restaurant XML files:

- `data/raw/Restaurants_Train_v2.xml`
- `data/raw/Restaurants_Test_Gold.xml`

### What was rebuilt
The following were rebuilt from the raw XML:

- sentence tables
- aspect-term tables
- aspect-category tables
- full dataset summary JSON

### Full-dataset handling
The project now works directly from the official restaurant train XML and official restaurant gold test XML. This is the path that should be trusted for final tables and report claims.

## 4. EDA findings

### Headline structure
The dataset is short, dense, and strongly skewed toward positive sentiment.

From the parsed full train split:

- average tokens per sentence: 12.76
- average aspect terms per sentence: 1.21
- median tokens per sentence: 12
- median aspect terms per sentence: 1
- multi-category sentence share: 18.94%
- zero-aspect-term sentence share: 33.54%

### Polarity imbalance
Train aspect-term labels are distributed as:

- positive: 2,164
- negative: 805
- neutral: 633
- conflict: 91

### Category imbalance
Train aspect-category labels are distributed as:

- food: 1,232
- anecdotes/miscellaneous: 1,132
- service: 597
- ambience: 431
- price: 321

## 5. Aspect generation evaluation

### What the current rule-based approach does
The baseline rule-based mapper uses seed keywords to map aspect terms into one of the five restaurant categories.

### Core limitation in this dataset
The dataset does not directly tell us which category belongs to which aspect term in multi-category sentences. To handle this honestly, aspect-generation evaluation was done on single-category sentences only.

### Baseline versus improved method
Evaluation set sizes:

- train eval rows: 2,477
- test eval rows: 709

#### Rule-based baseline
- accuracy: 0.7941
- macro F1: 0.5588

#### Hybrid method
- accuracy: 0.7870
- macro F1: 0.6284

The hybrid method lowers raw accuracy slightly, but improves macro F1 by about 0.07, which makes it more balanced across categories.

## 6. Sentiment baseline evaluation

### Majority baseline
- accuracy: 0.6420
- macro F1: 0.1955

### TF-IDF + logistic regression
- accuracy: 0.6526
- macro F1: 0.4686

This is only a small gain in accuracy, but a much stronger gain in macro F1, which is the more useful metric for this imbalanced problem.

## 7. Code deliverable status

### Working now
- full XML parsing
- clean CSV export
- EDA output generation
- rule-based and hybrid aspect-category mapping evaluation
- full-dataset aspect-level sentiment baseline

### Partial
- aspect generation evaluation is limited by the dataset label design
- sentiment model uses gold aspect terms rather than full extraction from raw text

### Future work
- full end-to-end aspect extraction
- transformer baselines
- review-level aggregation
- evaluation on newer review sources

## 8. Practical use and ethics

The outputs are suitable for lightweight restaurant insight summaries, not for fully automated business decisions. They can help highlight recurring strengths and weaknesses in food, service, price, and ambience, but they should not be treated as complete truth. The dataset is old, category labels are coarse, and minority classes are weakly represented. Human review is still important when decisions affect staff, vendors, or public claims.
