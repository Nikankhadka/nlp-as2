from __future__ import annotations

import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from config import MODELS_DIR, POLARITY_ORDER, PROCESSED_DIR


def mark_aspect(sentence: str, term: str) -> str:
    sentence_lower = sentence.lower()
    term_lower = term.lower()
    start = sentence_lower.find(term_lower)
    if start == -1:
        return f'[ASPECT] {term} [/ASPECT] || {sentence}'
    end = start + len(term)
    return sentence[:start] + '[ASPECT] ' + sentence[start:end] + ' [/ASPECT]' + sentence[end:]


def build_features(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: f"term={row['term_normalized']} || context={mark_aspect(row['text'], row['term'])}",
        axis=1,
    )


def majority_baseline(y_train: pd.Series, y_test: pd.Series) -> dict:
    majority_label = y_train.value_counts().idxmax()
    predictions = pd.Series([majority_label] * len(y_test), index=y_test.index)
    return {
        'label': majority_label,
        'accuracy': round(float(accuracy_score(y_test, predictions)), 4),
        'macro_f1': round(float(f1_score(y_test, predictions, average='macro', zero_division=0)), 4),
    }


def main() -> None:
    train_aspects = pd.read_csv(PROCESSED_DIR / 'train_aspect_terms.csv')
    test_aspects = pd.read_csv(PROCESSED_DIR / 'test_aspect_terms.csv')

    X_train = build_features(train_aspects)
    y_train = train_aspects['polarity']
    X_test = build_features(test_aspects)
    y_test = test_aspects['polarity']

    baseline = majority_baseline(y_train, y_test)

    pipeline = Pipeline(
        [
            (
                'tfidf',
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                'clf',
                LogisticRegression(
                    max_iter=4000,
                    class_weight='balanced',
                    solver='lbfgs',
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    test_pred = pipeline.predict(X_test)

    test_metrics = {
        'accuracy': round(float(accuracy_score(y_test, test_pred)), 4),
        'macro_f1': round(float(f1_score(y_test, test_pred, average='macro', zero_division=0)), 4),
        'rows': int(len(y_test)),
    }

    report = classification_report(
        y_test,
        test_pred,
        labels=POLARITY_ORDER,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(MODELS_DIR / 'sentiment_classification_report.csv')

    confusion_df = pd.DataFrame(
        confusion_matrix(y_test, test_pred, labels=POLARITY_ORDER),
        index=[f'gold_{label}' for label in POLARITY_ORDER],
        columns=[f'pred_{label}' for label in POLARITY_ORDER],
    )
    confusion_df.to_csv(MODELS_DIR / 'sentiment_confusion_matrix.csv')

    predictions = test_aspects[['sentence_id', 'term', 'term_normalized', 'polarity']].copy()
    predictions['predicted_polarity'] = test_pred
    predictions.to_csv(MODELS_DIR / 'sentiment_test_predictions.csv', index=False)

    summary = {
        'majority_baseline': baseline,
        'logreg_tfidf_full_test': test_metrics,
        'model_note': 'Uses gold aspect terms and predicts aspect-term polarity from marked sentence context.',
    }
    with open(MODELS_DIR / 'sentiment_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print('Sentiment model outputs saved in outputs/models.')


if __name__ == '__main__':
    main()
