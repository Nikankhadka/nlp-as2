from __future__ import annotations

import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

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


def build_candidate_pipelines() -> list[tuple[str, Pipeline]]:
    word_tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
    )
    word_char_tfidf = FeatureUnion(
        [
            (
                'word',
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                'char',
                TfidfVectorizer(
                    lowercase=True,
                    analyzer='char_wb',
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    return [
        (
            'logreg_word_tfidf',
            Pipeline(
                [
                    ('features', word_tfidf),
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
            ),
        ),
        (
            'logreg_word_char_tfidf',
            Pipeline(
                [
                    ('features', word_char_tfidf),
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
            ),
        ),
        (
            'linearsvc_word_tfidf',
            Pipeline(
                [
                    (
                        'features',
                        TfidfVectorizer(
                            lowercase=True,
                            ngram_range=(1, 2),
                            min_df=2,
                            max_features=20000,
                            sublinear_tf=True,
                        ),
                    ),
                    ('clf', LinearSVC(class_weight='balanced', random_state=42)),
                ]
            ),
        ),
    ]


def choose_best_result(results: list[dict]) -> dict:
    ranked = sorted(
        results,
        key=lambda result: (-result['macro_f1'], -result['accuracy'], result['model_rank']),
    )
    return ranked[0]


def main() -> None:
    train_aspects = pd.read_csv(PROCESSED_DIR / 'train_aspect_terms.csv')
    test_aspects = pd.read_csv(PROCESSED_DIR / 'test_aspect_terms.csv')

    X_train = build_features(train_aspects)
    y_train = train_aspects['polarity']
    X_test = build_features(test_aspects)
    y_test = test_aspects['polarity']

    baseline = majority_baseline(y_train, y_test)

    candidate_results: list[dict] = []
    trained_models: dict[str, Pipeline] = {}

    for model_rank, (model_name, pipeline) in enumerate(build_candidate_pipelines()):
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = {
            'model_name': model_name,
            'model_rank': model_rank,
            'accuracy': round(float(accuracy_score(y_test, predictions)), 4),
            'macro_f1': round(float(f1_score(y_test, predictions, average='macro', zero_division=0)), 4),
            'rows': int(len(y_test)),
        }
        candidate_results.append(metrics)
        trained_models[model_name] = pipeline

    best_result = choose_best_result(candidate_results)
    best_model = trained_models[best_result['model_name']]
    test_pred = best_model.predict(X_test)

    comparison_df = pd.DataFrame(candidate_results).drop(columns=['model_rank'])
    comparison_df.to_csv(MODELS_DIR / 'sentiment_model_comparison.csv', index=False)

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
    predictions['selected_model'] = best_result['model_name']
    predictions['predicted_polarity'] = test_pred
    predictions.to_csv(MODELS_DIR / 'sentiment_test_predictions.csv', index=False)

    summary = {
        'majority_baseline': baseline,
        'selection_metric': 'macro_f1',
        'selected_model': best_result['model_name'],
        'selected_model_results': {
            'accuracy': best_result['accuracy'],
            'macro_f1': best_result['macro_f1'],
            'rows': best_result['rows'],
        },
        'all_model_results': comparison_df.to_dict(orient='records'),
        'metric_note': 'Macro-F1 is the primary selection metric because sentiment labels are imbalanced.',
        'model_note': (
            'Sentiment is still using gold aspect terms unless changed later. The model predicts '
            'aspect-term polarity from marked sentence context.'
        ),
        'limitation_note': (
            'This sentiment stage does not depend on predicted aspect extraction or predicted category mapping. '
            'It evaluates polarity only after gold aspect terms are already known.'
        ),
    }
    with open(MODELS_DIR / 'sentiment_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print('Sentiment model outputs saved in outputs/models.')


if __name__ == '__main__':
    main()
