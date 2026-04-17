from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from config import MODELS_DIR, PROCESSED_DIR
from prepare_data import normalize_term, tokenize

STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'for', 'from', 'had', 'has',
    'have', 'he', 'her', 'here', 'him', 'his', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'me',
    'my', 'of', 'on', 'or', 'our', 'she', 'so', 'that', 'the', 'their', 'them', 'there', 'they',
    'this', 'to', 'too', 'us', 'was', 'we', 'were', 'with', 'you', 'your',
}

NON_ASPECT_WORDS = {
    'amazing', 'awful', 'bad', 'better', 'best', 'delicious', 'excellent', 'friendly', 'good',
    'great', 'horrible', 'love', 'loved', 'nice', 'poor', 'slow', 'terrible', 'wonderful',
}

MIN_SINGLE_TERM_COUNT = 2
MIN_MULTI_TERM_COUNT = 2
MIN_HEADWORD_COUNT = 3
MAX_MULTIWORD_LENGTH = 3


def build_term_resources(train_aspects: pd.DataFrame) -> tuple[set[str], set[str], set[str]]:
    term_counts = Counter(train_aspects['term_normalized'])
    headword_counts = Counter(train_aspects['term_normalized'].apply(lambda term: term.split()[-1]))

    single_word_terms = {
        term for term, count in term_counts.items()
        if count >= MIN_SINGLE_TERM_COUNT and len(term.split()) == 1
    }
    multi_word_terms = {
        term for term, count in term_counts.items()
        if count >= MIN_MULTI_TERM_COUNT and 1 < len(term.split()) <= MAX_MULTIWORD_LENGTH
    }
    frequent_headwords = {
        headword for headword, count in headword_counts.items()
        if count >= MIN_HEADWORD_COUNT and headword not in STOPWORDS
    }
    return single_word_terms, multi_word_terms, frequent_headwords


def extract_candidates(
    text: str,
    single_word_terms: set[str],
    multi_word_terms: set[str],
    frequent_headwords: set[str],
) -> list[tuple[str, str]]:
    tokens = tokenize(text)
    predictions: list[tuple[str, str]] = []
    seen: set[str] = set()

    for size in range(MAX_MULTIWORD_LENGTH, 1, -1):
        for start in range(0, len(tokens) - size + 1):
            phrase = ' '.join(tokens[start:start + size])
            if phrase in multi_word_terms and phrase not in seen:
                predictions.append((phrase, 'train_lexicon_multiword'))
                seen.add(phrase)

    for token in tokens:
        if token in single_word_terms and token not in seen:
            predictions.append((token, 'train_lexicon_single'))
            seen.add(token)

    for token in tokens:
        if token in seen:
            continue
        if token in STOPWORDS or token in NON_ASPECT_WORDS or len(token) < 3:
            continue
        if token in frequent_headwords:
            predictions.append((token, 'noun_like_headword'))
            seen.add(token)

    return predictions


def evaluate_predictions(
    sentences: pd.DataFrame,
    gold_terms: pd.DataFrame,
    single_word_terms: set[str],
    multi_word_terms: set[str],
    frequent_headwords: set[str],
) -> tuple[pd.DataFrame, dict]:
    gold_lookup = (
        gold_terms.groupby('sentence_id')['term_normalized']
        .apply(lambda values: sorted(set(values)))
        .to_dict()
    )

    rows: list[dict] = []
    true_positive = 0
    false_positive = 0
    false_negative = 0
    predicted_term_total = 0

    for row in sentences.itertuples(index=False):
        predictions = extract_candidates(row.text, single_word_terms, multi_word_terms, frequent_headwords)
        predicted_terms = sorted({term for term, _ in predictions})
        predicted_term_total += len(predicted_terms)
        gold_sentence_terms = set(gold_lookup.get(row.sentence_id, []))

        true_positive += len(set(predicted_terms).intersection(gold_sentence_terms))
        false_positive += len(set(predicted_terms) - gold_sentence_terms)
        false_negative += len(gold_sentence_terms - set(predicted_terms))

        if predictions:
            for term, source in predictions:
                rows.append(
                    {
                        'sentence_id': row.sentence_id,
                        'text': row.text,
                        'predicted_term_normalized': term,
                        'prediction_source': source,
                        'is_exact_gold_match': int(term in gold_sentence_terms),
                        'gold_terms_normalized': ' | '.join(sorted(gold_sentence_terms)),
                    }
                )
        else:
            rows.append(
                {
                    'sentence_id': row.sentence_id,
                    'text': row.text,
                    'predicted_term_normalized': '',
                    'prediction_source': 'no_prediction',
                    'is_exact_gold_match': 0,
                    'gold_terms_normalized': ' | '.join(sorted(gold_sentence_terms)),
                }
            )

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    summary = {
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1': round(float(f1), 4),
        'sentences': int(len(sentences)),
        'gold_terms': int(len(gold_terms)),
        'predicted_terms': int(predicted_term_total),
        'average_predicted_terms_per_sentence': round(predicted_term_total / len(sentences), 4),
        'match_rule': 'exact_normalized_match',
        'metric_note': (
            'Exact normalized match is a strict extraction metric. It may underestimate partially correct '
            'predictions when a predicted span overlaps a gold aspect but is not an exact normalized match.'
        ),
        'method_note': (
            'The extraction baseline uses short train-lexicon matches and conservative noun-like headword '
            'fallbacks. It is intended as a simple, explainable baseline rather than a full sequence tagger.'
        ),
        'limitation_note': (
            'This extraction stage is evaluated separately from category mapping and sentiment. Sentiment still '
            'uses gold aspect terms unless changed later.'
        ),
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    train_aspects = pd.read_csv(PROCESSED_DIR / 'train_aspect_terms.csv')
    test_aspects = pd.read_csv(PROCESSED_DIR / 'test_aspect_terms.csv')
    test_sentences = pd.read_csv(PROCESSED_DIR / 'test_sentences.csv')

    single_word_terms, multi_word_terms, frequent_headwords = build_term_resources(train_aspects)
    predictions, summary = evaluate_predictions(
        test_sentences,
        test_aspects,
        single_word_terms,
        multi_word_terms,
        frequent_headwords,
    )

    predictions.to_csv(MODELS_DIR / 'aspect_extraction_predictions.csv', index=False)
    with open(MODELS_DIR / 'aspect_extraction_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print('Aspect extraction baseline outputs saved in outputs/models.')


if __name__ == '__main__':
    main()
