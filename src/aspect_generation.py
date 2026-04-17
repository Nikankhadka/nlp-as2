from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from config import CATEGORY_ORDER, MODELS_DIR, PROCESSED_DIR
from prepare_data import normalize_term

SEED_KEYWORDS = {
    'food': {
        'food', 'dish', 'dishes', 'meal', 'meals', 'menu', 'pizza', 'sushi', 'wine', 'drinks',
        'dessert', 'desserts', 'bread', 'salad', 'steak', 'pasta', 'soup', 'burger', 'sandwich',
        'appetizer', 'appetizers', 'fish', 'portion', 'portions', 'taste', 'flavor', 'flavour',
        'fresh', 'coffee', 'beer', 'cocktail', 'roll', 'rolls', 'bagel', 'bagels', 'risotto',
    },
    'service': {
        'service', 'staff', 'waiter', 'waiters', 'waitress', 'waitresses', 'hostess', 'manager',
        'delivery', 'server', 'servers', 'reservation', 'reservations', 'check', 'attentive',
    },
    'price': {
        'price', 'prices', 'priced', 'expensive', 'cheap', 'affordable', 'value', 'worth', 'cost',
        'bill', 'bargain', 'overpriced', 'reasonable', 'money',
    },
    'ambience': {
        'atmosphere', 'ambience', 'ambiance', 'decor', 'design', 'music', 'crowded', 'space',
        'room', 'rooms', 'vibe', 'romantic', 'cozy', 'garden', 'chairs', 'seats', 'noisy', 'quiet',
        'loud', 'interior',
    },
    'anecdotes/miscellaneous': {
        'place', 'restaurant', 'spot', 'experience', 'experiences', 'issue', 'issues', 'option',
        'options', 'thing', 'things', 'deal', 'deals',
    },
}

REGEX_PATTERNS = [
    ('price', re.compile(r'price|cost|bill|value|cheap|expensive|overpriced|affordable')),
    ('service', re.compile(r'staff|service|waiter|delivery|manager|reservation|server|hostess')),
    ('ambience', re.compile(r'atmosphere|ambience|ambiance|decor|music|seat|room|space|vibe')),
    ('anecdotes/miscellaneous', re.compile(r'place|restaurant|experience|issue|option|deal')),
]

SMOOTHING_ALPHA = 1.0
TERM_MIN_SUPPORT = 3
HEADWORD_MIN_SUPPORT = 5


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_aspects = pd.read_csv(PROCESSED_DIR / 'train_aspect_terms.csv')
    test_aspects = pd.read_csv(PROCESSED_DIR / 'test_aspect_terms.csv')
    train_categories = pd.read_csv(PROCESSED_DIR / 'train_aspect_categories.csv')
    test_categories = pd.read_csv(PROCESSED_DIR / 'test_aspect_categories.csv')
    return train_aspects, test_aspects, train_categories, test_categories


def build_single_category_eval_frame(aspects: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
    category_counts = categories.groupby('sentence_id')['category'].nunique().rename('category_count')
    single_category_sentences = category_counts[category_counts == 1].index
    single_category_map = (
        categories[categories['sentence_id'].isin(single_category_sentences)]
        .drop_duplicates(['sentence_id', 'category'])
        .set_index('sentence_id')['category']
    )
    eval_frame = aspects[aspects['sentence_id'].isin(single_category_sentences)].copy()
    eval_frame['gold_category'] = eval_frame['sentence_id'].map(single_category_map)
    return eval_frame


def seed_or_regex_match(term: str) -> tuple[str, str] | None:
    norm = normalize_term(term)
    tokens = set(norm.split())

    for category, keywords in SEED_KEYWORDS.items():
        if norm in keywords or tokens.intersection(keywords):
            return category, 'seed_keyword'

    for category, pattern in REGEX_PATTERNS:
        if pattern.search(norm):
            return category, f'regex_{category.replace("/", "_")}'

    return None


def baseline_rule_predict(term: str) -> tuple[str, str]:
    matched = seed_or_regex_match(term)
    if matched is not None:
        return matched

    norm = normalize_term(term)
    tokens = set(norm.split())
    if any(token.endswith(('ing', 'ed')) for token in tokens):
        return 'service', 'verbish_fallback'

    return 'food', 'default_food_fallback'


class HybridAspectMapper:
    def __init__(self, smoothing_alpha: float = SMOOTHING_ALPHA) -> None:
        self.smoothing_alpha = smoothing_alpha
        self.term_category_scores: dict[str, dict[str, float]] = {}
        self.headword_scores: dict[str, dict[str, float]] = {}
        self.term_support: dict[str, int] = {}
        self.headword_support: dict[str, int] = {}
        self.majority_category = 'food'

    @staticmethod
    def headword(term: str) -> str:
        pieces = normalize_term(term).split()
        return pieces[-1] if pieces else ''

    def fit(self, train_eval_frame: pd.DataFrame) -> None:
        term_counts: dict[str, Counter] = defaultdict(Counter)
        headword_counts: dict[str, Counter] = defaultdict(Counter)

        for row in train_eval_frame.itertuples(index=False):
            term_counts[row.term_normalized][row.gold_category] += 1
            headword = self.headword(row.term_normalized)
            if headword:
                headword_counts[headword][row.gold_category] += 1

        self.term_support = {key: int(sum(counter.values())) for key, counter in term_counts.items()}
        self.headword_support = {key: int(sum(counter.values())) for key, counter in headword_counts.items()}
        self.term_category_scores = self._to_scores(term_counts)
        self.headword_scores = self._to_scores(headword_counts)
        self.majority_category = train_eval_frame['gold_category'].value_counts().idxmax()

    def _to_scores(self, counts: dict[str, Counter]) -> dict[str, dict[str, float]]:
        scores: dict[str, dict[str, float]] = {}
        category_count = len(CATEGORY_ORDER)
        alpha = self.smoothing_alpha

        for key, counter in counts.items():
            scores[key] = {}
            key_total = sum(counter.values())
            smoothed_key_total = key_total + alpha * category_count

            for category in CATEGORY_ORDER:
                probability = (counter.get(category, 0) + alpha) / smoothed_key_total
                scores[key][category] = round(float(probability), 6)

        return scores

    def _predict_from_learned(
        self,
        key: str,
        score_lookup: dict[str, dict[str, float]],
        support_lookup: dict[str, int],
        minimum_support: int,
        source_name: str,
    ) -> tuple[str, str] | None:
        if not key:
            return None
        if support_lookup.get(key, 0) < minimum_support:
            return None
        if key not in score_lookup:
            return None
        score_map = score_lookup[key]
        return max(score_map, key=score_map.get), source_name

    def predict(self, term: str) -> tuple[str, str]:
        norm = normalize_term(term)

        learned_prediction = self._predict_from_learned(
            norm,
            self.term_category_scores,
            self.term_support,
            TERM_MIN_SUPPORT,
            'term_support_score',
        )
        if learned_prediction is not None:
            return learned_prediction

        head = self.headword(norm)
        learned_headword_prediction = self._predict_from_learned(
            head,
            self.headword_scores,
            self.headword_support,
            HEADWORD_MIN_SUPPORT,
            'headword_support_score',
        )
        if learned_headword_prediction is not None:
            return learned_headword_prediction

        matched = seed_or_regex_match(term)
        if matched is not None:
            return matched

        return self.majority_category, 'low_confidence_majority_fallback'


def evaluate_method(eval_frame: pd.DataFrame, predictor) -> tuple[pd.DataFrame, dict]:
    frame = eval_frame.copy()
    predictions = frame['term'].apply(predictor)
    frame['predicted_category'] = [item[0] for item in predictions]
    frame['prediction_source'] = [item[1] for item in predictions]
    frame['is_low_confidence'] = frame['prediction_source'].eq('low_confidence_majority_fallback')

    accuracy = accuracy_score(frame['gold_category'], frame['predicted_category'])
    report = classification_report(
        frame['gold_category'],
        frame['predicted_category'],
        labels=CATEGORY_ORDER,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = report['macro avg']['f1-score']

    metrics = {
        'rows': int(frame.shape[0]),
        'accuracy': round(float(accuracy), 4),
        'macro_f1': round(float(macro_f1), 4),
        'coverage': 1.0,
        'low_confidence_predictions': int(frame['is_low_confidence'].sum()),
    }
    return frame, metrics


def build_source_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    breakdown = (
        frame.groupby('prediction_source')
        .agg(
            rows=('prediction_source', 'size'),
            accuracy=('predicted_category', lambda col: float((col == frame.loc[col.index, 'gold_category']).mean())),
            low_confidence=('is_low_confidence', 'sum'),
        )
        .reset_index()
        .sort_values(['rows', 'prediction_source'], ascending=[False, True])
    )
    breakdown['accuracy'] = breakdown['accuracy'].round(4)
    return breakdown


def main() -> None:
    train_aspects, test_aspects, train_categories, test_categories = load_frames()

    train_eval = build_single_category_eval_frame(train_aspects, train_categories)
    test_eval = build_single_category_eval_frame(test_aspects, test_categories)

    baseline_predictions, baseline_metrics = evaluate_method(test_eval, baseline_rule_predict)

    mapper = HybridAspectMapper()
    mapper.fit(train_eval)
    hybrid_predictions, hybrid_metrics = evaluate_method(test_eval, mapper.predict)

    train_eval.to_csv(MODELS_DIR / 'aspect_generation_train_eval_frame.csv', index=False)
    test_eval.to_csv(MODELS_DIR / 'aspect_generation_test_eval_frame.csv', index=False)
    baseline_predictions.to_csv(MODELS_DIR / 'aspect_generation_baseline_predictions.csv', index=False)
    hybrid_predictions.to_csv(MODELS_DIR / 'aspect_generation_hybrid_predictions.csv', index=False)

    source_breakdown = build_source_breakdown(hybrid_predictions)
    source_breakdown.to_csv(MODELS_DIR / 'aspect_generation_source_breakdown.csv', index=False)

    summary = {
        'evaluation_note': (
            'Aspect-category evaluation uses only single-category sentences, because the SemEval restaurant '
            'data does not directly align each aspect term to a category in multi-category sentences.'
        ),
        'metric_note': 'Macro-F1 is the primary metric because category labels are imbalanced.',
        'method_note': (
            'The hybrid mapper uses smoothed support-aware category scores with minimum-support checks, '
            'then falls back through seed keywords, regex rules, and a low-confidence majority-category fallback.'
        ),
        'limitation_note': (
            'Category mapping is evaluated separately from raw aspect-term extraction. Any majority-category '
            'fallback predictions are low-confidence cases and are tracked explicitly in the outputs.'
        ),
        'train_eval_rows': int(train_eval.shape[0]),
        'test_eval_rows': int(test_eval.shape[0]),
        'smoothing_alpha': SMOOTHING_ALPHA,
        'term_min_support': TERM_MIN_SUPPORT,
        'headword_min_support': HEADWORD_MIN_SUPPORT,
        'majority_category_fallback': mapper.majority_category,
        'baseline': baseline_metrics,
        'hybrid': hybrid_metrics,
        'hybrid_source_counts': {
            row['prediction_source']: int(row['rows'])
            for row in source_breakdown.to_dict(orient='records')
        },
        'improvement_accuracy': round(hybrid_metrics['accuracy'] - baseline_metrics['accuracy'], 4),
        'improvement_macro_f1': round(hybrid_metrics['macro_f1'] - baseline_metrics['macro_f1'], 4),
    }
    with open(MODELS_DIR / 'aspect_generation_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    term_lexicon_rows = []
    for term, score_map in sorted(mapper.term_category_scores.items()):
        best_category = max(score_map, key=score_map.get)
        term_lexicon_rows.append(
            {
                'term_normalized': term,
                'support': mapper.term_support.get(term, 0),
                'predicted_category': best_category,
                'best_score': score_map[best_category],
            }
        )
    pd.DataFrame(term_lexicon_rows).to_csv(MODELS_DIR / 'learned_term_category_lexicon.csv', index=False)

    print('Aspect generation evaluation saved in outputs/models.')


if __name__ == '__main__':
    main()
