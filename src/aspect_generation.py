from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

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
        'bill', 'bargain', 'overpriced', 'reasonable', 'money', 'priced',
    },
    'ambience': {
        'atmosphere', 'ambience', 'ambiance', 'decor', 'design', 'music', 'crowded', 'space',
        'room', 'rooms', 'vibe', 'romantic', 'cozy', 'garden', 'chairs', 'seats', 'noisy', 'quiet',
        'loud', 'interior',
    },
}


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


def baseline_rule_predict(term: str) -> tuple[str, str]:
    norm = normalize_term(term)
    tokens = set(norm.split())

    for category, keywords in SEED_KEYWORDS.items():
        if norm in keywords or tokens.intersection(keywords):
            return category, 'seed_keyword'

    if any(token.endswith(('ing', 'ed')) for token in tokens):
        return 'service', 'verbish_fallback'

    return 'food', 'default_food_fallback'


class HybridAspectMapper:
    def __init__(self) -> None:
        self.term_category_scores: dict[str, dict[str, float]] = {}
        self.headword_scores: dict[str, dict[str, float]] = {}

    @staticmethod
    def headword(term: str) -> str:
        pieces = normalize_term(term).split()
        return pieces[-1] if pieces else ''

    def fit(self, train_eval_frame: pd.DataFrame) -> None:
        term_counts: dict[str, Counter] = defaultdict(Counter)
        headword_counts: dict[str, Counter] = defaultdict(Counter)
        category_counts: Counter = Counter(train_eval_frame['gold_category'])
        total = len(train_eval_frame)

        for row in train_eval_frame.itertuples(index=False):
            term_counts[row.term_normalized][row.gold_category] += 1
            headword_counts[self.headword(row.term_normalized)][row.gold_category] += 1

        self.term_category_scores = self._to_scores(term_counts, category_counts, total)
        self.headword_scores = self._to_scores(headword_counts, category_counts, total)

    @staticmethod
    def _to_scores(counts: dict[str, Counter], category_counts: Counter, total: int) -> dict[str, dict[str, float]]:
        scores: dict[str, dict[str, float]] = {}
        for key, counter in counts.items():
            scores[key] = {}
            key_total = sum(counter.values())
            for category in CATEGORY_ORDER:
                joint = counter.get(category, 0)
                if joint == 0:
                    scores[key][category] = float('-inf')
                    continue
                p_joint = joint / total
                p_key = key_total / total
                p_category = category_counts[category] / total
                pmi = math.log2(p_joint / (p_key * p_category))
                scores[key][category] = round(pmi, 6)
        return scores

    def predict(self, term: str) -> tuple[str, str]:
        norm = normalize_term(term)
        tokens = set(norm.split())

        if norm in self.term_category_scores:
            score_map = self.term_category_scores[norm]
            return max(score_map, key=score_map.get), 'term_pmi'

        head = self.headword(norm)
        if head and head in self.headword_scores:
            score_map = self.headword_scores[head]
            return max(score_map, key=score_map.get), 'headword_pmi'

        for category, keywords in SEED_KEYWORDS.items():
            if norm in keywords or tokens.intersection(keywords):
                return category, 'seed_keyword'

        if re.search(r'price|cost|bill|value|cheap|expensive', norm):
            return 'price', 'regex_price'
        if re.search(r'staff|service|waiter|delivery|manager|reservation', norm):
            return 'service', 'regex_service'
        if re.search(r'atmosphere|ambience|decor|music|seat|room|space|vibe', norm):
            return 'ambience', 'regex_ambience'

        return 'food', 'default_food_fallback'


def evaluate_method(eval_frame: pd.DataFrame, predictor) -> tuple[pd.DataFrame, dict]:
    frame = eval_frame.copy()
    predictions = frame['term'].apply(predictor)
    frame['predicted_category'] = [item[0] for item in predictions]
    frame['prediction_source'] = [item[1] for item in predictions]

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
    }
    return frame, metrics


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

    summary = {
        'evaluation_note': (
            'Evaluation uses only single-category sentences, because SemEval restaurant data '
            'does not directly align each aspect term to a category in multi-category sentences.'
        ),
        'train_eval_rows': int(train_eval.shape[0]),
        'test_eval_rows': int(test_eval.shape[0]),
        'baseline': baseline_metrics,
        'hybrid': hybrid_metrics,
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
                'predicted_category': best_category,
                'best_score': score_map[best_category],
            }
        )
    pd.DataFrame(term_lexicon_rows).to_csv(MODELS_DIR / 'learned_term_category_lexicon.csv', index=False)

    print('Aspect generation evaluation saved in outputs/models.')


if __name__ == '__main__':
    main()
