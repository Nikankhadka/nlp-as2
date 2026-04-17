from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import CATEGORY_ORDER, EDA_DIR, POLARITY_ORDER, PROCESSED_DIR


def save_bar_chart(series: pd.Series, title: str, output_name: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 5))
    series.plot(kind='bar')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(EDA_DIR / output_name, dpi=180)
    plt.close()


def save_histogram(series: pd.Series, title: str, output_name: str, xlabel: str, bins: int = 20) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(series, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(EDA_DIR / output_name, dpi=180)
    plt.close()


def main() -> None:
    train_sentences = pd.read_csv(PROCESSED_DIR / 'train_sentences.csv')
    test_sentences = pd.read_csv(PROCESSED_DIR / 'test_sentences.csv')
    train_aspects = pd.read_csv(PROCESSED_DIR / 'train_aspect_terms.csv')
    test_aspects = pd.read_csv(PROCESSED_DIR / 'test_aspect_terms.csv')
    train_categories = pd.read_csv(PROCESSED_DIR / 'train_aspect_categories.csv')
    test_categories = pd.read_csv(PROCESSED_DIR / 'test_aspect_categories.csv')

    dataset_overview = pd.DataFrame(
        [
            {
                'split': 'train',
                'sentences': int(train_sentences.shape[0]),
                'aspect_terms': int(train_aspects.shape[0]),
                'aspect_categories': int(train_categories.shape[0]),
                'avg_tokens_per_sentence': round(train_sentences['token_count'].mean(), 2),
                'avg_aspects_per_sentence': round(train_sentences['aspect_term_count'].mean(), 2),
                'share_multi_category_sentences': round(train_sentences['has_multiple_categories'].mean(), 4),
            },
            {
                'split': 'test',
                'sentences': int(test_sentences.shape[0]),
                'aspect_terms': int(test_aspects.shape[0]),
                'aspect_categories': int(test_categories.shape[0]),
                'avg_tokens_per_sentence': round(test_sentences['token_count'].mean(), 2),
                'avg_aspects_per_sentence': round(test_sentences['aspect_term_count'].mean(), 2),
                'share_multi_category_sentences': round(test_sentences['has_multiple_categories'].mean(), 4),
            },
        ]
    )
    dataset_overview.to_csv(EDA_DIR / 'dataset_overview.csv', index=False)

    term_polarities = (
        pd.concat([train_aspects.assign(split='train'), test_aspects.assign(split='test')])
        .groupby(['split', 'polarity'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=POLARITY_ORDER, fill_value=0)
    )
    term_polarities.to_csv(EDA_DIR / 'term_polarity_distribution.csv')

    category_distribution = (
        pd.concat([train_categories.assign(split='train'), test_categories.assign(split='test')])
        .groupby(['split', 'category'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORY_ORDER, fill_value=0)
    )
    category_distribution.to_csv(EDA_DIR / 'category_distribution.csv')

    top_terms = (
        train_aspects['term_normalized']
        .value_counts()
        .head(20)
        .rename_axis('term')
        .reset_index(name='count')
    )
    top_terms.to_csv(EDA_DIR / 'top_train_aspect_terms.csv', index=False)

    save_bar_chart(
        train_aspects['polarity'].value_counts().reindex(POLARITY_ORDER, fill_value=0),
        'Train aspect-term polarity distribution',
        'train_term_polarity_distribution.png',
        'Polarity',
        'Aspect terms',
    )
    save_bar_chart(
        train_categories['category'].value_counts().reindex(CATEGORY_ORDER, fill_value=0),
        'Train aspect-category distribution',
        'train_category_distribution.png',
        'Category',
        'Category annotations',
    )
    save_histogram(
        train_sentences['token_count'],
        'Train sentence length distribution',
        'train_sentence_length_histogram.png',
        'Tokens per sentence',
        bins=25,
    )
    save_histogram(
        train_sentences['aspect_term_count'],
        'Train aspect terms per sentence',
        'train_aspects_per_sentence_histogram.png',
        'Aspect terms per sentence',
        bins=10,
    )
    save_bar_chart(
        top_terms.set_index('term')['count'],
        'Top 20 train aspect terms',
        'top_train_aspect_terms.png',
        'Aspect term',
        'Count',
    )

    findings = {
        'positive_term_share_train': round((train_aspects['polarity'] == 'positive').mean(), 4),
        'negative_term_share_train': round((train_aspects['polarity'] == 'negative').mean(), 4),
        'conflict_term_share_train': round((train_aspects['polarity'] == 'conflict').mean(), 4),
        'food_category_share_train': round((train_categories['category'] == 'food').mean(), 4),
        'multi_category_sentence_share_train': round(train_sentences['has_multiple_categories'].mean(), 4),
        'median_tokens_per_sentence_train': int(train_sentences['token_count'].median()),
        'median_aspect_terms_per_sentence_train': int(train_sentences['aspect_term_count'].median()),
    }
    with open(EDA_DIR / 'eda_findings.json', 'w', encoding='utf-8') as handle:
        json.dump(findings, handle, indent=2)

    print('EDA outputs created in outputs/eda.')


if __name__ == '__main__':
    main()
