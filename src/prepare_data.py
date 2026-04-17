from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import CATEGORY_ORDER, POLARITY_ORDER, PROCESSED_DIR, TEST_XML, TRAIN_XML

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


@dataclass(frozen=True)
class ParsedDataset:
    name: str
    sentences: pd.DataFrame
    aspects: pd.DataFrame
    categories: pd.DataFrame


def normalize_text(text: str) -> str:
    return ' '.join((text or '').split())


def normalize_term(term: str) -> str:
    term = normalize_text(term).lower().strip()
    term = re.sub(r"[^a-z0-9\s\-']", ' ', term)
    term = re.sub(r'\s+', ' ', term)
    return term


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or '').lower())


def parse_restaurant_xml(path: Path, split_name: str) -> ParsedDataset:
    root = ET.parse(path).getroot()

    sentence_rows: list[dict] = []
    aspect_rows: list[dict] = []
    category_rows: list[dict] = []

    for sentence in root.findall('.//sentence'):
        sentence_id = sentence.attrib['id']
        text = normalize_text(sentence.findtext('text', default=''))

        aspect_terms_node = sentence.find('aspectTerms')
        aspect_categories_node = sentence.find('aspectCategories')

        aspect_terms = aspect_terms_node.findall('aspectTerm') if aspect_terms_node is not None else []
        aspect_categories = aspect_categories_node.findall('aspectCategory') if aspect_categories_node is not None else []

        sentence_rows.append(
            {
                'split': split_name,
                'sentence_id': sentence_id,
                'text': text,
                'token_count': len(tokenize(text)),
                'aspect_term_count': len(aspect_terms),
                'aspect_category_count': len(aspect_categories),
                'has_aspect_term': int(bool(aspect_terms)),
                'has_multiple_categories': int(len(aspect_categories) > 1),
            }
        )

        for idx, aspect in enumerate(aspect_terms):
            term = aspect.attrib.get('term', '')
            aspect_rows.append(
                {
                    'split': split_name,
                    'sentence_id': sentence_id,
                    'aspect_id': f'{sentence_id}::term::{idx}',
                    'text': text,
                    'term': term,
                    'term_normalized': normalize_term(term),
                    'polarity': aspect.attrib.get('polarity', '').lower(),
                    'char_from': int(aspect.attrib.get('from', -1)),
                    'char_to': int(aspect.attrib.get('to', -1)),
                }
            )

        for idx, category in enumerate(aspect_categories):
            category_rows.append(
                {
                    'split': split_name,
                    'sentence_id': sentence_id,
                    'category_id': f'{sentence_id}::cat::{idx}',
                    'text': text,
                    'category': category.attrib.get('category', '').lower(),
                    'polarity': category.attrib.get('polarity', '').lower(),
                }
            )

    return ParsedDataset(
        name=split_name,
        sentences=pd.DataFrame(sentence_rows),
        aspects=pd.DataFrame(aspect_rows),
        categories=pd.DataFrame(category_rows),
    )


def export_dataset(dataset: ParsedDataset) -> None:
    dataset.sentences.to_csv(PROCESSED_DIR / f'{dataset.name}_sentences.csv', index=False)
    dataset.aspects.to_csv(PROCESSED_DIR / f'{dataset.name}_aspect_terms.csv', index=False)
    dataset.categories.to_csv(PROCESSED_DIR / f'{dataset.name}_aspect_categories.csv', index=False)


def build_dataset_summary(train: ParsedDataset, test: ParsedDataset) -> dict:
    def polarity_counts(df: pd.DataFrame) -> dict[str, int]:
        counts = df['polarity'].value_counts().to_dict()
        return {label: int(counts.get(label, 0)) for label in POLARITY_ORDER}

    def category_counts(df: pd.DataFrame) -> dict[str, int]:
        counts = df['category'].value_counts().to_dict()
        return {label: int(counts.get(label, 0)) for label in CATEGORY_ORDER}

    return {
        'train': {
            'sentence_count': int(train.sentences.shape[0]),
            'aspect_term_count': int(train.aspects.shape[0]),
            'aspect_category_count': int(train.categories.shape[0]),
            'aspect_term_polarities': polarity_counts(train.aspects),
            'aspect_category_distribution': category_counts(train.categories),
        },
        'test': {
            'sentence_count': int(test.sentences.shape[0]),
            'aspect_term_count': int(test.aspects.shape[0]),
            'aspect_category_count': int(test.categories.shape[0]),
            'aspect_term_polarities': polarity_counts(test.aspects),
            'aspect_category_distribution': category_counts(test.categories),
        },
        'notes': {
            'full_dataset_present': bool(TRAIN_XML.exists() and TEST_XML.exists()),
            'format': 'SemEval 2014 restaurant XML with sentence text, aspect terms, and aspect categories.',
        },
    }


def main() -> None:
    train = parse_restaurant_xml(TRAIN_XML, 'train')
    test = parse_restaurant_xml(TEST_XML, 'test')

    export_dataset(train)
    export_dataset(test)

    summary = build_dataset_summary(train, test)
    with open(PROCESSED_DIR / 'dataset_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print('Prepared train and test files in data/processed.')


if __name__ == '__main__':
    main()
