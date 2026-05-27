"""Text cleaning and XML data loading.
Why: All user input and training text must be normalized before analysis —
lowercase, expand contractions, strip noise. XML parsing creates structured
DataFrames from the SemEval-2014 annotation files."""

import re
import xml.etree.ElementTree as ET
import pandas as pd
import contractions
from dataclasses import dataclass

from .config import TOKEN_RE


def normalize_text(text):
    """Collapse multiple spaces into one."""
    return ' '.join((text or '').split())


def normalize_term(term):
    """Lowercase and strip special chars from aspect terms like 'Pizza' -> 'pizza'."""
    term = normalize_text(term).lower().strip()
    term = re.sub(r'[^a-z0-9\s\-\']', ' ', term)
    return re.sub(r'\s+', ' ', term)


def clean_text(text):
    """Full cleaning pipeline: expand 'don't' -> 'do not', lowercase, remove URLs/HTML."""
    text = contractions.fix(str(text)).lower()
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
    return re.sub(r'\s+', ' ', text).strip()


# --- Parsed dataset container ---
@dataclass(frozen=True)
class ParsedDataset:
    name: str
    sentences: pd.DataFrame
    aspects: pd.DataFrame
    categories: pd.DataFrame


def parse_restaurant_xml(path, split_name):
    """Read SemEval-2014 XML into three DataFrames: sentences, aspects, categories.
    Each review sentence may have multiple aspect terms and category labels.
    We pull them all out into flat tables for training."""
    root = ET.parse(path).getroot()
    sentences, aspects, categories = [], [], []

    for sentence in root.findall('.//sentence'):
        sid = sentence.attrib['id']
        text = normalize_text(sentence.findtext('text', default=''))
        at_node = sentence.find('aspectTerms')
        ac_node = sentence.find('aspectCategories')
        at_list = at_node.findall('aspectTerm') if at_node is not None else []
        ac_list = ac_node.findall('aspectCategory') if ac_node is not None else []

        sentences.append({
            'split': split_name, 'sentence_id': sid, 'text': text,
            'token_count': len(TOKEN_RE.findall(text)),
            'aspect_term_count': len(at_list),
            'aspect_category_count': len(ac_list)
        })

        for idx, asp in enumerate(at_list):
            aspects.append({
                'split': split_name, 'sentence_id': sid,
                'aspect_id': f'{sid}::term::{idx}', 'text': text,
                'term': asp.attrib.get('term', ''),
                'term_normalized': normalize_term(asp.attrib.get('term', '')),
                'polarity': asp.attrib.get('polarity', '').lower()
            })

        for idx, cat in enumerate(ac_list):
            categories.append({
                'split': split_name, 'sentence_id': sid,
                'category_id': f'{sid}::cat::{idx}', 'text': text,
                'category': cat.attrib.get('category', '').lower(),
                'polarity': cat.attrib.get('polarity', '').lower()
            })

    return ParsedDataset(
        name=split_name,
        sentences=pd.DataFrame(sentences),
        aspects=pd.DataFrame(aspects),
        categories=pd.DataFrame(categories)
    )
