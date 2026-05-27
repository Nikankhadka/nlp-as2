#!/usr/bin/env python3
"""
Chatbot Runner — Consolidated from chatbot.ipynb
Supports both Colab and local execution.
"""

import sys, os, re, json, csv
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import contractions
import spacy
from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from transformers import pipeline as hf_pipeline
from nltk.corpus import stopwords

# ============================================================
# PATH SETUP — supports both Colab and local
# ============================================================
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/drive/MyDrive/absa')
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_XML = PROJECT_ROOT / 'data' / 'raw' / 'Restaurants_Train_v2.xml'
TEST_XML  = PROJECT_ROOT / 'data' / 'raw' / 'Restaurants_Test_Gold.xml'

print(f'Project root: {PROJECT_ROOT}')
print(f'Train XML exists: {TRAIN_XML.exists()}')
print(f'Test XML  exists: {TEST_XML.exists()}')
if not TRAIN_XML.exists():
    raise FileNotFoundError(f'Training XML not found at {TRAIN_XML}')

# ============================================================
# LOAD SPACY & NLTK
# ============================================================
print('Loading spaCy and NLTK...')
nlp = spacy.load('en_core_web_sm')
STOPWORDS = set(stopwords.words('english'))
print(f'Loaded {len(STOPWORDS)} stopwords')

TOKEN_RE       = re.compile(r'[A-Za-z][A-Za-z\-\']+')
CATEGORIES     = ['food', 'service', 'price', 'ambience', 'miscellaneous']

# ============================================================
# TEXT CLEANING
# ============================================================
def normalize_text(text):
    return ' '.join((text or '').split())

def normalize_term(term):
    term = normalize_text(term).lower().strip()
    term = re.sub(r'[^a-z0-9\s\-\']', ' ', term)
    return re.sub(r'\s+', ' ', term)

def clean_text(text):
    text = contractions.fix(str(text)).lower()
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ============================================================
# XML PARSER
# ============================================================
@dataclass(frozen=True)
class ParsedDataset:
    name: str
    sentences: pd.DataFrame
    aspects: pd.DataFrame
    categories: pd.DataFrame

def parse_restaurant_xml(path, split_name):
    root = ET.parse(path).getroot()
    sentences, aspects, categories = [], [], []
    for sentence in root.findall('.//sentence'):
        sid     = sentence.attrib['id']
        text    = normalize_text(sentence.findtext('text', default=''))
        at_node = sentence.find('aspectTerms')
        ac_node = sentence.find('aspectCategories')
        at_list = at_node.findall('aspectTerm')    if at_node is not None else []
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

# ============================================================
# ASPECT EXTRACTION
# ============================================================
NON_ASPECTS = set([
    'good','great','bad','excellent','amazing','terrible',
    'delicious','friendly','nice','love','loved','horrible',
    'poor','slow','wonderful','best','worst','perfect','awful',
    'unhelpful','overcooked','undercooked','rude','cold','hot',
    'loud','dirty','clean','fresh','stale','burnt','raw'
])

def build_extraction_lexicon(df):
    tc = Counter(df['term_normalized'])
    hc = Counter(df['term_normalized'].apply(lambda t: t.split()[-1]))
    return (
        {t for t,c in tc.items() if c>=2 and len(t.split())==1},
        {t for t,c in tc.items() if c>=2 and 1<len(t.split())<=3},
        {h for h,c in hc.items() if c>=3 and h not in STOPWORDS}
    )

def extract_aspects_spacy(text, single, multi, heads):
    cleaned = clean_text(text)
    doc     = nlp(cleaned)
    found   = set()
    for token in doc:
        word = token.text.lower()
        if (token.pos_ in ('NOUN', 'PROPN')
                and word not in STOPWORDS
                and word not in NON_ASPECTS
                and len(word) > 2):
            found.add(word)
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        if (phrase not in STOPWORDS
                and phrase not in NON_ASPECTS
                and len(phrase) > 2):
            found.add(phrase)
    tokens = re.findall(r'[a-z][a-z\-\']+', cleaned)
    for size in [3, 2]:
        for i in range(len(tokens)-size+1):
            p = ' '.join(tokens[i:i+size])
            if p in multi: found.add(p)
    for t in tokens:
        if t in single: found.add(t)
    return list(found)

# ============================================================
# BART ZERO-SHOT CATEGORY MAPPER
# ============================================================
print('Loading BART zero-shot classifier (1.6GB model)...', flush=True)
zero_shot = hf_pipeline(
    'zero-shot-classification',
    model='facebook/bart-large-mnli',
    device=-1  # force CPU
)
print('BART loaded.', flush=True)

def predict_category(term):
    try:
        result = zero_shot(term, CATEGORIES)
        return result['labels'][0]
    except Exception:
        term_lower = term.lower()
        if any(w in term_lower for w in ['food','dish','pasta','pizza','taste','flavor']): return 'food'
        if any(w in term_lower for w in ['staff','waiter','service','server']): return 'service'
        if any(w in term_lower for w in ['price','bill','cost','expensive','cheap']): return 'price'
        if any(w in term_lower for w in ['atmosphere','decor','music','ambience']): return 'ambience'
        return 'miscellaneous'

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def make_feature(text, term):
    txt = clean_text(text)
    tn  = normalize_term(term)
    idx = txt.find(tn)
    if idx == -1:
        return f'[ASPECT] {tn} [/ASPECT] || {txt}'
    return txt[:idx] + f' [ASPECT] {txt[idx:idx+len(tn)]} [/ASPECT] ' + txt[idx+len(tn):]

# ============================================================
# INTENT DETECTION
# ============================================================
GREETINGS  = {'hi','hello','hey','howdy','greetings','morning','evening'}
FAREWELL   = {'bye','goodbye','quit','thanks','thank'}
HELP_WORDS = {'help','capabilities'}

def detect_intent(text):
    tokens = set(re.findall(r'[a-z]+', text.lower()))
    if tokens & GREETINGS: return 'greeting'
    if tokens & FAREWELL:  return 'farewell'
    if tokens & HELP_WORDS:return 'help'
    doc   = nlp(clean_text(text))
    nouns = [token.text.lower() for token in doc
             if token.pos_ in ('NOUN', 'PROPN')
             and len(token.text) > 2]
    if nouns:
        try:
            result = zero_shot(text, ['restaurant review', 'general conversation'])
            if result['labels'][0] == 'restaurant review':
                return 'restaurant_query'
        except Exception:
            basic = {
                'food','meal','dish','menu','taste','service','staff',
                'waiter','price','bill','cost','ambience','atmosphere',
                'restaurant','table','reservation','drink','wine'
            }
            if tokens & basic:
                return 'restaurant_query'
    return 'general'

EMOJI = {'positive':'😊', 'negative':'😞', 'neutral':'😐', 'conflict':'🤔'}

def format_absa_response(results):
    if not results:
        return ('I could not identify specific aspects in your message.\n'
                'Try mentioning food, service, price, or ambience specifically.')
    lines  = ['Here is what I found:\n']
    by_cat = defaultdict(list)
    for a in results:
        by_cat[a['category']].append(a)
    for cat, items in by_cat.items():
        dominant = Counter(i['sentiment'] for i in items).most_common(1)[0][0]
        unique_terms = set()
        for i in items:
            aspect = re.sub(r'^(the|a|an)\s+', '', i['aspect'].strip())
            unique_terms.add(aspect)
        terms = ', '.join(unique_terms)
        lines.append(f'  {EMOJI.get(dominant,"")} {cat.upper()}: {dominant}  (about: {terms})')
    lines.append('\nWant to ask about a specific aspect like food quality or service?')
    return '\n'.join(lines)

def general_responses(intent):
    if intent == 'greeting':
        return ('Hello! 👋 I am your restaurant review expert chatbot.\n'
                'I can analyse reviews and tell you how people feel about\n'
                'the food, service, price, or ambience.\n'
                'Just type a review or ask a question!')
    if intent == 'farewell':
        return 'Thanks for chatting! Hope the insights were helpful. Goodbye! 👋'
    if intent == 'help':
        return ('I can help you with:\n'
                '  • Analysing sentiment in restaurant reviews\n'
                '  • Identifying which aspects are positive or negative\n'
                '  • Answering questions about food, service, price, ambience\n\n'
                'Try typing:\n'
                '  "The pasta was cold but the waiter was friendly"\n'
                '  "What do people think about the service?"')
    return ('I am not sure I understood that.\n'
            'Try typing a restaurant review like:\n'
            '  "The pasta was cold but the waiter was friendly"')

# ============================================================
# TRAINING
# ============================================================
print('Loading XML data...', flush=True)
train_data = parse_restaurant_xml(TRAIN_XML, 'train')
print(f'Loaded {len(train_data.aspects)} training aspect annotations')

print('Building extraction lexicon...', flush=True)
single_lex, multi_lex, head_lex = build_extraction_lexicon(train_data.aspects)
print(f'Single: {len(single_lex)} | Multi: {len(multi_lex)} | Head: {len(head_lex)}')

print('Preparing features...', flush=True)
train_df = train_data.aspects.copy()
train_df['clean_text'] = train_df['text'].apply(clean_text)
train_df['feature'] = train_df.apply(
    lambda r: make_feature(r['text'], r['term_normalized']), axis=1
)

print('Training TF-IDF + SMOTE + Logistic Regression...', flush=True)
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True)
X_vec = tfidf.fit_transform(train_df['feature'])
y_all = train_df['polarity']

smote    = SMOTE(random_state=42)
X_s, y_s = smote.fit_resample(X_vec, y_all)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_s, y_s)
print('Training complete!')
print(f'Classes: {list(clf.classes_)}')

# ============================================================
# CHAT FUNCTIONS
# ============================================================
def analyse(text):
    aspects = extract_aspects_spacy(text, single_lex, multi_lex, head_lex)
    results = []
    seen    = set()
    for asp in aspects:
        clean_asp = re.sub(r'^(the|a|an)\s+', '', asp.strip())
        if clean_asp in seen:
            continue
        seen.add(clean_asp)
        try:
            feat = make_feature(text, clean_asp)
            vec  = tfidf.transform([feat])
            results.append({
                'aspect':    clean_asp,
                'category':  predict_category(clean_asp),
                'sentiment': clf.predict(vec)[0]
            })
        except Exception as e:
            results.append({
                'aspect':    clean_asp,
                'category':  'miscellaneous',
                'sentiment': 'neutral'
            })
    return results

def chat(user_input):
    if not user_input.strip():
        return 'Please type something!'
    intent = detect_intent(user_input)
    if intent == 'restaurant_query':
        return format_absa_response(analyse(user_input))
    return general_responses(intent)

# ============================================================
# EVALUATION ON TEST SET
# ============================================================
def evaluate_test_set():
    print('\n' + '='*60)
    print('EVALUATING ON TEST SET')
    print('='*60)
    test_data = parse_restaurant_xml(TEST_XML, 'test')
    print(f'Loaded {len(test_data.aspects)} test aspect annotations')

    predictions = []
    for _, row in test_data.aspects.iterrows():
        feat = make_feature(row['text'], row['term_normalized'])
        vec  = tfidf.transform([feat])
        pred = clf.predict(vec)[0]
        predictions.append(pred)

    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    y_true = test_data.aspects['polarity'].values

    acc = accuracy_score(y_true, predictions)
    f1_macro = f1_score(y_true, predictions, average='macro')
    f1_weighted = f1_score(y_true, predictions, average='weighted')

    print(f'\nAccuracy:      {acc:.4f}')
    print(f'Macro-F1:      {f1_macro:.4f}')
    print(f'Weighted-F1:   {f1_weighted:.4f}')
    print(f'\nClassification Report:')
    print(classification_report(y_true, predictions, digits=4))
    print(f'\nConfusion Matrix:')
    print(confusion_matrix(y_true, predictions))

    # Test aspect extraction quality
    print('\n--- Aspect Extraction Test ---')
    total = len(test_data.aspects)
    test_sentences = test_data.sentences
    extracted_count = 0
    for _, srow in test_sentences.iterrows():
        ex = extract_aspects_spacy(srow['text'], single_lex, multi_lex, head_lex)
        if ex:
            extracted_count += 1
    print(f'Sentences with at least 1 aspect extracted: {extracted_count}/{len(test_sentences)}')

    # Category mapping test
    print('\n--- Category Mapping Sample ---')
    test_cats = test_data.categories
    cat_predictions = []
    for _, row in test_data.aspects.iterrows():
        cat_predictions.append(predict_category(row['term_normalized']))
    cat_true = []
    for _, row in test_data.aspects.iterrows():
        # get the first category match from test categories for this sentence
        sid = row['sentence_id']
        matching = test_cats[test_cats['sentence_id'] == sid]
        if len(matching) > 0:
            cat_true.append(matching.iloc[0]['category'])
        else:
            cat_true.append('miscellaneous')
    from sklearn.metrics import accuracy_score as acc2
    cat_acc = acc2(cat_true[:len(cat_predictions)], cat_predictions)
    print(f'Category mapping accuracy (approx): {cat_acc:.4f}')

    return {
        'accuracy': acc,
        'macro_f1': f1_macro,
        'weighted_f1': f1_weighted,
        'samples': len(y_true)
    }

# Run evaluation
results = evaluate_test_set()

# ============================================================
# 50-QUESTION TEST HARNESS
# ============================================================
print('\n' + '='*60)
print('RUNNING 50-QUESTION TEST HARNESS')
print('='*60)

test_questions = [
    # --- Greetings & Farewells (6) ---
    ('greeting', 'Hello!'),
    ('greeting', 'Hey there'),
    ('greeting', 'Hi'),
    ('greeting', 'Good morning'),
    ('farewell', 'Goodbye'),
    ('farewell', 'Thanks'),

    # --- Simple restaurant reviews - positive (6) ---
    ('restaurant_pos', 'The pizza was amazing'),
    ('restaurant_pos', 'The service was excellent'),
    ('restaurant_pos', 'Great food and friendly staff'),
    ('restaurant_pos', 'The desserts were delicious'),
    ('restaurant_pos', 'The waitress was very nice'),
    ('restaurant_pos', 'The ambience was wonderful'),

    # --- Simple restaurant reviews - negative (6) ---
    ('restaurant_neg', 'The pasta was cold'),
    ('restaurant_neg', 'The waiter was rude to us'),
    ('restaurant_neg', 'The food was terrible and overpriced'),
    ('restaurant_neg', 'The music was too loud'),
    ('restaurant_neg', 'The service was very slow'),
    ('restaurant_neg', 'The bill was outrageous'),

    # --- Mixed sentiment / complex reviews (8) ---
    ('restaurant_mixed', 'The pasta was cold but the waiter was incredibly friendly and fast'),
    ('restaurant_mixed', 'Overpriced for the tiny portions, though the atmosphere was cozy'),
    ('restaurant_mixed', 'Great value and quick service, but the music was too loud to talk'),
    ('restaurant_mixed', 'The staff was rude and the waiting time was too long'),
    ('restaurant_mixed', 'Amazing desserts and the ambience was perfect for a date night'),
    ('restaurant_mixed', 'The risotto was overcooked and the sommelier was rude'),
    ('restaurant_mixed', 'Food was decent but the place was dirty'),
    ('restaurant_mixed', 'Excellent taste but small portions'),

    # --- Questions / queries (6) ---
    ('restaurant_query', 'Is the food here good?'),
    ('restaurant_query', 'What do people say about the service?'),
    ('restaurant_query', 'How is the ambience?'),
    ('restaurant_query', 'Tell me about the desserts'),
    ('restaurant_query', 'What about the prices?'),
    ('restaurant_query', 'Are the waiters friendly?'),

    # --- Help & capabilities (2) ---
    ('help', 'What can you do?'),
    ('help', 'Help'),

    # --- Out-of-domain - general chat (10) ---
    ('general', 'What is the weather like today?'),
    ('general', 'Tell me about the football game'),
    ('general', 'How do I fix my car?'),
    ('general', 'What is the meaning of life?'),
    ('general', 'Who won the election?'),
    ('general', 'Tell me a joke'),
    ('general', 'What time is it?'),
    ('general', 'How old are you?'),
    ('general', 'That goal was out of this world'),
    ('general', 'Can you recommend a movie?'),

    # --- Edge cases (6) ---
    ('edge', ''),
    ('edge', 'Pizza'),
    ('edge', 'spagetti'),
    ('edge', 'A'),
    ('edge', '12345'),
    ('edge', '!!!!'),
]

test_results = []
for qtype, question in test_questions:
    try:
        response = chat(question)
    except Exception as e:
        response = f'[ERROR: {e}]'
    test_results.append({
        'type': qtype,
        'input': question,
        'response': response
    })
    # Print compact result
    short = response[:80].replace('\n', ' | ')
    print(f'[{qtype:20s}] {question[:40]:40s} → {short}')

# Save results
results_df = pd.DataFrame(test_results)
results_path = PROJECT_ROOT / 'outputs' / 'test_results_50.csv'
results_df.to_csv(results_path, index=False)
print(f'\nTest results saved to {results_path}')

# ============================================================
# SUMMARY
# ============================================================
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'Test set accuracy:       {results["accuracy"]:.4f}')
print(f'Test set macro-F1:       {results["macro_f1"]:.4f}')
print(f'Test set weighted-F1:    {results["weighted_f1"]:.4f}')
print(f'50-question test saved:  {results_path}')

# Count various stats
type_counts = Counter(r['type'] for r in test_results)
restaurant_queries = sum(1 for r in test_results if 'restaurant' in r['type'])
general_count = sum(1 for r in test_results if r['type'] == 'general')
print(f'\nQuestion breakdown:')
for t, c in sorted(type_counts.items()):
    print(f'  {t}: {c}')

# Show errors
errors = [r for r in test_results if r['response'].startswith('[ERROR')]
if errors:
    print(f'\nErrors encountered: {len(errors)}')
    for e in errors:
        print(f'  [{e["type"]}] {e["input"]}: {e["response"]}')
else:
    print(f'\nNo runtime errors encountered.')

print('\nDone!')
