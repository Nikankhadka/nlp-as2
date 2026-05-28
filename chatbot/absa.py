"""ABSA engine — aspect extraction, category mapping, feature engineering,
sentiment classification, and response formatting."""

import random
import re
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

from .config import nlp, STOPWORDS, CATEGORIES
from .data_utils import clean_text, normalize_term


NON_ASPECTS = set([
    'good', 'great', 'bad', 'excellent', 'amazing', 'terrible',
    'delicious', 'friendly', 'nice', 'love', 'loved', 'horrible',
    'poor', 'slow', 'wonderful', 'best', 'worst', 'perfect', 'awful',
    'unhelpful', 'overcooked', 'undercooked', 'rude', 'cold', 'hot',
    'loud', 'dirty', 'clean', 'fresh', 'stale', 'burnt', 'raw'
])

EMOJI = {
    'positive': '\U0001f60a', 'negative': '\U0001f61e',
    'neutral': '\U0001f610', 'conflict': '\U0001f914'
}


def build_extraction_lexicon(df):
    tc = Counter(df['term_normalized'])
    hc = Counter(df['term_normalized'].apply(lambda t: t.split()[-1]))
    return (
        {t for t, c in tc.items() if c >= 2 and len(t.split()) == 1},
        {t for t, c in tc.items() if c >= 2 and 1 < len(t.split()) <= 3},
        {h for h, c in hc.items() if c >= 3 and h not in STOPWORDS}
    )


def extract_aspects_spacy(text, single, multi, heads):
    cleaned = clean_text(text)
    doc = nlp(cleaned)
    found = set()

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
        for i in range(len(tokens) - size + 1):
            p = ' '.join(tokens[i:i + size])
            if p in multi:
                found.add(p)
    for t in tokens:
        if t in single:
            found.add(t)
    return list(found)


def predict_category_fast(term):
    tl = term.lower()
    if any(w in tl for w in [
        'food', 'dish', 'pasta', 'pizza', 'taste', 'flavor', 'dessert', 'meal',
        'cuisine', 'ingredient', 'steak', 'sushi', 'cocktail', 'appetizer',
        'soup', 'salad', 'burger', 'sandwich', 'seafood', 'wine', 'cocktails',
        'tiramisu', 'risotto', 'noodle', 'rice', 'sauce', 'noodles', 'pizzas',
        'desserts', 'drinks', 'lobster', 'chicken', 'chocolate', 'bite', 'sashimi',
        'beef', 'bread', 'cake', 'cheese', 'fries', 'coffee', 'roll', 'lamb', 'pork'
    ]):
        return 'food'
    if any(w in tl for w in [
        'staff', 'waiter', 'server', 'waitress', 'host', 'bartender', 'service',
        'sommelier', 'waiters', 'servers'
    ]):
        return 'service'
    if any(w in tl for w in [
        'price', 'bill', 'cost', 'expensive', 'cheap', 'value', 'money', 'dollar',
        'overpriced', 'prices', 'deal'
    ]):
        return 'price'
    if any(w in tl for w in [
        'atmosphere', 'decor', 'music', 'ambience', 'lighting', 'mood', 'vibe',
        'setting', 'scene', 'environment'
    ]):
        return 'ambience'
    return 'miscellaneous'


def make_feature(text, term):
    txt = clean_text(text)
    tn = normalize_term(term)
    idx = txt.find(tn)
    if idx == -1:
        return f'[ASPECT] {tn} [/ASPECT] || {txt}'
    return txt[:idx] + f' [ASPECT] {txt[idx:idx + len(tn)]} [/ASPECT] ' + txt[idx + len(tn):]


def train_model(train_df):
    train_df = train_df.copy()
    train_df['clean_text'] = train_df['text'].apply(clean_text)
    train_df['feature'] = train_df.apply(
        lambda r: make_feature(r['text'], r['term_normalized']), axis=1
    )

    print('Training TF-IDF + SMOTE + Logistic Regression...', flush=True)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_vec = tfidf.fit_transform(train_df['feature'])
    y_all = train_df['polarity']

    smote = SMOTE(random_state=42)
    X_s, y_s = smote.fit_resample(X_vec, y_all)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_s, y_s)
    print(f'Training complete! Classes: {list(clf.classes_)}', flush=True)
    return tfidf, clf


def format_absa_response(results):
    if not results:
        return ('I could not identify specific aspects in your message.\n'
                'Try mentioning food, service, price, or ambience specifically.')

    by_cat = defaultdict(list)
    for a in results:
        by_cat[a['category']].append(a)

    dominant_overall = Counter(i['sentiment'] for i in results).most_common(1)[0][0]

    templates = [
        lambda: _format_absa_v1(by_cat, results),
        lambda: _format_absa_v2(by_cat, results, dominant_overall),
        lambda: _format_absa_v3(by_cat, results, dominant_overall),
        lambda: _format_absa_v4(by_cat, results, dominant_overall),
        lambda: _format_absa_v5(by_cat, results),
    ]
    return random.choice(templates)()


def _format_absa_v1(by_cat, results):
    lines = ['Here is what I found:\n']
    for cat, items in by_cat.items():
        dominant = Counter(i['sentiment'] for i in items).most_common(1)[0][0]
        unique_terms = {re.sub(r'^(the|a|an)\s+', '', i['aspect'].strip()) for i in items}
        terms = ', '.join(unique_terms)
        lines.append(f'  {EMOJI.get(dominant, "")} {cat.upper()}: {dominant}  (about: {terms})')
    lines.append('\nWant to ask about a specific aspect like food quality or service?')
    return '\n'.join(lines)


def _format_absa_v2(by_cat, results, overall):
    lines = [f'Overall, your review feels {overall} to me. Breaking it down:\n']
    for cat, items in by_cat.items():
        dominant = Counter(i['sentiment'] for i in items).most_common(1)[0][0]
        unique_terms = {re.sub(r'^(the|a|an)\s+', '', i['aspect'].strip()) for i in items}
        terms = ', '.join(unique_terms)
        lines.append(f'  {cat.capitalize()} ({terms}): {dominant} {EMOJI.get(dominant, "")}')
    lines.append('\nWould you like a deeper breakdown on any of these?')
    return '\n'.join(lines)


def _format_absa_v3(by_cat, results, overall):
    lines = [f'I analysed your review — the overall tone is {overall}.\n']
    for cat, items in by_cat.items():
        for i in items[:2]:
            aspect = re.sub(r'^(the|a|an)\s+', '', i['aspect'].strip())
            lines.append(f'  {EMOJI.get(i["sentiment"], "")} {aspect}: {i["sentiment"]}')
    lines.append('\nCurious about any of these in more detail?')
    return '\n'.join(lines)


def _format_absa_v4(by_cat, results, overall):
    pos_items = [i for i in results if i['sentiment'] == 'positive']
    neg_items = [i for i in results if i['sentiment'] == 'negative']
    lines = []
    if pos_items:
        aspects = ', '.join({re.sub(r'^(the|a|an)\s+', '', i['aspect'].strip()) for i in pos_items})
        lines.append(f'\U0001f60a Things you liked: {aspects}')
    if neg_items:
        aspects = ', '.join({re.sub(r'^(the|a|an)\s+', '', i['aspect'].strip()) for i in neg_items})
        lines.append(f'\U0001f61e Things you didn\'t like: {aspects}')
    if not lines:
        lines = [f'Overall tone: {overall}']
    lines.append('\nAnything else you\'d like me to analyze?')
    return '\n'.join(lines)


def _format_absa_v5(by_cat, results):
    lines = ['Here is a summary of your review:\n']
    for cat, items in by_cat.items():
        unique_terms = {re.sub(r'^(the|a|an)\s+', '', i['aspect'].strip()) for i in items}
        terms = ', '.join(unique_terms)
        sentiments = Counter(i['sentiment'] for i in items)
        sent_str = ' / '.join(f'{s}({c}x)' for s, c in sentiments.items())
        lines.append(f'  {cat.capitalize()}: {sent_str}  [{terms}]')
    if len(by_cat) > 1:
        lines.append('\nThat\'s a detailed review — multiple aspects always give a better picture!')
    return '\n'.join(lines)


def analyse(text, single_lex, multi_lex, head_lex, tfidf, clf):
    aspects = extract_aspects_spacy(text, single_lex, multi_lex, head_lex)
    results = []
    seen = set()
    for asp in aspects:
        clean_asp = re.sub(r'^(the|a|an)\s+', '', asp.strip())
        if clean_asp in seen:
            continue
        seen.add(clean_asp)
        try:
            feat = make_feature(text, clean_asp)
            vec = tfidf.transform([feat])
            results.append({
                'aspect': clean_asp,
                'category': predict_category_fast(clean_asp),
                'sentiment': clf.predict(vec)[0]
            })
        except Exception:
            results.append({
                'aspect': clean_asp,
                'category': 'miscellaneous',
                'sentiment': 'neutral'
            })
    return results
