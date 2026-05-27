#!/usr/bin/env python3
"""
Chatbot Runner — Consolidated from chatbot.ipynb
Supports both Colab and local execution.

Phase 1: Knowledge-grounded Q&A, expanded intent (6 paths), self-knowledge,
         conversation memory, diversified templates, BART removed.
Phase 2: OpenRouter LLM wrapper for natural conversation formatting.
"""

import sys, os, re, json, csv, random, time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import contractions
import spacy
import requests as http_requests
from dataclasses import dataclass
from collections import Counter, defaultdict, deque
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load .env file if present
_ENV_PATH = Path(__file__).resolve().parent / '.env'
if _ENV_PATH.exists():
    with open(_ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _val = _line.split('=', 1)
                os.environ.setdefault(_key.strip(), _val.strip())

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
LEMMATIZER = WordNetLemmatizer()
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
# KEYWORD CATEGORY MAPPER (replaces BART — 340x faster)
# ============================================================
def predict_category_fast(term):
    tl = term.lower()
    if any(w in tl for w in ['food','dish','pasta','pizza','taste','flavor','dessert','meal',
                              'cuisine','ingredient','steak','sushi','cocktail','appetizer',
                              'soup','salad','burger','sandwich','seafood','wine','cocktails',
                              'tiramisu','risotto','noodle','rice','sauce','noodles','pizzas',
                              'desserts','drinks','lobster','chicken','chocolate','bite','sashimi',
                              'beef','bread','cake','cheese','fries','coffee','roll','lamb','pork']):
        return 'food'
    if any(w in tl for w in ['staff','waiter','server','waitress','host','bartender','service',
                              'sommelier','waiters','servers']):
        return 'service'
    if any(w in tl for w in ['price','bill','cost','expensive','cheap','value','money','dollar',
                              'overpriced','prices','deal']):
        return 'price'
    if any(w in tl for w in ['atmosphere','decor','music','ambience','lighting','mood','vibe',
                              'setting','scene','environment']):
        return 'ambience'
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
# DOMAIN KNOWLEDGE BASE — aggregate 3,693 annotations into stats
# ============================================================
def compute_domain_knowledge(train_xml_path):
    root = ET.parse(train_xml_path).getroot()
    cat_polarity = defaultdict(list)
    term_counts = Counter()

    for s in root.findall('.//sentence'):
        ac_node = s.find('aspectCategories')
        at_node = s.find('aspectTerms')
        if ac_node is not None:
            for c in ac_node.findall('aspectCategory'):
                cat_polarity[c.get('category', '').lower()].append(
                    c.get('polarity', '').lower())
        if at_node is not None:
            for a in at_node.findall('aspectTerm'):
                term_counts[a.get('term', '').lower()] += 1

    knowledge = {}
    for category, pols in cat_polarity.items():
        pc = Counter(pols)
        total = len(pols)
        knowledge[category] = {
            'total': total,
            'positive': pc.get('positive', 0),
            'negative': pc.get('negative', 0),
            'neutral': pc.get('neutral', 0),
            'conflict': pc.get('conflict', 0),
            'positive_pct': round(100 * pc.get('positive', 0) / total),
            'negative_pct': round(100 * pc.get('negative', 0) / total),
            'neutral_pct': round(100 * pc.get('neutral', 0) / total),
            'conflict_pct': round(100 * pc.get('conflict', 0) / total),
        }

    knowledge['_overall_total'] = sum(len(p) for p in cat_polarity.values())
    knowledge['_top_terms'] = term_counts.most_common(30)
    return knowledge

print('Computing domain knowledge from training annotations...', flush=True)
DOMAIN_KNOWLEDGE = compute_domain_knowledge(TRAIN_XML)
print(f'Aggregated {DOMAIN_KNOWLEDGE["_overall_total"]} annotations into domain knowledge base.')

# ============================================================
# DOMAIN QUERY PATTERNS & CLASSIFICATION
# ============================================================
DOMAIN_QUERY_PATTERNS = {
    'food':     ['food', 'dish', 'meal', 'eat', 'pizza', 'pasta', 'sushi', 'taste',
                 'dessert', 'desserts', 'cuisine', 'flavor', 'menu', 'appetizer',
                 'steak', 'seafood', 'wine', 'drink', 'drinks', 'cocktail', 'salad',
                 'burger', 'sandwich', 'soup', 'noodle', 'rice', 'sauce', 'bread',
                 'cake', 'coffee', 'chicken', 'seafood', 'cheese'],
    'service':  ['service', 'staff', 'waiter', 'waitress', 'server', 'bartender',
                 'waiters', 'servers', 'sommelier', 'host', 'manager', 'service staff',
                 'service quality', 'wait time', 'wait times', 'waiting'],
    'price':    ['price', 'prices', 'cost', 'expensive', 'cheap', 'value', 'bill',
                 'money', 'overpriced', 'deal', 'budget', 'affordable', 'costs'],
    'ambience': ['ambience', 'atmosphere', 'decor', 'music', 'mood', 'vibe',
                 'lighting', 'noise', 'loud', 'setting', 'scene', 'environment',
                 'interior', 'design', 'cozy', 'romantic'],
}

QUERY_INTENT_WORDS = ['how is', 'how are', 'what do people', 'what about', 'tell me about',
                      'is the', 'are the', 'do people', 'how about', 'what is the',
                      'what are the', 'what do you', 'what is your', 'what about',
                      'any good', 'people like', 'people say', 'people think',
                      'how would you', 'anything to', 'can you tell']

COMPLAINT_KEYWORDS = ['complaint', 'worst', 'bad things', 'disappointing', 'negative about',
                      'most hated', 'what do people hate', 'why do people complain',
                      'common complaints', 'biggest problem', 'biggest problems']

POPULAR_KEYWORDS = ['popular', 'most mentioned', 'common', 'frequent', 'discussed',
                    'most liked', 'most loved', 'best thing', 'best things',
                    'what is popular', 'what are popular']

def classify_domain_query(text):
    text_lower = text.lower()
    tokens = set(re.findall(r'[a-z]+', text_lower))
    tokens_lem = {LEMMATIZER.lemmatize(t) for t in tokens}
    restaurant_terms_lem = {LEMMATIZER.lemmatize(t) for t in RESTAURANT_TERMS}

    has_query_intent = any(q in text_lower for q in QUERY_INTENT_WORDS)
    has_query_intent = has_query_intent or ('?' in text_lower and any(
        k in text_lower for cat_kws in DOMAIN_QUERY_PATTERNS.values() for k in cat_kws))

    if not has_query_intent:
        return None

    # Guard: if query intent detected but zero restaurant words, it's not a domain query
    all_domain_keywords = set()
    for kws in DOMAIN_QUERY_PATTERNS.values():
        all_domain_keywords.update(kws)
    if not (tokens_lem & restaurant_terms_lem) and not any(k in text_lower for k in all_domain_keywords):
        return None

    for cat, keywords in DOMAIN_QUERY_PATTERNS.items():
        if any(k in text_lower for k in keywords):
            return cat

    return None

def answer_domain_query(category):
    stats = DOMAIN_KNOWLEDGE.get(category)
    if not stats:
        return ("I have data on food, service, price, and ambience. "
                "Which would you like to know about?")

    templates = [
        f"Based on {DOMAIN_KNOWLEDGE['_overall_total']} reviews in my training data, "
        f"{category} is rated positively {stats['positive_pct']}% of the time "
        f"and negatively {stats['negative_pct']}% of the time.",

        f"Across my training set, {category} gets positive marks "
        f"{stats['positive_pct']}% of the time ({stats['positive']} out of "
        f"{stats['total']} mentions) — with {stats['negative_pct']}% negative.",

        f"Looking at {stats['total']} {category} mentions in my data: "
        f"{stats['positive_pct']}% positive, {stats['negative_pct']}% negative, "
        f"{stats['neutral_pct']}% neutral.",

        f"From {stats['total']} {category} annotations: most are positive "
        f"({stats['positive_pct']}%), followed by negative ({stats['negative_pct']}%), "
        f"with {stats['neutral_pct']}% neutral.",

        f"My knowledge base shows {category} sentiment is "
        f"{'mostly favorable' if stats['positive_pct'] > 50 else 'mixed'}: "
        f"{stats['positive_pct']}% positive vs {stats['negative_pct']}% negative "
        f"across {stats['total']} mentions.",
    ]
    return random.choice(templates)

def answer_overall_query():
    overall_pos = round(100 * sum(
        DOMAIN_KNOWLEDGE[c]['positive'] for c in CATEGORIES
        if c in DOMAIN_KNOWLEDGE) / DOMAIN_KNOWLEDGE['_overall_total'])
    templates = [
        f"Across {DOMAIN_KNOWLEDGE['_overall_total']} annotations, about "
        f"{overall_pos}% of all restaurant feedback is positive.",
        f"My training data of {DOMAIN_KNOWLEDGE['_overall_total']} reviews shows "
        f"roughly {overall_pos}% positive sentiment overall.",
        f"Overall, {overall_pos}% of annotations in my knowledge base are positive. "
        f"Food and ambience are the highest-rated categories.",
    ]
    return random.choice(templates)

def answer_complaints_query():
    categories_by_neg = sorted(
        [(c, DOMAIN_KNOWLEDGE[c]) for c in CATEGORIES if c in DOMAIN_KNOWLEDGE],
        key=lambda x: x[1]['negative_pct'], reverse=True
    )
    lines = [f"Based on {DOMAIN_KNOWLEDGE['_overall_total']} annotations, "
             f"the most common complaints by category:"]
    for cat, stats in categories_by_neg:
        lines.append(f"  {cat}: {stats['negative_pct']}% negative ({stats['negative']}/{stats['total']})")
    return '\n'.join(lines)

def answer_popular_query():
    top = DOMAIN_KNOWLEDGE['_top_terms'][:10]
    terms = ', '.join(f'"{t}" ({c}x)' for t, c in top)
    return (f"The most mentioned terms in my training data are: {terms}.\n"
            f"Food is the dominant category with {DOMAIN_KNOWLEDGE['food']['total']} annotations "
            f"({DOMAIN_KNOWLEDGE['food']['positive_pct']}% positive).")

# ============================================================
# EXAMINER / SELF-KNOWLEDGE RESPONSES
# ============================================================
EXAMINER_KEYWORDS = {
    'model':      ['model', 'algorithm', 'classifier', 'logistic regression',
                   'tf-idf', 'tfidf', 'what model', 'what algorithm',
                   'what kind of model', 'which model', 'how do you classify'],
    'accuracy':   ['accurate', 'accuracy', 'performance', 'f1', 'f1-score',
                   'f1 score', 'precision', 'recall', 'how well', 'how good are you',
                   'how accurate', 'what is your accuracy'],
    'training':   ['training', 'trained', 'data', 'dataset', 'semeval',
                   'training data', 'what data', 'what dataset', 'what were you trained on',
                   'what corpus', 'which data'],
    'limitations': ['limit', 'limitation', 'weakness', 'weaknesses', 'struggle',
                    'fail', 'fails', 'struggles', 'what can', 'what can\'t',
                    'what are your limitations', 'what do you struggle with',
                    'what are your weaknesses'],
    'sarcasm':    ['sarcasm', 'irony', 'sarcastic', 'ironic', 'handle sarcasm',
                   'sarcasm detection', 'how do you handle irony'],
}

EXAMINER_RESPONSES = {
    'model': (
        "I use a Logistic Regression classifier with TF-IDF features, "
        "trained on 3,693 aspect annotations from the SemEval-2014 restaurant review "
        "corpus. For category mapping, I use a keyword-based classifier that's 340x "
        "faster than the BART zero-shot model I experimented with."
    ),
    'accuracy': (
        "I achieve 70.99% accuracy on the standard test set. My weighted F1 is 0.715. "
        "I'm strongest at detecting positive sentiment (84% F1) but struggle with "
        "neutral reviews (45% F1) and conflict cases (21% F1)."
    ),
    'training': (
        "I was trained on the SemEval-2014 restaurant corpus: 3,041 review sentences "
        "with 3,693 manually annotated aspect terms and 3,713 category annotations. "
        "Each annotation includes the aspect term, its category (food/service/price/ambience), "
        "and polarity (positive/negative/neutral/conflict)."
    ),
    'limitations': (
        "My main limitations: (1) I cannot detect sarcasm — 'Oh great, another cold meal' "
        "reads as positive to me. (2) I can't compare two specific restaurants. "
        "(3) My vocabulary is from 2014 — newer food terms and restaurant names may not "
        "be recognized. (4) Neutral and conflict sentiment are hard to classify accurately "
        "(F1 scores of 45% and 21% respectively)."
    ),
    'sarcasm': (
        "Sarcasm is a known limitation of my system. For example, 'Oh great, another "
        "cold meal' would be classified as positive because I detect the word 'great' "
        "without understanding the ironic context. Handling sarcasm would require "
        "a more sophisticated model like BERT trained on sarcasm-labeled data."
    ),
    'how_it_works': (
        "Here's my pipeline: First, I detect your intent (greeting, review, question, etc.). "
        "For reviews, I extract aspect terms using spaCy noun chunks and a learned lexicon, "
        "then classify each aspect's sentiment with my TF-IDF + Logistic Regression model. "
        "For questions about restaurant stats, I query my pre-computed knowledge base from "
        f"{DOMAIN_KNOWLEDGE['_overall_total']} training annotations."
    ),
    'compare': (
        "I cannot compare two specific restaurants because my training data doesn't include "
        "restaurant identities — only anonymized review sentences. I can tell you general "
        "trends (e.g., food is 70% positive, service is 54% positive), but not which "
        "restaurant is better."
    ),
}

def detect_examiner_intent(text):
    text_lower = text.lower()

    if any(w in text_lower for w in ['how do you work', 'how you work', 'explain yourself',
                                      'describe yourself', 'what are you', 'who are you',
                                      'tell me about yourself', 'how do you operate',
                                      'explain your process', 'what is your purpose']):
        return 'how_it_works'

    if any(w in text_lower for w in ['compare', 'comparing', 'difference between',
                                      'better restaurant', 'which restaurant',
                                      'two restaurants', 'vs']):
        return 'compare'

    for topic, keywords in EXAMINER_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return topic

    return None

def answer_examiner(topic):
    if topic in EXAMINER_RESPONSES:
        return EXAMINER_RESPONSES[topic]
    return ("I'd be happy to tell you about my system! I use Logistic Regression with "
            "TF-IDF features for sentiment analysis, trained on the SemEval-2014 corpus. "
            "Ask me about my model, accuracy, training data, or limitations.")

# ============================================================
# CONVERSATION MEMORY — 3-turn sliding window
# ============================================================
class ConversationMemory:
    def __init__(self, max_turns=3):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)
        self.last_topic = None

    def add_exchange(self, user_msg, bot_msg, intent, topic=None):
        self.history.append({
            'user': user_msg,
            'bot': bot_msg,
            'intent': intent,
            'topic': topic
        })
        if topic:
            self.last_topic = topic

    def get_context(self):
        if not self.history:
            return ''
        recent = self.history[-1]
        return (f'User just asked: "{recent["user"]}". '
                f'The intent was {recent["intent"]}. '
                f'I responded about: {recent["topic"] or "N/A"}.')

    def get_previous_intent(self):
        if self.history:
            return self.history[-1]['intent']
        return None

    def get_previous_topic(self):
        return self.last_topic

MEMORY = ConversationMemory()

# ============================================================
# EXPANDED INTENT DETECTION (6 paths)
# ============================================================
GREETINGS  = {'hi','hello','hey','howdy','greetings','morning','evening',
              'hiya','hola','yo','sup','good afternoon','good evening'}

FAREWELL   = {'bye','goodbye','quit','thanks','thank','see you','later',
              'farewell','cya','cheers','take care','exit','done','stop'}

HELP_PATTERNS = ['help', 'capabilities', 'what can you do', 'how do you work',
                 'what do you do', 'what are you', 'what can i ask', 'how can you',
                 'what is your purpose', 'what are your functions', 'commands',
                 'options', 'features', 'what questions can', 'what kind of',
                 'how does this work', 'explain yourself']

RESTAURANT_TERMS = {
    # Food items & dishes
    'food','meal','dish','menu','taste','flavor','cuisine','ingredient','portion',
    'pizza','pasta','sushi','steak','burger','sandwich','salad','soup','appetizer',
    'dessert','wine','cocktail','drink','coffee','seafood','chicken','beef','pork',
    'lamb','rice','noodle','bread','cake','cheese','fries','sauce','tiramisu',
    'risotto','lobster','roll','sashimi','desserts','drinks','cocktails','noodles',
    # Restaurant places & concepts
    'restaurant','cafe','bistro','diner','eatery','bar','pub','brunch',
    'reservation','table','booking','seating',
    # Service
    'service','staff','waiter','waitress','server','bartender','host','manager',
    'sommelier','waiters','servers','waitstaff',
    # Price
    'price','bill','cost','expensive','cheap','value','money','overpriced','deal',
    'prices','budget','affordable',
    # Ambience
    'ambience','atmosphere','decor','music','lighting','mood','vibe','setting',
    'scene','interior','environment','design',
    # Review words
    'delicious','tasty','yummy','disgusting','bland','fresh','stale','cold',
    'warm','hot','crispy','tender','juicy','dry','burnt','overcooked','raw',
    'friendly','rude','polite','slow','fast','quick','attentive','helpful',
    'noisy','quiet','loud','crowded','cozy','romantic','dirty','clean'
}

def lemmatize_tokens(tokens):
    return {LEMMATIZER.lemmatize(t) for t in tokens}

# Single-word greeting/farewell tokens for fast set intersection
GREETING_TOKENS  = {'hi','hello','hey','howdy','greetings','morning','evening',
                    'hiya','hola','yo','sup'}
FAREWELL_TOKENS  = {'bye','goodbye','quit','thanks','thank','later','farewell',
                    'cya','cheers','exit','done','stop'}
# Multi-word patterns (checked via substring)
GREETING_PHRASES = ['good afternoon', 'good evening']
FAREWELL_PHRASES = ['see you', 'take care']

def detect_intent(text):
    if not text or not text.strip():
        return 'off_domain'

    text_lower = text.lower()
    tokens = set(re.findall(r'[a-z]+', text_lower))
    tokens_lem = lemmatize_tokens(tokens)

    # 1. Examiner / self-knowledge (check first — catches "what are your limitations?" etc.)
    examiner_topic = detect_examiner_intent(text)
    if examiner_topic:
        return 'examiner'

    # 2. Help (check before greeting — "What can you do?" should not be a greeting)
    if any(p in text_lower for p in HELP_PATTERNS):
        return 'help'

    # 3. Domain query ("Is food good?")
    domain_cat = classify_domain_query(text)
    if domain_cat:
        return 'domain_query'

    # 4. Complaint / popular / overall queries
    if any(k in text_lower for k in COMPLAINT_KEYWORDS):
        return 'domain_query'
    if any(k in text_lower for k in POPULAR_KEYWORDS):
        return 'domain_query'
    if any(w in text_lower for w in ['how is it', 'how are things', 'overall',
                                      'in general', 'what do you think', 'opinion']):
        return 'domain_query'

    # 5. Greetings (token intersection, not substring)
    if tokens & GREETING_TOKENS or any(p in text_lower for p in GREETING_PHRASES):
        return 'greeting'

    # 6. Farewells (token intersection + multi-word phrases)
    if tokens & FAREWELL_TOKENS or any(p in text_lower for p in FAREWELL_PHRASES):
        return 'farewell'

    # 7. Restaurant review (expanded keywords + lemmatized tokens)
    restaurant_terms_lem = {LEMMATIZER.lemmatize(t) for t in RESTAURANT_TERMS}
    if tokens_lem & restaurant_terms_lem:
        return 'restaurant_review'

    # 8. spaCy fallback for reviews with unknown food terms
    # Only trigger if text has review-like structure (adjective before/after noun)
    doc = nlp(clean_text(text))
    nouns = [token.text.lower() for token in doc
             if token.pos_ in ('NOUN', 'PROPN') and len(token.text) > 2]
    # At least 2 nouns + review-like linking words (was/is/were + adjective pattern)
    has_review_structure = any(frag in text_lower for frag in [
        ' was ', ' is ', ' were ', ' are ', 'tasted ', 'taste '
    ])
    # Exclude query-like sentences that have nouns but aren't reviews
    is_query = any(text_lower.startswith(q) for q in [
        'what ', 'how ', 'who ', 'when ', 'where ', 'why ', 'tell me ', 'do you ',
        'can you ', 'is the ', 'are the '
    ])
    if nouns and len(tokens) >= 3 and has_review_structure and not is_query:
        return 'restaurant_review'

    # 9. Off-domain
    return 'off_domain'

# ============================================================
# DIVERSIFIED RESPONSE TEMPLATES
# ============================================================
EMOJI = {'positive':'\U0001f60a', 'negative':'\U0001f61e', 'neutral':'\U0001f610', 'conflict':'\U0001f914'}

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
        unique_terms = set()
        for i in items:
            aspect = re.sub(r'^(the|a|an)\s+', '', i['aspect'].strip())
            unique_terms.add(aspect)
        terms = ', '.join(unique_terms)
        lines.append(f'  {EMOJI.get(dominant,"")} {cat.upper()}: {dominant}  (about: {terms})')
    lines.append('\nWant to ask about a specific aspect like food quality or service?')
    return '\n'.join(lines)

def _format_absa_v2(by_cat, results, overall):
    lines = [f'Overall, your review feels {overall} to me. Breaking it down:\n']
    for cat, items in by_cat.items():
        dominant = Counter(i['sentiment'] for i in items).most_common(1)[0][0]
        unique_terms = set()
        for i in items:
            aspect = re.sub(r'^(the|a|an)\s+', '', i['aspect'].strip())
            unique_terms.add(aspect)
        terms = ', '.join(unique_terms)
        lines.append(f'  {cat.capitalize()} ({terms}): {dominant} {EMOJI.get(dominant,"")}')
    lines.append('\nWould you like a deeper breakdown on any of these?')
    return '\n'.join(lines)

def _format_absa_v3(by_cat, results, overall):
    lines = [f'I analysed your review — the overall tone is {overall}.\n']
    for cat, items in by_cat.items():
        items_list = sorted(items, key=lambda x: Counter(i['sentiment'] for i in [x]).get(x['sentiment'], 0))
        for i in items[:2]:
            aspect = re.sub(r'^(the|a|an)\s+', '', i['aspect'].strip())
            lines.append(f'  {EMOJI.get(i["sentiment"],"")} {aspect}: {i["sentiment"]}')
    lines.append('\nCurious about any of these in more detail?')
    return '\n'.join(lines)

def _format_absa_v4(by_cat, results, overall):
    pos_items = [i for i in results if i['sentiment'] == 'positive']
    neg_items = [i for i in results if i['sentiment'] == 'negative']
    lines = []

    if pos_items:
        aspects = ', '.join(set(re.sub(r'^(the|a|an)\s+', '', i['aspect'].strip()) for i in pos_items))
        lines.append(f'\U0001f60a Things you liked: {aspects}')
    if neg_items:
        aspects = ', '.join(set(re.sub(r'^(the|a|an)\s+', '', i['aspect'].strip()) for i in neg_items))
        lines.append(f'\U0001f61e Things you didn\'t like: {aspects}')
    if not lines:
        lines = [f'Overall tone: {overall}']
    lines.append('\nAnything else you\'d like me to analyze?')
    return '\n'.join(lines)

def _format_absa_v5(by_cat, results):
    lines = ['Here is a summary of your review:\n']
    for cat, items in by_cat.items():
        unique_terms = set()
        for i in items:
            aspect = re.sub(r'^(the|a|an)\s+', '', i['aspect'].strip())
            unique_terms.add(aspect)
        terms = ', '.join(unique_terms)
        sentiments = Counter(i['sentiment'] for i in items)
        sent_str = ' / '.join(f'{s}({c}x)' for s, c in sentiments.items())
        lines.append(f'  {cat.capitalize()}: {sent_str}  [{terms}]')
    if len(by_cat) > 1:
        lines.append('\nThat\'s a detailed review — multiple aspects always give a better picture!')
    return '\n'.join(lines)

def general_responses(intent, context=None):
    if intent == 'greeting':
        templates = [
            "Hello! \U0001f44b I am your restaurant review expert chatbot. I can analyse reviews and tell you how people feel about the food, service, price, or ambience. Just type a review or ask a question!",
            "Hi there! I'm a restaurant review analyst. I can evaluate the sentiment in dining reviews, answer questions about what people like and dislike, and explain how my system works. What would you like to explore?",
            "Hey! Welcome to your restaurant review assistant. I can break down the sentiment of any dining experience — tell me about a meal you had, or ask me about trends in restaurant feedback!",
            "Good to see you! I specialise in restaurant review analysis. Share a review and I'll tell you which aspects people loved or hated. Or ask me 'Is the food good?' for insights from my training data.",
            "\U0001f374 Hello! I analyse restaurant reviews. Describe a dining experience and I'll break down the sentiment for each aspect — food, service, price, and ambience. What can I help with?",
        ]
        response = random.choice(templates)
        MEMORY.add_exchange('(greeting)', response, 'greeting')
        return response

    if intent == 'farewell':
        templates = [
            "Thanks for chatting! Hope the insights were helpful. Goodbye! \U0001f44b",
            "Glad I could help with the restaurant analysis. Come back if you have more reviews to share. Bye!",
            "Enjoyed our conversation! When you have more dining experiences to analyse, I'll be here. Take care!",
            "Goodbye! Hope the restaurant insights were useful. Feel free to return anytime. \U0001f44b",
            "Thanks for the chat! Best of luck with your dining adventures. See you next time!",
        ]
        return random.choice(templates)

    if intent == 'help':
        templates = [
            "I can help you with:\n  \u2022 Analysing sentiment in restaurant reviews\n  \u2022 Identifying which aspects are positive or negative\n  \u2022 Answering questions about food, service, price, ambience\n  \u2022 Explaining how my ABSA system works\n\nTry typing:\n  'The pasta was cold but the waiter was friendly'\n  'What do people think about the service?'\n  'What model are you using?'",
            "Here's what I do:\n  \u2022 Sentiment analysis on restaurant reviews (food, service, price, ambience)\n  \u2022 Domain Q&A — 'Is food good?' answers from 3,693 annotated examples\n  \u2022 System explanation — ask me about my model, accuracy, or training data\n\nTry a review like 'The pizza was amazing' or a question like 'How is the service?'",
            "I'm a restaurant review expert! My capabilities:\n  \u2022 ABSA: Extract aspects and classify sentiment (positive/negative/neutral)\n  \u2022 Knowledge Q&A: Stats from 3,693 training annotations\n  \u2022 Self-knowledge: I can explain my architecture and performance\n\nGo ahead, try any restaurant question or share a dining review!",
            "You can ask me to:\n  1. Analyse a review — 'The steak was overcooked but the ambience was lovely'\n  2. Check trends — 'Is the food good?' or 'What do people complain about?'\n  3. Explain myself — 'How accurate are you?' or 'What model do you use?'\n\nWhat would you like to try?",
            "Welcome! I offer:\n  \u2022 Review Analysis — I break down the sentiment of each aspect in your review\n  \u2022 Knowledge Lookup — I have stats on 3,693 restaurant annotations\n  \u2022 System Info — Ask me about my model, accuracy, or limitations\n\nTry something like: 'The service was slow but the food was delicious'",
        ]
        return random.choice(templates)

    if intent == 'off_domain':
        templates = [
            ("I specialise in restaurant review analysis — I'd be more helpful "
             "analysing a dining experience or answering questions about restaurant "
             "trends. Would you like to share a review?"),
            ("That's outside my domain — I focus on restaurant reviews and sentiment "
             "analysis. Try describing a restaurant experience, or ask me 'Is the "
             "food good?' and I'll check my knowledge base."),
            ("I'm a restaurant review chatbot, so I'm not equipped for that. But I can "
             "analyse a dining review or tell you what people tend to say about "
             "restaurant food, service, price, or ambience. Want to try?"),
            ("That's not my area of expertise unfortunately. I analyse restaurant "
             "reviews and answer questions about dining trends. Share a review "
             "like 'The pizza was great' and I'll break it down for you!"),
            ("Sorry, I stick to restaurant reviews! I can break down the sentiment "
             "of any dining experience or answer questions about what people say "
             "about restaurants. Would you like to analyze a review instead?"),
        ]
        return random.choice(templates)

    return ("I'm not sure I understood that. Try typing a restaurant review like:\n"
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
# PHASE 2: OpenRouter LLM Integration
# ============================================================
LLM_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
LLM_MODEL = 'z-ai/glm-4.5-air'
LLM_API_URL = 'https://openrouter.ai/api/v1/chat/completions'
LLM_ENABLED = bool(LLM_API_KEY)
LLM_OFF_DOMAIN_COUNT = {}

def _build_llm_system_prompt():
    stats_text = []
    for cat in ['food', 'service', 'price', 'ambience']:
        if cat in DOMAIN_KNOWLEDGE:
            s = DOMAIN_KNOWLEDGE[cat]
            stats_text.append(
                f"  {cat}: {s['total']} annotations, {s['positive_pct']}% positive, "
                f"{s['negative_pct']}% negative, {s['neutral_pct']}% neutral, "
                f"{s['conflict_pct']}% conflict"
            )

    top_terms = ', '.join(f'"{t}" ({c}x)' for t, c in DOMAIN_KNOWLEDGE['_top_terms'][:15])

    return f"""You are RestaurantXpert, a restaurant review analysis chatbot.

You help users by:
1. Analysing restaurant reviews for sentiment on FOOD, SERVICE, PRICE, and AMBIENCE
2. Answering questions about restaurant review trends using your training statistics
3. Explaining how your ABSA system (model, accuracy, limitations) works

CRITICAL: All facts MUST come from the data below. Never invent restaurant facts, names, or statistics.

DOMAIN KNOWLEDGE (use these exact numbers):
{chr(10).join(stats_text)}

Most mentioned terms: {top_terms}
Overall dataset: {DOMAIN_KNOWLEDGE['_overall_total']} annotations, majority positive.

MODEL FACTS:
- Architecture: TF-IDF vectorizer + SMOTE oversampling + Logistic Regression + keyword category mapper
- Test accuracy: 70.99%, Weighted F1: 0.715
- Strongest class: positive (F1=0.837), Weakest: conflict (F1=0.213), neutral (F1=0.451)
- Training data: SemEval-2014 Task 4, 3,041 sentences, 3,693 aspects
- Category mapping: Keyword-based rule engine (340x faster than BART zero-shot)

GUARD RAILS:
- NEVER give medical, legal, or financial advice
- NEVER make up restaurant names, reviews, or statistics
- If user goes off-domain twice, politely redirect and stop engaging with off-topic queries
- Never engage with harmful, offensive, or NSFW content
- Keep responses to 1-4 sentences, conversational and natural

OFF-DOMAIN RESPONSE: Politely redirect to restaurant review domain. Mention what you can do (analyse reviews, answer restaurant questions, explain your system).

YOUR TASK: Take the structured analysis result I provide and rephrase it naturally. If I give you "intent=greeting", reply with a warm greeting. If I give you "intent=restaurant_review" with analysis results, summarise them conversationally. Always use the facts provided — never make up your own."""


LLM_SYSTEM_PROMPT = _build_llm_system_prompt()


def call_llm(user_message, context, phase1_result, intent):
    if not LLM_ENABLED:
        return _llm_fallback(phase1_result, intent)

    messages = [
        {'role': 'system', 'content': LLM_SYSTEM_PROMPT},
    ]

    if context:
        messages.append({'role': 'assistant', 'content': f'[Previous context] {context}'})

    user_content = f'User message: "{user_message}"\nIntent: {intent}\n'
    if phase1_result:
        user_content += f'Phase 1 result (use these facts only):\n{phase1_result}'
    else:
        user_content += 'No structured result available — respond based on intent only.'

    messages.append({'role': 'user', 'content': user_content})

    try:
        response = http_requests.post(
            LLM_API_URL,
            headers={
                'Authorization': f'Bearer {LLM_API_KEY}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'http://localhost',
                'X-Title': 'RestaurantXpert Chatbot'
            },
            json={
                'model': LLM_MODEL,
                'messages': messages,
                'max_tokens': 300,
                'temperature': 0.7,
            },
            timeout=15
        )

        if response.status_code == 200:
            data = response.json()
            llm_response = data['choices'][0]['message']['content'].strip()
            return _apply_guard_rails(llm_response, user_message, intent)
        else:
            print(f'[LLM API error {response.status_code}]', flush=True)
            return _llm_fallback(phase1_result, intent)

    except Exception as e:
        print(f'[LLM connection error: {e}]', flush=True)
        return _llm_fallback(phase1_result, intent)


def _apply_guard_rails(llm_response, user_message, intent):
    response_lower = llm_response.lower()
    text_lower = user_message.lower()

    forbidden_keywords = [
        'medical advice', 'diagnosis', 'treatment', 'prescription',
        'legal advice', 'sue', 'lawsuit', 'attorney',
        'financial advice', 'invest in', 'stock tip', 'bitcoin price',
        'suicide', 'self-harm', 'illegal', 'hack',
    ]
    for keyword in forbidden_keywords:
        if keyword in response_lower:
            return ("I can't provide advice on that topic. I specialize in restaurant reviews "
                    "and sentiment analysis — would you like to analyze a dining experience instead?")

    if intent == 'off_domain':
        user_id = text_lower[:50]
        LLM_OFF_DOMAIN_COUNT[user_id] = LLM_OFF_DOMAIN_COUNT.get(user_id, 0) + 1
        if LLM_OFF_DOMAIN_COUNT.get(user_id, 0) >= 2:
            return ("I've mentioned this before — I'm a restaurant review analyst. "
                    "I'd be happy to help if you have a dining experience to share or "
                    "a question about restaurant reviews. Otherwise, I should let you find "
                    "a more suitable assistant.")

    return llm_response


def _llm_fallback(phase1_result, intent):
    if phase1_result:
        return phase1_result
    return general_responses(intent)


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
                'category':  predict_category_fast(clean_asp),
                'sentiment': clf.predict(vec)[0]
            })
        except Exception:
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
    context = MEMORY.get_context()

    # Handle each intent path
    if intent == 'restaurant_review':
        absa_results = analyse(user_input)
        phase1_response = format_absa_response(absa_results)
        MEMORY.add_exchange(user_input, phase1_response, intent, topic='restaurant_review')
        try:
            return call_llm(user_input, context, phase1_response, intent)
        except Exception:
            return phase1_response

    elif intent == 'domain_query':
        prev_topic = MEMORY.get_previous_topic()
        text_lower = user_input.lower()

        if any(w in text_lower for w in COMPLAINT_KEYWORDS):
            phase1_response = answer_complaints_query()
            category = 'complaints'
        elif any(w in text_lower for w in POPULAR_KEYWORDS):
            phase1_response = answer_popular_query()
            category = 'popular'
        elif any(w in text_lower for w in ['how is it', 'how are things', 'overall',
                                            'in general', 'what do you think']):
            phase1_response = answer_overall_query()
            category = 'overall'
        elif prev_topic and any(w in text_lower for w in ['what about', 'how about',
                                                            'and the', 'what of']):
            domain_cat = classify_domain_query(user_input)
            if domain_cat:
                phase1_response = answer_domain_query(domain_cat)
                category = domain_cat
            else:
                phase1_response = answer_domain_query(prev_topic)
                category = prev_topic
        else:
            domain_cat = classify_domain_query(user_input)
            if domain_cat:
                phase1_response = answer_domain_query(domain_cat)
                category = domain_cat
            else:
                phase1_response = answer_overall_query()
                category = 'overall'

        MEMORY.add_exchange(user_input, phase1_response, intent, topic=category)
        try:
            return call_llm(user_input, context, phase1_response, intent)
        except Exception:
            return phase1_response

    elif intent == 'examiner':
        examiner_topic = detect_examiner_intent(user_input)
        phase1_response = answer_examiner(examiner_topic)
        MEMORY.add_exchange(user_input, phase1_response, intent, topic=examiner_topic)
        try:
            return call_llm(user_input, context, phase1_response, intent)
        except Exception:
            return phase1_response

    elif intent == 'help':
        phase1_response = general_responses(intent)
        MEMORY.add_exchange(user_input, phase1_response, intent)
        try:
            return call_llm(user_input, context, phase1_response, intent)
        except Exception:
            return phase1_response

    elif intent == 'greeting':
        phase1_response = general_responses(intent)
        try:
            return call_llm(user_input, context, phase1_response, intent)
        except Exception:
            return phase1_response

    elif intent == 'farewell':
        phase1_response = general_responses(intent)
        try:
            return call_llm(user_input, context, phase1_response, intent)
        except Exception:
            return phase1_response

    elif intent == 'off_domain':
        phase1_response = general_responses(intent)
        MEMORY.add_exchange(user_input, phase1_response, intent)
        try:
            return call_llm(user_input, context, phase1_response, intent)
        except Exception:
            return phase1_response

    return general_responses('general')


def analyse_keyword(text):
    """Expose keyword-only analysis for testing (no LLM)."""
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
                'category':  predict_category_fast(clean_asp),
                'sentiment': clf.predict(vec)[0]
            })
        except Exception:
            results.append({
                'aspect':    clean_asp,
                'category':  'miscellaneous',
                'sentiment': 'neutral'
            })
    return results


def chat_keyword(user_input):
    """Keyword-only chat for testing (no LLM wrapping)."""
    if not user_input.strip():
        return 'Please type something!'

    intent = detect_intent(user_input)

    if intent == 'restaurant_review':
        absa_results = analyse_keyword(user_input)
        return format_absa_response(absa_results)
    elif intent == 'domain_query':
        prev_topic = MEMORY.get_previous_topic()
        text_lower = user_input.lower()
        if any(w in text_lower for w in COMPLAINT_KEYWORDS):
            return answer_complaints_query()
        elif any(w in text_lower for w in POPULAR_KEYWORDS):
            return answer_popular_query()
        elif any(w in text_lower for w in ['how is it', 'how are things', 'overall',
                                            'in general', 'what do you think']):
            return answer_overall_query()
        elif prev_topic and any(w in text_lower for w in ['what about', 'how about',
                                                            'and the', 'what of']):
            domain_cat = classify_domain_query(user_input)
            if domain_cat:
                return answer_domain_query(domain_cat)
            return answer_domain_query(prev_topic)
        else:
            domain_cat = classify_domain_query(user_input)
            if domain_cat:
                return answer_domain_query(domain_cat)
            return answer_overall_query()
    elif intent == 'examiner':
        examiner_topic = detect_examiner_intent(user_input)
        return answer_examiner(examiner_topic)
    else:
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

    print('\n--- Aspect Extraction Test ---')
    total = len(test_data.aspects)
    test_sentences = test_data.sentences
    extracted_count = 0
    for _, srow in test_sentences.iterrows():
        ex = extract_aspects_spacy(srow['text'], single_lex, multi_lex, head_lex)
        if ex:
            extracted_count += 1
    print(f'Sentences with at least 1 aspect extracted: {extracted_count}/{len(test_sentences)}')

    print('\n--- Category Mapping Sample ---')
    test_cats = test_data.categories
    cat_predictions = []
    for _, row in test_data.aspects.iterrows():
        cat_predictions.append(predict_category_fast(row['term_normalized']))
    cat_true = []
    for _, row in test_data.aspects.iterrows():
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


results = evaluate_test_set()

# ============================================================
# 50-QUESTION TEST HARNESS
# ============================================================
print('\n' + '='*60)
print('RUNNING 50-QUESTION TEST HARNESS')
print('='*60)

test_questions = [
    ('greeting', 'Hello!'),
    ('greeting', 'Hey there'),
    ('greeting', 'Hi'),
    ('greeting', 'Good morning'),
    ('farewell', 'Goodbye'),
    ('farewell', 'Thanks'),

    ('restaurant_pos', 'The pizza was amazing'),
    ('restaurant_pos', 'The service was excellent'),
    ('restaurant_pos', 'Great food and friendly staff'),
    ('restaurant_pos', 'The desserts were delicious'),
    ('restaurant_pos', 'The waitress was very nice'),
    ('restaurant_pos', 'The ambience was wonderful'),

    ('restaurant_neg', 'The pasta was cold'),
    ('restaurant_neg', 'The waiter was rude to us'),
    ('restaurant_neg', 'The food was terrible and overpriced'),
    ('restaurant_neg', 'The music was too loud'),
    ('restaurant_neg', 'The service was very slow'),
    ('restaurant_neg', 'The bill was outrageous'),

    ('restaurant_mixed', 'The pasta was cold but the waiter was incredibly friendly and fast'),
    ('restaurant_mixed', 'Overpriced for the tiny portions, though the atmosphere was cozy'),
    ('restaurant_mixed', 'Great value and quick service, but the music was too loud to talk'),
    ('restaurant_mixed', 'The staff was rude and the waiting time was too long'),
    ('restaurant_mixed', 'Amazing desserts and the ambience was perfect for a date night'),
    ('restaurant_mixed', 'The risotto was overcooked and the sommelier was rude'),
    ('restaurant_mixed', 'Food was decent but the place was dirty'),
    ('restaurant_mixed', 'Excellent taste but small portions'),

    ('restaurant_query', 'Is the food here good?'),
    ('restaurant_query', 'What do people say about the service?'),
    ('restaurant_query', 'How is the ambience?'),
    ('restaurant_query', 'Tell me about the desserts'),
    ('restaurant_query', 'What about the prices?'),
    ('restaurant_query', 'Are the waiters friendly?'),

    ('help', 'What can you do?'),
    ('help', 'Help'),

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
        response = chat_keyword(question)
    except Exception as e:
        response = f'[ERROR: {e}]'
    test_results.append({
        'type': qtype,
        'input': question,
        'response': response
    })
    short = response[:80].replace('\n', ' | ')
    print(f'[{qtype:20s}] {question[:40]:40s} -> {short}')

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
print(f'Domain knowledge:        {DOMAIN_KNOWLEDGE["_overall_total"]} annotations aggregated')
print(f'Model save location:     {PROJECT_ROOT / "outputs" / "chatbot_model.pkl"}')

type_counts = Counter(r['type'] for r in test_results)
print(f'\nQuestion breakdown:')
for t, c in sorted(type_counts.items()):
    print(f'  {t}: {c}')

errors = [r for r in test_results if r['response'].startswith('[ERROR')]
if errors:
    print(f'\nErrors encountered: {len(errors)}')
    for e in errors:
        print(f'  [{e["type"]}] {e["input"]}: {e["response"]}')
else:
    print(f'\nNo runtime errors encountered.')

print('\nDone!')
