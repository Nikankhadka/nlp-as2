#!/usr/bin/env python3
"""
Comprehensive Chatbot Test Harness — 72 questions across 12 categories
Updated: BART removed, 6-path intent system, domain Q&A, examiner responses.
Runs keyword-only (no LLM wrapping for deterministic testing).
"""

import sys, os, re, pickle, csv, time, random
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import contractions
import spacy
from collections import Counter, defaultdict
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR    = PROJECT_ROOT / 'outputs'
MODEL_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================
print('=' * 70)
print('LOADING TRAINED MODEL')
print('=' * 70)
with open(MODEL_DIR / 'chatbot_model.pkl', 'rb') as f:
    model_state = pickle.load(f)
tfidf      = model_state['tfidf']
clf        = model_state['clf']
single_lex = model_state['single_lex']
multi_lex  = model_state['multi_lex']
head_lex   = model_state['head_lex']
print(f'Model loaded. Test accuracy: {model_state.get("accuracy", "N/A")}')
print(f'Single lexicon: {len(single_lex)} | Multi lexicon: {len(multi_lex)} | Head lexicon: {len(head_lex)}')

# ============================================================
# SPACY & NLTK
# ============================================================
nlp        = spacy.load('en_core_web_sm')
STOPWORDS  = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
CATEGORIES = ['food', 'service', 'price', 'ambience', 'miscellaneous']

# ============================================================
# TEXT CLEANING (identical to chatbot)
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
# ASPECT EXTRACTION
# ============================================================
NON_ASPECTS = set([
    'good','great','bad','excellent','amazing','terrible',
    'delicious','friendly','nice','love','loved','horrible',
    'poor','slow','wonderful','best','worst','perfect','awful',
    'unhelpful','overcooked','undercooked','rude','cold','hot',
    'loud','dirty','clean','fresh','stale','burnt','raw'
])

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
# KEYWORD CATEGORY MAPPER (expanded, matches run_chatbot.py)
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
# DOMAIN KNOWLEDGE (load from XML for stats)
# ============================================================
TRAIN_XML = PROJECT_ROOT / 'data' / 'raw' / 'Restaurants_Train_v2.xml'

def compute_domain_knowledge(train_xml_path):
    root = ET.parse(train_xml_path).getroot()
    cat_polarity = defaultdict(list)
    term_counts = Counter()
    for s in root.findall('.//sentence'):
        ac_node = s.find('aspectCategories')
        at_node = s.find('aspectTerms')
        if ac_node is not None:
            for c in ac_node.findall('aspectCategory'):
                cat_polarity[c.get('category', '').lower()].append(c.get('polarity', '').lower())
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

print('Computing domain knowledge...', flush=True)
DOMAIN_KNOWLEDGE = compute_domain_knowledge(TRAIN_XML)
print(f'Aggregated {DOMAIN_KNOWLEDGE["_overall_total"]} annotations.')

# ============================================================
# DOMAIN QUERY CLASSIFICATION
# ============================================================
DOMAIN_QUERY_PATTERNS = {
    'food':     ['food', 'dish', 'meal', 'eat', 'pizza', 'pasta', 'sushi', 'taste',
                 'dessert', 'desserts', 'cuisine', 'flavor', 'menu', 'appetizer',
                 'steak', 'seafood', 'wine', 'drink', 'drinks', 'cocktail', 'salad',
                 'burger', 'sandwich', 'soup', 'noodle', 'rice', 'sauce', 'bread',
                 'cake', 'coffee', 'chicken', 'seafood', 'cheese'],
    'service':  ['service', 'staff', 'waiter', 'waitress', 'server', 'bartender',
                 'waiters', 'servers', 'sommelier', 'host', 'manager'],
    'price':    ['price', 'prices', 'cost', 'expensive', 'cheap', 'value', 'bill',
                 'money', 'overpriced', 'deal', 'budget', 'affordable', 'costs'],
    'ambience': ['ambience', 'atmosphere', 'decor', 'music', 'mood', 'vibe',
                 'lighting', 'noise', 'loud', 'setting', 'scene', 'environment',
                 'interior', 'design', 'cozy', 'romantic'],
}

QUERY_INTENT_WORDS = ['how is', 'how are', 'what do people', 'what about', 'tell me about',
                      'is the', 'are the', 'do people', 'how about', 'what is the',
                      'what are the', 'what do you', 'what is your', 'what about',
                      'any good', 'people like', 'people say', 'people think']

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
        return "I have data on food, service, price, and ambience."
    return (f"Based on {DOMAIN_KNOWLEDGE['_overall_total']} reviews in my training data, "
            f"{category} is rated positively {stats['positive_pct']}% of the time "
            f"and negatively {stats['negative_pct']}% of the time.")

# ============================================================
# EXAMINER / SELF-KNOWLEDGE
# ============================================================
EXAMINER_KEYWORDS = {
    'model':      ['model', 'algorithm', 'classifier', 'logistic regression',
                   'tf-idf', 'tfidf', 'what model', 'what algorithm',
                   'which model', 'how do you classify'],
    'accuracy':   ['accurate', 'accuracy', 'performance', 'f1', 'f1-score',
                   'f1 score', 'how accurate', 'what is your accuracy'],
    'training':   ['training', 'trained', 'data', 'dataset', 'semeval',
                   'training data', 'what data', 'what dataset', 'what were you trained on'],
    'limitations': ['limit', 'limitation', 'weakness', 'weaknesses', 'struggle',
                    'fail', 'fails', 'struggles', 'what can', 'what can\'t',
                    'what are your limitations'],
    'sarcasm':    ['sarcasm', 'irony', 'sarcastic', 'ironic', 'handle sarcasm'],
}

EXAMINER_RESPONSES = {
    'model': "Logistic Regression with TF-IDF features, trained on 3,693 annotations from SemEval-2014.",
    'accuracy': "70.99% accuracy on test set, weighted F1: 0.715. Strongest on positive (84% F1).",
    'training': "SemEval-2014 restaurant corpus: 3,041 sentences, 3,693 annotated aspect terms.",
    'limitations': "Cannot detect sarcasm, neutral F1=45%, conflict F1=21%, 2014 vocabulary.",
    'sarcasm': "Sarcasm is a known limitation. 'Oh great, another cold meal' would be misclassified as positive.",
    'how_it_works': "Intent detection -> aspect extraction -> TF-IDF + Logistic Regression -> category mapping.",
    'compare': "Cannot compare specific restaurants — no restaurant identities in training data.",
}

def detect_examiner_intent(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ['how do you work', 'how you work', 'explain yourself',
                                      'what are you', 'who are you', 'how do you operate']):
        return 'how_it_works'
    if any(w in text_lower for w in ['compare', 'comparing', 'two restaurants', 'vs']):
        return 'compare'
    for topic, keywords in EXAMINER_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return topic
    return None

def answer_examiner(topic):
    if topic in EXAMINER_RESPONSES:
        return EXAMINER_RESPONSES[topic]
    return "I use Logistic Regression with TF-IDF, trained on SemEval-2014."

# ============================================================
# CONVERSATION MEMORY (simplified for testing)
# ============================================================
_last_topic = None
def set_last_topic(topic):
    global _last_topic
    _last_topic = topic
def get_last_topic():
    return _last_topic

# ============================================================
# EXPANDED INTENT DETECTION (6 paths)
# ============================================================
GREETINGS  = {'hi','hello','hey','howdy','greetings','morning','evening',
              'hiya','hola','yo','sup','good afternoon','good evening'}
FAREWELL   = {'bye','goodbye','quit','thanks','thank','see you','later',
              'farewell','cya','cheers','take care','exit','done','stop'}
HELP_PATTERNS = ['help', 'capabilities', 'what can you do', 'how do you work',
                 'what do you do', 'what are you', 'what can i ask', 'how can you',
                 'what is your purpose', 'features', 'commands', 'options']

RESTAURANT_TERMS = {
    'food','meal','dish','menu','taste','flavor','cuisine','ingredient','portion',
    'pizza','pasta','sushi','steak','burger','sandwich','salad','soup','appetizer',
    'dessert','wine','cocktail','drink','coffee','seafood','chicken','beef','pork',
    'lamb','rice','noodle','bread','cake','cheese','fries','sauce','tiramisu',
    'risotto','lobster','roll','sashimi','desserts','drinks','cocktails','noodles',
    'restaurant','cafe','bistro','diner','eatery','bar','pub','brunch',
    'reservation','table','booking','seating',
    'service','staff','waiter','waitress','server','bartender','host','manager',
    'sommelier','waiters','servers','waitstaff',
    'price','bill','cost','expensive','cheap','value','money','overpriced','deal',
    'prices','budget','affordable',
    'ambience','atmosphere','decor','music','lighting','mood','vibe','setting',
    'scene','interior','environment','design',
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
GREETING_PHRASES = ['good afternoon', 'good evening']
FAREWELL_PHRASES = ['see you', 'take care']

def detect_intent(text):
    if not text or not text.strip():
        return 'off_domain'

    text_lower = text.lower()
    tokens = set(re.findall(r'[a-z]+', text_lower))
    tokens_lem = lemmatize_tokens(tokens)

    # 1. Examiner (first — "what are your limitations?" is not help)
    examiner_topic = detect_examiner_intent(text)
    if examiner_topic:
        return 'examiner'

    # 2. Help
    if any(p in text_lower for p in HELP_PATTERNS):
        return 'help'

    # 3. Domain query
    domain_cat = classify_domain_query(text)
    if domain_cat:
        return 'domain_query'

    # 4. Greetings (token intersection, not substring)
    if tokens & GREETING_TOKENS or any(p in text_lower for p in GREETING_PHRASES):
        return 'greeting'

    # 5. Farewells
    if tokens & FAREWELL_TOKENS or any(p in text_lower for p in FAREWELL_PHRASES):
        return 'farewell'

    # 6. Restaurant review
    restaurant_terms_lem = {LEMMATIZER.lemmatize(t) for t in RESTAURANT_TERMS}
    if tokens_lem & restaurant_terms_lem:
        return 'restaurant_review'

    # 7. spaCy fallback (only for structured review-like text)
    doc = nlp(clean_text(text))
    nouns = [token.text.lower() for token in doc
             if token.pos_ in ('NOUN', 'PROPN') and len(token.text) > 2]
    has_review_structure = any(frag in text_lower for frag in [
        ' was ', ' is ', ' were ', ' are ', 'tasted ', 'taste '
    ])
    is_query = any(text_lower.startswith(q) for q in [
        'what ', 'how ', 'who ', 'when ', 'where ', 'why ', 'tell me ', 'do you ',
        'can you ', 'is the ', 'are the '
    ])
    if nouns and len(tokens) >= 3 and has_review_structure and not is_query:
        return 'restaurant_review'

    return 'off_domain'

# ============================================================
# FORMATTING FUNCTIONS
# ============================================================
EMOJI = {'positive':'\U0001f60a', 'negative':'\U0001f61e', 'neutral':'\U0001f610', 'conflict':'\U0001f914'}

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
        return ("Hello! \U0001f44b I am your restaurant review expert chatbot.\n"
                "I can analyse reviews and tell you how people feel about\n"
                "the food, service, price, or ambience.\n"
                "Just type a review or ask a question!")
    if intent == 'farewell':
        return 'Thanks for chatting! Hope the insights were helpful. Goodbye! \U0001f44b'
    if intent == 'help':
        return ('I can help you with:\n'
                '  \u2022 Analysing sentiment in restaurant reviews\n'
                '  \u2022 Identifying which aspects are positive or negative\n'
                '  \u2022 Answering questions about food, service, price, ambience\n\n'
                'Try typing:\n'
                '  "The pasta was cold but the waiter was friendly"\n'
                '  "What do people think about the service?"')
    if intent == 'off_domain':
        return ("I specialise in restaurant review analysis. "
                "Try typing a restaurant review like:\n"
                '  "The pasta was cold but the waiter was friendly"')
    return ('I am not sure I understood that.\n'
            'Try typing a restaurant review like:\n'
            '  "The pasta was cold but the waiter was friendly"')

# ============================================================
# ANALYSE
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

    if intent == 'restaurant_review':
        return format_absa_response(analyse(user_input))

    elif intent == 'domain_query':
        domain_cat = classify_domain_query(user_input)
        if domain_cat:
            set_last_topic(domain_cat)
            return answer_domain_query(domain_cat)
        return answer_domain_query('food')

    elif intent == 'examiner':
        examiner_topic = detect_examiner_intent(user_input)
        return answer_examiner(examiner_topic)

    elif intent == 'help':
        return general_responses('help')

    elif intent in ('greeting', 'farewell', 'off_domain'):
        return general_responses(intent)

    return general_responses('general')

# ============================================================
# JUDGEMENT RULES (updated for new intent/response patterns)
# ============================================================
def judge(qtype, question, response):
    r = response.lower()
    q = question.lower()

    if qtype in ('greeting', 'farewell'):
        if any(w in r for w in ['hello', 'restaurant review', 'review expert', 'thanks',
                                 'goodbye', 'helpful', 'chat', 'restaurant analyst',
                                 'dining', 'analyse', 'analyze']):
            return 'correct'
        return 'wrong'

    if qtype == 'help':
        if any(w in r for w in ['can help', 'analys', 'analysis', 'capabilities',
                                 'sentiment', 'absa', 'reviews', 'aspect',
                                 'model', 'knowledge', 'trends']):
            return 'correct'
        if 'not sure' in r:
            return 'wrong'
        return 'partial'

    if qtype in ('restaurant_pos', 'restaurant_neg', 'restaurant_mixed'):
        if 'here is what i found' in r or 'aspect' in r.lower():
            if qtype == 'restaurant_pos' and ('positive' in r or 'you liked' in r or 'things you'):
                return 'correct'
            if qtype == 'restaurant_neg' and ('negative' in r or "didn't like" in r):
                return 'correct'
            if qtype == 'restaurant_mixed':
                return 'correct'
            if 'here is what i found' in r:
                return 'partial'
            return 'partial'
        if 'not sure' in r or 'could not identify' in r:
            return 'wrong'
        return 'wrong'

    if qtype == 'restaurant_query':
        if any(w in r for w in ['based on', 'training data', 'knowledge base',
                                 '% positive', '% negative', 'rated positively',
                                 'positive', 'negative', 'annotations']):
            return 'correct'
        if 'here is what i found' in r:
            return 'partial'
        if 'not sure' in r:
            return 'wrong'
        return 'partial'

    if qtype == 'domain_query':
        if any(w in r for w in ['based on', 'training data', 'knowledge base',
                                 '% positive', '% negative', 'rated positively',
                                 'annotations', 'reviews']):
            return 'correct'
        if 'not sure' in r:
            return 'wrong'
        return 'partial'

    if qtype == 'general':
        if any(w in r for w in ['not sure', "i'm a restaurant", 'specialise', 'specialize',
                                 'restaurant review', 'outside my', 'not equipped',
                                 'restaurant reviews', 'analyze a dining']):
            return 'correct'
        return 'wrong'

    if qtype == 'edge':
        if not q.strip():
            return 'correct' if 'please type' in r else 'wrong'
        if q.strip().lower() in ('pizza', 'spagetti'):
            if any(w in r for w in ['here is what i found', 'food', 'positive', 'negative',
                                     'not sure', 'restaurant']):
                return 'correct'
            return 'partial'
        return 'correct' if ('not sure' in r or 'please type' in r) else 'partial'

    if qtype == 'examiner':
        if any(w in r for w in ['logistic', 'model', 'tf-idf', 'tfidf', 'accuracy', '70',
                                 'semeval', 'trained', 'limitation', 'sarcasm',
                                 'f1', 'annotations', 'regression', 'classifier']):
            return 'correct'
        if 'not sure' in r:
            return 'wrong'
        return 'partial'

    if qtype == 'negation':
        if any(w in r for w in ['here is what i found', 'aspect']):
            return 'correct'
        if 'not sure' in r:
            return 'wrong'
        return 'partial'

    if qtype == 'long_review':
        if 'here is what i found' in r:
            return 'correct'
        if 'not sure' in r:
            return 'wrong'
        return 'partial'

    return 'unknown'

# ============================================================
# 72 TEST QUESTIONS ACROSS 12 CATEGORIES
# ============================================================
test_questions = [
    # ── 1. GREETINGS (5) ──
    ('greeting', 'Hello!'),
    ('greeting', 'Hi there'),
    ('greeting', 'Good morning'),
    ('greeting', 'Hey!'),
    ('greeting', 'Howdy'),

    # ── 2. FAREWELLS (3) ──
    ('farewell', 'Goodbye'),
    ('farewell', 'Bye'),
    ('farewell', 'Thanks for your help'),

    # ── 3. SIMPLE POSITIVE (6) ──
    ('restaurant_pos', 'The pizza was amazing'),
    ('restaurant_pos', 'The service was excellent'),
    ('restaurant_pos', 'The desserts were delicious'),
    ('restaurant_pos', 'The waitress was very nice'),
    ('restaurant_pos', 'The ambience was wonderful'),
    ('restaurant_pos', 'Best pasta I have ever had'),

    # ── 4. SIMPLE NEGATIVE (6) ──
    ('restaurant_neg', 'The pasta was cold'),
    ('restaurant_neg', 'The waiter was rude to us'),
    ('restaurant_neg', 'The food was terrible and overpriced'),
    ('restaurant_neg', 'The music was too loud'),
    ('restaurant_neg', 'The service was very slow'),
    ('restaurant_neg', 'This place is dirty'),

    # ── 5. MIXED SENTIMENT (8) ──
    ('restaurant_mixed', 'The pasta was cold but the waiter was incredibly friendly and fast'),
    ('restaurant_mixed', 'Overpriced for the tiny portions, though the atmosphere was cozy'),
    ('restaurant_mixed', 'Great value and quick service, but the music was too loud to talk'),
    ('restaurant_mixed', 'The staff was rude and the waiting time was too long'),
    ('restaurant_mixed', 'Amazing desserts and the ambience was perfect for a date night'),
    ('restaurant_mixed', 'The risotto was overcooked and the sommelier was rude'),
    ('restaurant_mixed', 'Food was decent but the place was dirty'),
    ('restaurant_mixed', 'Excellent taste but small portions'),

    # ── 6. DOMAIN QUERIES (7) ──
    ('domain_query', 'Is the food here good?'),
    ('domain_query', 'What do people say about the service?'),
    ('domain_query', 'How is the ambience?'),
    ('domain_query', 'Tell me about the desserts'),
    ('domain_query', 'What about the prices?'),
    ('domain_query', 'Are the waiters friendly?'),
    ('domain_query', 'How is the wine selection?'),

    # ── 7. HELP / CAPABILITIES (4) ──
    ('help', 'What can you do?'),
    ('help', 'Help'),
    ('help', 'What are your capabilities?'),
    ('help', 'How do you work?'),

    # ── 8. GENERAL / OFF-DOMAIN (10) ──
    ('general', 'What is the weather like today?'),
    ('general', 'Tell me about the football game'),
    ('general', 'How do I fix my car?'),
    ('general', 'What is the meaning of life?'),
    ('general', 'Who won the election?'),
    ('general', 'Tell me a joke'),
    ('general', 'What time is it?'),
    ('general', 'How old are you?'),
    ('general', 'Can you recommend a movie?'),
    ('general', 'What is the capital of France?'),

    # ── 9. EDGE CASES (8) ──
    ('edge', ''),
    ('edge', 'Pizza'),
    ('edge', 'spagetti'),
    ('edge', 'A'),
    ('edge', '12345'),
    ('edge', '!!!!'),
    ('edge', 'Not bad'),
    ('edge', 'The'),

    # ── 10. EXAMINER QUESTIONS (6) ──
    ('examiner', 'What model are you using for sentiment analysis?'),
    ('examiner', 'How accurate are your predictions?'),
    ('examiner', 'What training data did you use?'),
    ('examiner', 'What are your limitations?'),
    ('examiner', 'How do you handle sarcasm?'),
    ('examiner', 'Compare the food at two different restaurants'),

    # ── 11. NEGATION / LINGUISTIC (5) ──
    ('negation', 'Not bad at all'),
    ('negation', 'I would not say it was bad'),
    ('negation', 'The food was not great'),
    ('negation', 'Oh great another cold meal'),
    ('negation', "It's not the worst but it could be better"),

    # ── 12. LONG / COMPLEX REVIEWS (4) ──
    ('long_review', 'We went for dinner last night and had the most amazing steak cooked perfectly medium rare with a side of truffle fries that were crispy and delicious'),
    ('long_review', 'The appetizers were cold when they arrived, the main course took forty five minutes to come out, and when we complained the manager was completely unhelpful and dismissive'),
    ('long_review', 'The cocktails were innovative and well-crafted, the sushi was fresh and beautifully presented, but the dessert menu was disappointing and the waiter seemed distracted all evening'),
    ('long_review', 'Arrived at seven, seated by eight, ordered by eight thirty, food arrived at nine fifteen, no apology, no discount, the pasta was lukewarm and the wine was served at room temperature on a hot summer night'),
]

# ============================================================
# RUN TESTS — KEYWORD-ONLY (no BART, no LLM)
# ============================================================
print('\n' + '=' * 70)
print('RUNNING 72 QUESTIONS WITH IMPROVED 6-PATH INTENT SYSTEM')
print('=' * 70)

results_list = []
for i, (qtype, question) in enumerate(test_questions, 1):
    start = time.time()
    try:
        response = chat(question)
    except Exception as e:
        response = f'[ERROR: {e}]'
    elapsed = time.time() - start
    verdict = judge(qtype, question, response)

    results_list.append({
        'num': i,
        'type': qtype,
        'input': question,
        'response': response,
        'time_s': round(elapsed, 3),
        'verdict': verdict
    })

    short_r = response.replace('\n', ' // ')[:100]
    status = 'PASS' if verdict == 'correct' else ('FAIL' if verdict == 'wrong' else 'PART')
    print(f'[{i:2d}/72] [{qtype:18s}] {question[:45]:45s} | {status:4s} | {short_r}')

# ============================================================
# SAVE RESULTS
# ============================================================
print('\n' + '=' * 70)
print('SAVING RESULTS')
print('=' * 70)

df = pd.DataFrame(results_list)
csv_path = MODEL_DIR / 'comprehensive_test_results.csv'
df.to_csv(csv_path, index=False)
print(f'Saved to {csv_path}')

# Also save as after_improvements.csv
after_path = MODEL_DIR / 'after_improvements.csv'
df.to_csv(after_path, index=False)
print(f'Saved to {after_path}')

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print('\n' + '=' * 70)
print('SUMMARY STATISTICS')
print('=' * 70)

total = len(results_list)
correct = sum(1 for r in results_list if r['verdict'] == 'correct')
partial = sum(1 for r in results_list if r['verdict'] == 'partial')
wrong   = sum(1 for r in results_list if r['verdict'] == 'wrong')
passable = correct + partial

print(f'\nImproved System (all 72 questions):')
print(f'  Correct:  {correct}/{total} ({100*correct/total:.0f}%)')
print(f'  Partial:  {partial}/{total} ({100*partial/total:.0f}%)')
print(f'  Wrong:    {wrong}/{total} ({100*wrong/total:.0f}%)')
print(f'  Passable: {passable}/{total} ({100*passable/total:.0f}%)')

avg_time = np.mean([r['time_s'] for r in results_list])
print(f'  Avg response time: {avg_time:.3f}s (keyword-only, no BART)')

print('\n--- Results by Category ---')
by_type = defaultdict(list)
for r in results_list:
    by_type[r['type']].append(r)
for t in sorted(by_type):
    items = by_type[t]
    c = sum(1 for r in items if r['verdict'] == 'correct')
    p = sum(1 for r in items if r['verdict'] == 'partial')
    w = sum(1 for r in items if r['verdict'] == 'wrong')
    print(f'  {t:20s}: {c}/{len(items)} correct, {p} partial, {w} wrong  -> {(c+p)/len(items)*100:.0f}% passable')

print('\n--- FAILURES ---')
for r in results_list:
    if r['verdict'] == 'wrong':
        short_r = r['response'].replace('\n', ' // ')[:120]
        print(f'  [{r["type"]}] {r["input"][:50]:50s} -> {short_r}')

print('\nDone!')
