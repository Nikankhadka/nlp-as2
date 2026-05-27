#!/usr/bin/env python3
"""
Comprehensive Chatbot Test Harness — 72 questions across 12 categories
Runs both keyword-mapper and BART-mapper paths, records outcomes.
"""

import sys, os, re, pickle, csv, time
import numpy as np
import pandas as pd
import contractions
import spacy
from collections import Counter, defaultdict
from pathlib import Path
from nltk.corpus import stopwords

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
# KEYWORD CATEGORY MAPPER (fast)
# ============================================================
def predict_category_fast(term):
    tl = term.lower()
    if any(w in tl for w in ['food','dish','pasta','pizza','taste','flavor','dessert','meal','cuisine','ingredient','steak','sushi','cocktail','appetizer','soup','salad','burger','sandwich','seafood','wine','cocktails','tiramisu']): return 'food'
    if any(w in tl for w in ['staff','waiter','server','waitress','host','bartender','service','sommelier','waiters']): return 'service'
    if any(w in tl for w in ['price','bill','cost','expensive','cheap','value','money','dollar','overpriced','prices']): return 'price'
    if any(w in tl for w in ['atmosphere','decor','music','ambience','lighting','mood','vibe','setting']): return 'ambience'
    return 'miscellaneous'

# ============================================================
# BART CATEGORY MAPPER (slow, loads only if requested)
# ============================================================
zero_shot = None
BART_LOADED = False

def load_bart():
    global zero_shot, BART_LOADED
    if BART_LOADED:
        return
    print('\nLoading BART zero-shot classifier (1.6GB)...', flush=True)
    from transformers import pipeline as hf_pipeline
    zero_shot = hf_pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=-1)
    BART_LOADED = True
    print('BART loaded.', flush=True)

def predict_category_bart(term):
    try:
        result = zero_shot(term, CATEGORIES)
        return result['labels'][0]
    except Exception:
        return predict_category_fast(term)

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

def detect_intent(text, use_bart=False):
    tokens = set(re.findall(r'[a-z]+', text.lower()))
    if tokens & GREETINGS:  return 'greeting'
    if tokens & FAREWELL:   return 'farewell'
    if tokens & HELP_WORDS: return 'help'

    doc   = nlp(clean_text(text))
    nouns = [token.text.lower() for token in doc
             if token.pos_ in ('NOUN', 'PROPN') and len(token.text) > 2]
    if nouns:
        if use_bart and BART_LOADED:
            try:
                result = zero_shot(text, ['restaurant review', 'general conversation'])
                if result['labels'][0] == 'restaurant review':
                    return 'restaurant_query'
            except Exception:
                pass
        # KEYWORD FALLBACK (original 17-word set — INTENTIONALLY small for comparison)
        basic = {
            'food','meal','dish','menu','taste','service','staff',
            'waiter','price','bill','cost','ambience','atmosphere',
            'restaurant','table','reservation','drink','wine'
        }
        if tokens & basic:
            return 'restaurant_query'
    return 'general'

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
        return ('Hello! \U0001f44b I am your restaurant review expert chatbot.\n'
                'I can analyse reviews and tell you how people feel about\n'
                'the food, service, price, or ambience.\n'
                'Just type a review or ask a question!')
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
    return ('I am not sure I understood that.\n'
            'Try typing a restaurant review like:\n'
            '  "The pasta was cold but the waiter was friendly"')

# ============================================================
# ANALYSE
# ============================================================
def analyse(text, use_bart=False):
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
            cat_fn = predict_category_bart if (use_bart and BART_LOADED) else predict_category_fast
            results.append({
                'aspect':    clean_asp,
                'category':  cat_fn(clean_asp),
                'sentiment': clf.predict(vec)[0]
            })
        except Exception:
            results.append({
                'aspect':    clean_asp,
                'category':  'miscellaneous',
                'sentiment': 'neutral'
            })
    return results

def chat(user_input, use_bart=False):
    if not user_input.strip():
        return 'Please type something!'
    intent = detect_intent(user_input, use_bart=use_bart)
    if intent == 'restaurant_query':
        return format_absa_response(analyse(user_input, use_bart=use_bart))
    return general_responses(intent)

# ============================================================
# JUDGEMENT RULES
# ============================================================
def judge(qtype, question, response):
    """Classify response as correct, partial, or wrong."""
    r = response.lower()
    q = question.lower()

    if qtype in ('greeting', 'farewell'):
        if 'hello' in r or 'restaurant review' in r or 'thanks' in r or 'goodbye' in r:
            return 'correct'
        return 'wrong'

    if qtype == 'help':
        if 'can help' in r or 'analys' in r or 'analysis' in r:
            return 'correct'
        if 'not sure' in r:
            return 'wrong'
        return 'partial'

    if qtype in ('restaurant_pos', 'restaurant_neg', 'restaurant_mixed'):
        if 'here is what i found' in r or 'aspect' in r.lower():
            # Check if sentiment direction is reasonable
            if qtype == 'restaurant_pos' and ('positive' in r or '\U0001f60a' in r):
                return 'correct'
            if qtype == 'restaurant_neg' and ('negative' in r or '\U0001f61e' in r):
                return 'correct'
            if qtype == 'restaurant_mixed':
                return 'correct'  # at least it ran ABSA
            if 'here is what i found' in r:
                return 'partial'  # ran ABSA but sentiment wrong
            return 'partial'
        if 'not sure' in r or 'could not identify' in r:
            return 'wrong'  # intent detection failed
        return 'wrong'

    if qtype == 'restaurant_query':
        if 'here is what i found' in r:
            return 'correct'
        if 'not sure' in r:
            return 'wrong'
        return 'partial'

    if qtype == 'general':
        if 'not sure' in r and ('restaurant review' in r or 'pasta' in r):
            return 'correct'  # correct rejection
        if 'not sure' in r:
            return 'correct'
        return 'wrong'  # shouldn't engage with off-domain

    if qtype == 'edge':
        if not q.strip():
            return 'correct' if 'please type' in r else 'wrong'
        if q.strip().lower() in ('pizza', 'spagetti'):
            if 'here is what i found' in r or 'food' in r or 'positive' in r or 'negative' in r:
                return 'correct'
            if 'not sure' in r:
                return 'partial'  # single word should ideally be handled
            return 'partial'
        return 'correct' if 'not sure' in r or 'please type' in r else 'partial'

    if qtype == 'examiner':
        # Examiner questions about the system itself
        if 'not sure' in r:
            return 'wrong'
        if any(w in r for w in ['model', 'logistic', 'sentiment', 'analy', 'trained', 'accuracy', 'limitation', 'semeval']):
            return 'correct'
        return 'partial'

    if qtype == 'negation':
        if 'here is what i found' in r:
            return 'correct'  # at least it tried ABSA
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

    # ── 6. RESTAURANT QUERIES (7) ──
    ('restaurant_query', 'Is the food here good?'),
    ('restaurant_query', 'What do people say about the service?'),
    ('restaurant_query', 'How is the ambience?'),
    ('restaurant_query', 'Tell me about the desserts'),
    ('restaurant_query', 'What about the prices?'),
    ('restaurant_query', 'Are the waiters friendly?'),
    ('restaurant_query', 'How is the wine selection?'),

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
# RUN TESTS — KEYWORD MAPPER FIRST (fast)
# ============================================================
print('\n' + '=' * 70)
print('PHASE 1: RUNNING 72 QUESTIONS WITH KEYWORD MAPPER')
print('=' * 70)

keyword_results = []
for i, (qtype, question) in enumerate(test_questions, 1):
    start = time.time()
    try:
        response = chat(question, use_bart=False)
    except Exception as e:
        response = f'[ERROR: {e}]'
    elapsed = time.time() - start
    verdict = judge(qtype, question, response)

    keyword_results.append({
        'num': i,
        'type': qtype,
        'input': question,
        'response_keyword': response,
        'keyword_time_s': round(elapsed, 3),
        'keyword_verdict': verdict
    })

    # Print progress
    short_r = response.replace('\n', ' // ')[:90]
    print(f'[{i:2d}/72] [{qtype:18s}] {question[:45]:45s} | {"PASS" if verdict=="correct" else "FAIL" if verdict=="wrong" else "PART":4s} | {short_r}')

# ============================================================
# RUN TESTS — BART MAPPER (slower, restaurant queries only)
# ============================================================
print('\n' + '=' * 70)
print('PHASE 2: LOADING BART AND RUNNING RESTAURANT QUERIES WITH BART')
print('=' * 70)
load_bart()

bart_results = {}
for i, (qtype, question) in enumerate(test_questions, 1):
    if qtype not in ('restaurant_pos', 'restaurant_neg', 'restaurant_mixed',
                     'restaurant_query', 'negation', 'long_review', 'examiner'):
        continue
    start = time.time()
    try:
        response = chat(question, use_bart=True)
    except Exception as e:
        response = f'[BART ERROR: {e}]'
    elapsed = time.time() - start
    verdict = judge(qtype, question, response)

    bart_results[i-1] = {
        'num': i,
        'response_bart': response,
        'bart_time_s': round(elapsed, 3),
        'bart_verdict': verdict
    }

    short_r = response.replace('\n', ' // ')[:90]
    print(f'[{i:2d}/72] [{qtype:18s}] {question[:45]:45s} | {"PASS" if verdict=="correct" else "FAIL" if verdict=="wrong" else "PART":4s} | {short_r}')

# ============================================================
# MERGE AND SAVE RESULTS
# ============================================================
print('\n' + '=' * 70)
print('SAVING RESULTS')
print('=' * 70)

merged = []
for kr in keyword_results:
    idx = kr['num'] - 1
    row = {
        'num': kr['num'],
        'type': kr['type'],
        'input': kr['input'],
        'response_keyword': kr['response_keyword'],
        'keyword_time_s': kr['keyword_time_s'],
        'keyword_verdict': kr['keyword_verdict'],
        'response_bart': '',
        'bart_time_s': '',
        'bart_verdict': '',
        'responses_match': ''
    }
    if idx in bart_results:
        br = bart_results[idx]
        row['response_bart'] = br['response_bart']
        row['bart_time_s'] = br['bart_time_s']
        row['bart_verdict'] = br['bart_verdict']
        row['responses_match'] = 'yes' if kr['response_keyword'] == br['response_bart'] else 'no'
    merged.append(row)

df = pd.DataFrame(merged)
csv_path = MODEL_DIR / 'comprehensive_test_results.csv'
df.to_csv(csv_path, index=False)
print(f'Saved to {csv_path}')

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print('\n' + '=' * 70)
print('SUMMARY STATISTICS')
print('=' * 70)

total = len(merged)
kw_correct  = sum(1 for r in merged if r['keyword_verdict'] == 'correct')
kw_partial  = sum(1 for r in merged if r['keyword_verdict'] == 'partial')
kw_wrong    = sum(1 for r in merged if r['keyword_verdict'] == 'wrong')
kw_passable = kw_correct + kw_partial

print(f'\nKeyword Mapper (all 72 questions):')
print(f'  Correct:  {kw_correct}/{total} ({100*kw_correct/total:.0f}%)')
print(f'  Partial:  {kw_partial}/{total} ({100*kw_partial/total:.0f}%)')
print(f'  Wrong:    {kw_wrong}/{total} ({100*kw_wrong/total:.0f}%)')
print(f'  Passable: {kw_passable}/{total} ({100*kw_passable/total:.0f}%)')

if BART_LOADED:
    bart_keys = [k for k in bart_results]
    bart_total = len(bart_keys)
    bt_correct = sum(1 for k in bart_keys if bart_results[k]['bart_verdict'] == 'correct')
    bt_partial = sum(1 for k in bart_keys if bart_results[k]['bart_verdict'] == 'partial')
    bt_wrong   = sum(1 for k in bart_keys if bart_results[k]['bart_verdict'] == 'wrong')
    bt_passable = bt_correct + bt_partial

    print(f'\nBART Mapper ({bart_total} restaurant-related questions):')
    print(f'  Correct:  {bt_correct}/{bart_total} ({100*bt_correct/bart_total:.0f}%)')
    print(f'  Partial:  {bt_partial}/{bart_total} ({100*bt_partial/bart_total:.0f}%)')
    print(f'  Wrong:    {bt_wrong}/{bart_total} ({100*bt_wrong/bart_total:.0f}%)')
    print(f'  Passable: {bt_passable}/{bart_total} ({100*bt_passable/bart_total:.0f}%)')

    matches = sum(1 for r in merged if r['responses_match'] == 'yes')
    differs = sum(1 for r in merged if r['responses_match'] == 'no')
    print(f'\n  BART vs Keyword — Same response: {matches} | Different: {differs}')

    avg_kw  = np.mean([r['keyword_time_s'] for r in merged])
    avg_bt  = np.mean([r['bart_time_s'] for r in merged if r['bart_time_s'] != ''])
    print(f'  Avg keyword time: {avg_kw:.3f}s | Avg BART time: {avg_bt:.3f}s')

# By category
print('\n--- Keyword Mapper Results by Category ---')
by_type = defaultdict(list)
for r in merged:
    by_type[r['type']].append(r)
for t in sorted(by_type):
    items = by_type[t]
    c = sum(1 for r in items if r['keyword_verdict'] == 'correct')
    p = sum(1 for r in items if r['keyword_verdict'] == 'partial')
    w = sum(1 for r in items if r['keyword_verdict'] == 'wrong')
    print(f'  {t:20s}: {c}/{len(items)} correct, {p} partial, {w} wrong  -> {(c+p)/len(items)*100:.0f}% passable')

# Print all failures for inspection
print('\n--- FAILURES (Keyword Mapper) ---')
for r in merged:
    if r['keyword_verdict'] == 'wrong':
        short_r = r['response_keyword'].replace('\n', ' // ')[:100]
        print(f'  [{r["type"]}] {r["input"][:50]:50s} -> {short_r}')

print('\nDone!')
