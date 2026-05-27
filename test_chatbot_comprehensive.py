#!/usr/bin/env python3
"""Comprehensive Chatbot Test Harness — 72 questions across 12 categories.
Imports from the chatbot package (no duplicated code).
Runs keyword-only for deterministic testing."""

import sys, os, re, pickle, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from chatbot.config import PROJECT_ROOT
from chatbot.app import ChatbotState, chat_keyword, init
from chatbot.knowledge import compute_domain_knowledge
from chatbot.intents import lemmatize_tokens, RESTAURANT_TERMS

MODEL_DIR = PROJECT_ROOT / 'outputs'

print('=' * 70)
print('LOADING TRAINED MODEL')
print('=' * 70)

model_path = MODEL_DIR / 'chatbot_model.pkl'
if model_path.exists():
    with open(model_path, 'rb') as f:
        model_state = pickle.load(f)
    print(f'Model loaded from {model_path}')
else:
    print('No saved model found — training from scratch...')
    state = init(no_llm=True)
    model_state = {
        'tfidf': state.tfidf,
        'clf': state.clf,
        'single_lex': state.single_lex,
        'multi_lex': state.multi_lex,
        'head_lex': state.head_lex,
    }

TRAIN_XML = PROJECT_ROOT / 'data' / 'raw' / 'Restaurants_Train_v2.xml'
print('Computing domain knowledge...', flush=True)
DOMAIN_KNOWLEDGE = compute_domain_knowledge(TRAIN_XML)
print(f'Aggregated {DOMAIN_KNOWLEDGE["_overall_total"]} annotations.')

RESTAURANT_TERMS_LEM = lemmatize_tokens(RESTAURANT_TERMS)

state = ChatbotState(
    tfidf=model_state['tfidf'],
    clf=model_state['clf'],
    single_lex=model_state['single_lex'],
    multi_lex=model_state['multi_lex'],
    head_lex=model_state['head_lex'],
    domain_knowledge=DOMAIN_KNOWLEDGE,
    restaurant_terms_lem=RESTAURANT_TERMS_LEM,
    no_llm=True,
)

# ============================================================
# JUDGEMENT RULES
# ============================================================

def judge(qtype, question, response):
    r = response.lower()
    q = question.lower()

    if qtype in ('greeting', 'farewell'):
        if any(w in r for w in ['hello', 'restaurant', 'review', 'thanks',
                                 'goodbye', 'helpful', 'chat', 'dining',
                                 'doing well', 'how about you', 'assistant',
                                 'take care', 'glad', 'hope', 'come back',
                                 'enjoy', 'see you', 'happy dining']):
            return 'correct'
        return 'wrong'

    if qtype == 'help':
        if any(w in r for w in ['can help', 'analys', 'analysis', 'capabilities',
                                 'sentiment', 'reviews', 'aspect',
                                 'model', 'knowledge', 'trends',
                                 'here\'s what i do', 'ask me anything',
                                 'i can help with', 'i can tell you',
                                 'i\'m your restaurant']):
            return 'correct'
        if 'not sure' in r:
            return 'wrong'
        return 'partial'

    if qtype in ('restaurant_pos', 'restaurant_neg', 'restaurant_mixed'):
        is_absa_response = any(keyword in r for keyword in [
            'here is what i found', 'aspect', 'things you liked',
            "things you didn't like", 'overall, your review',
            'i analysed your review', 'here is a summary',
            'overall tone'
        ])
        if is_absa_response:
            if qtype == 'restaurant_pos' and ('positive' in r or 'things you liked' in r):
                return 'correct'
            if qtype == 'restaurant_neg' and ('negative' in r or "didn't like" in r or 'things you didn' in r):
                return 'correct'
            if qtype == 'restaurant_mixed':
                return 'correct'
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
                                 'restaurant', 'review', 'food', 'service']):
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

    if qtype == 'tech_questions':
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
# 72 TEST QUESTIONS
# ============================================================

test_questions = [
    ('greeting', 'Hello!'),
    ('greeting', 'Hi there'),
    ('greeting', 'Good morning'),
    ('greeting', 'Hey!'),
    ('greeting', 'Howdy'),

    ('farewell', 'Goodbye'),
    ('farewell', 'Bye'),
    ('farewell', 'Thanks for your help'),

    ('restaurant_pos', 'The pizza was amazing'),
    ('restaurant_pos', 'The service was excellent'),
    ('restaurant_pos', 'The desserts were delicious'),
    ('restaurant_pos', 'The waitress was very nice'),
    ('restaurant_pos', 'The ambience was wonderful'),
    ('restaurant_pos', 'Best pasta I have ever had'),

    ('restaurant_neg', 'The pasta was cold'),
    ('restaurant_neg', 'The waiter was rude to us'),
    ('restaurant_neg', 'The food was terrible and overpriced'),
    ('restaurant_neg', 'The music was too loud'),
    ('restaurant_neg', 'The service was very slow'),
    ('restaurant_neg', 'This place is dirty'),

    ('restaurant_mixed', 'The pasta was cold but the waiter was incredibly friendly and fast'),
    ('restaurant_mixed', 'Overpriced for the tiny portions, though the atmosphere was cozy'),
    ('restaurant_mixed', 'Great value and quick service, but the music was too loud to talk'),
    ('restaurant_mixed', 'The staff was rude and the waiting time was too long'),
    ('restaurant_mixed', 'Amazing desserts and the ambience was perfect for a date night'),
    ('restaurant_mixed', 'The risotto was overcooked and the sommelier was rude'),
    ('restaurant_mixed', 'Food was decent but the place was dirty'),
    ('restaurant_mixed', 'Excellent taste but small portions'),

    ('domain_query', 'Is the food here good?'),
    ('domain_query', 'What do people say about the service?'),
    ('domain_query', 'How is the ambience?'),
    ('domain_query', 'Tell me about the desserts'),
    ('domain_query', 'What about the prices?'),
    ('domain_query', 'Are the waiters friendly?'),
    ('domain_query', 'How is the wine selection?'),

    ('help', 'What can you do?'),
    ('help', 'Help'),
    ('help', 'What are your capabilities?'),
    ('help', 'How do you work?'),

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

    ('edge', ''),
    ('edge', 'Pizza'),
    ('edge', 'spagetti'),
    ('edge', 'A'),
    ('edge', '12345'),
    ('edge', '!!!!'),
    ('edge', 'Not bad'),
    ('edge', 'The'),

    ('tech_questions', 'What model are you using for sentiment analysis?'),
    ('tech_questions', 'How accurate are your predictions?'),
    ('tech_questions', 'What training data did you use?'),
    ('tech_questions', 'What are your limitations?'),
    ('tech_questions', 'How do you handle sarcasm?'),
    ('tech_questions', 'Compare the food at two different restaurants'),

    ('negation', 'Not bad at all'),
    ('negation', 'I would not say it was bad'),
    ('negation', 'The food was not great'),
    ('negation', 'Oh great another cold meal'),
    ('negation', "It's not the worst but it could be better"),

    ('long_review', 'We went for dinner last night and had the most amazing steak cooked perfectly medium rare with a side of truffle fries that were crispy and delicious'),
    ('long_review', 'The appetizers were cold when they arrived, the main course took forty five minutes to come out, and when we complained the manager was completely unhelpful and dismissive'),
    ('long_review', 'The cocktails were innovative and well-crafted, the sushi was fresh and beautifully presented, but the dessert menu was disappointing and the waiter seemed distracted all evening'),
    ('long_review', 'Arrived at seven, seated by eight, ordered by eight thirty, food arrived at nine fifteen, no apology, no discount, the pasta was lukewarm and the wine was served at room temperature on a hot summer night'),
]

# ============================================================
# RUN TESTS
# ============================================================

print('\n' + '=' * 70)
print('RUNNING 72 QUESTIONS WITH REFACTORED CHATBOT')
print('=' * 70)

results_list = []
for i, (qtype, question) in enumerate(test_questions, 1):
    start = time.time()
    try:
        response = chat_keyword(question, state)
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

after_path = MODEL_DIR / 'after_improvements.csv'
df.to_csv(after_path, index=False)
print(f'Saved to {after_path}')

# ============================================================
# SUMMARY
# ============================================================

print('\n' + '=' * 70)
print('SUMMARY STATISTICS')
print('=' * 70)

total = len(results_list)
correct = sum(1 for r in results_list if r['verdict'] == 'correct')
partial = sum(1 for r in results_list if r['verdict'] == 'partial')
wrong = sum(1 for r in results_list if r['verdict'] == 'wrong')
passable = correct + partial

print(f'\nRefactored System (all {total} questions):')
print(f'  Correct:  {correct}/{total} ({100 * correct / total:.0f}%)')
print(f'  Partial:  {partial}/{total} ({100 * partial / total:.0f}%)')
print(f'  Wrong:    {wrong}/{total} ({100 * wrong / total:.0f}%)')
print(f'  Passable: {passable}/{total} ({100 * passable / total:.0f}%)')

avg_time = np.mean([r['time_s'] for r in results_list])
print(f'  Avg response time: {avg_time:.3f}s (keyword-only)')

print('\n--- Results by Category ---')
by_type = defaultdict(list)
for r in results_list:
    by_type[r['type']].append(r)
for t in sorted(by_type):
    items = by_type[t]
    c = sum(1 for r in items if r['verdict'] == 'correct')
    p = sum(1 for r in items if r['verdict'] == 'partial')
    w = sum(1 for r in items if r['verdict'] == 'wrong')
    print(f'  {t:20s}: {c}/{len(items)} correct, {p} partial, {w} wrong  -> {(c + p) / len(items) * 100:.0f}% passable')

print('\n--- FAILURES ---')
for r in results_list:
    if r['verdict'] == 'wrong':
        short_r = r['response'].replace('\n', ' // ')[:120]
        print(f'  [{r["type"]}] {r["input"][:50]:50s} -> {short_r}')

print('\nDone!')
