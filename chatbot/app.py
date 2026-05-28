"""Main application — chatbot orchestrator with training, evaluation,
interactive CLI, and 50-question test harness."""

import sys, pickle, time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import (
    PROJECT_ROOT, TRAIN_XML, TEST_XML, LLM_ENABLED, CATEGORIES, LEMMATIZER
)
from .data_utils import parse_restaurant_xml, clean_text
from .absa import (
    build_extraction_lexicon, extract_aspects_spacy, predict_category_fast,
    make_feature, train_model, analyse, format_absa_response
)
from .knowledge import (
    compute_domain_knowledge, classify_domain_query,
    answer_domain_query, answer_overall_query, answer_complaints_query,
    answer_popular_query, answer_restaurant_overview
)
from .intents import (
    detect_intent, detect_tech_question_intent, lemmatize_tokens,
    RESTAURANT_TERMS
)
from .responses import general_responses, answer_tech_question
from .memory import ConversationMemory
from .llm import call_llm


@dataclass
class ChatbotState:
    tfidf: object
    clf: object
    single_lex: set
    multi_lex: set
    head_lex: set
    domain_knowledge: dict
    memory: object = field(default_factory=ConversationMemory)
    restaurant_terms_lem: set = field(default_factory=set)
    no_llm: bool = False


def init(no_llm=False):
    print('Loading XML data...', flush=True)
    train_data = parse_restaurant_xml(TRAIN_XML, 'train')
    print(f'Loaded {len(train_data.aspects)} training aspect annotations', flush=True)

    print('Building extraction lexicon...', flush=True)
    single_lex, multi_lex, head_lex = build_extraction_lexicon(train_data.aspects)
    print(f'Single: {len(single_lex)} | Multi: {len(multi_lex)} | Head: {len(head_lex)}', flush=True)

    print('Preparing features...', flush=True)
    tfidf, clf = train_model(train_data.aspects)

    print('Computing domain knowledge from training annotations...', flush=True)
    domain_knowledge = compute_domain_knowledge(TRAIN_XML)
    print(f'Aggregated {domain_knowledge["_overall_total"]} annotations into domain knowledge base.', flush=True)

    restaurant_terms_lem = lemmatize_tokens(RESTAURANT_TERMS)

    model_path = PROJECT_ROOT / 'outputs' / 'chatbot_model.pkl'
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({
            'tfidf': tfidf,
            'clf': clf,
            'single_lex': single_lex,
            'multi_lex': multi_lex,
            'head_lex': head_lex,
        }, f)
    print(f'Model saved to {model_path}', flush=True)

    return ChatbotState(
        tfidf=tfidf,
        clf=clf,
        single_lex=single_lex,
        multi_lex=multi_lex,
        head_lex=head_lex,
        domain_knowledge=domain_knowledge,
        memory=ConversationMemory(),
        restaurant_terms_lem=restaurant_terms_lem,
        no_llm=no_llm,
    )


def chat_keyword(user_input, state):
    if not user_input.strip():
        return 'Please type something!'

    intent = detect_intent(user_input, state.domain_knowledge, state.restaurant_terms_lem)

    if intent == 'restaurant_review':
        absa_results = analyse(user_input, state.single_lex, state.multi_lex,
                               state.head_lex, state.tfidf, state.clf)
        return format_absa_response(absa_results)

    elif intent == 'domain_query':
        prev_topic = state.memory.get_previous_topic()
        text_lower = user_input.lower()

        from .knowledge import COMPLAINT_KEYWORDS, POPULAR_KEYWORDS

        if any(w in text_lower for w in COMPLAINT_KEYWORDS):
            return answer_complaints_query(state.domain_knowledge)
        elif any(w in text_lower for w in POPULAR_KEYWORDS):
            return answer_popular_query(state.domain_knowledge)
        elif any(w in text_lower for w in ['how is it', 'how are things', 'overall',
                                            'in general', 'what do you think']):
            return answer_overall_query(state.domain_knowledge)
        elif any(p in text_lower for p in [
            'i just want to know about your', 'want to know about your restaurant',
            'know about your restaurant', 'know about your place',
            'tell me about your restaurant', 'about your restaurant',
            'what kind of restaurant',
        ]):
            return answer_restaurant_overview(state.domain_knowledge)
        elif prev_topic and any(w in text_lower for w in ['what about', 'how about',
                                                            'and the', 'what of']):
            domain_cat = classify_domain_query(user_input, state.restaurant_terms_lem)
            if domain_cat:
                return answer_domain_query(domain_cat, state.domain_knowledge)
            return answer_domain_query(prev_topic, state.domain_knowledge)
        else:
            domain_cat = classify_domain_query(user_input, state.restaurant_terms_lem)
            if domain_cat:
                return answer_domain_query(domain_cat, state.domain_knowledge)
            return answer_overall_query(state.domain_knowledge)

    elif intent == 'tech_questions':
        tech_topic = detect_tech_question_intent(user_input)
        return answer_tech_question(tech_topic)

    else:
        return general_responses(intent)


def chat(user_input, state):
    if not user_input.strip():
        return 'Please type something!'

    intent = detect_intent(user_input, state.domain_knowledge, state.restaurant_terms_lem)
    context = state.memory.get_context()

    if intent in ('greeting', 'farewell'):
        response = general_responses(intent)
        if intent == 'greeting':
            state.memory.add_exchange(user_input, response, intent)
        return response

    should_use_llm = LLM_ENABLED and not state.no_llm

    if intent == 'restaurant_review':
        absa_results = analyse(user_input, state.single_lex, state.multi_lex,
                               state.head_lex, state.tfidf, state.clf)
        phase1_response = format_absa_response(absa_results)
        state.memory.add_exchange(user_input, phase1_response, intent, topic='restaurant_review')
        if should_use_llm:
            try:
                return call_llm(user_input, context, phase1_response, intent,
                                state.domain_knowledge, state.no_llm)
            except Exception:
                return phase1_response
        return phase1_response

    elif intent == 'domain_query':
        prev_topic = state.memory.get_previous_topic()
        text_lower = user_input.lower()

        from .knowledge import COMPLAINT_KEYWORDS, POPULAR_KEYWORDS

        if any(w in text_lower for w in COMPLAINT_KEYWORDS):
            phase1_response = answer_complaints_query(state.domain_knowledge)
            category = 'complaints'
        elif any(w in text_lower for w in POPULAR_KEYWORDS):
            phase1_response = answer_popular_query(state.domain_knowledge)
            category = 'popular'
        elif any(w in text_lower for w in ['how is it', 'how are things', 'overall',
                                            'in general', 'what do you think']):
            phase1_response = answer_overall_query(state.domain_knowledge)
            category = 'overall'
        elif any(p in text_lower for p in [
            'i just want to know about your', 'want to know about your restaurant',
            'know about your restaurant', 'know about your place',
            'tell me about your restaurant', 'about your restaurant',
            'what kind of restaurant',
        ]):
            phase1_response = answer_restaurant_overview(state.domain_knowledge)
            category = 'overall'
        elif prev_topic and any(w in text_lower for w in ['what about', 'how about',
                                                            'and the', 'what of']):
            domain_cat = classify_domain_query(user_input, state.restaurant_terms_lem)
            if domain_cat:
                phase1_response = answer_domain_query(domain_cat, state.domain_knowledge)
                category = domain_cat
            else:
                phase1_response = answer_domain_query(prev_topic, state.domain_knowledge)
                category = prev_topic
        else:
            domain_cat = classify_domain_query(user_input, state.restaurant_terms_lem)
            if domain_cat:
                phase1_response = answer_domain_query(domain_cat, state.domain_knowledge)
                category = domain_cat
            else:
                phase1_response = answer_overall_query(state.domain_knowledge)
                category = 'overall'

        state.memory.add_exchange(user_input, phase1_response, intent, topic=category)
        if should_use_llm:
            try:
                return call_llm(user_input, context, phase1_response, intent,
                                state.domain_knowledge, state.no_llm)
            except Exception:
                return phase1_response
        return phase1_response

    elif intent == 'tech_questions':
        tech_topic = detect_tech_question_intent(user_input)
        phase1_response = answer_tech_question(tech_topic)
        state.memory.add_exchange(user_input, phase1_response, intent, topic=tech_topic)
        if should_use_llm:
            try:
                return call_llm(user_input, context, phase1_response, intent,
                                state.domain_knowledge, state.no_llm)
            except Exception:
                return phase1_response
        return phase1_response

    elif intent == 'help':
        phase1_response = general_responses(intent)
        state.memory.add_exchange(user_input, phase1_response, intent)
        if should_use_llm:
            try:
                return call_llm(user_input, context, phase1_response, intent,
                                state.domain_knowledge, state.no_llm)
            except Exception:
                return phase1_response
        return phase1_response

    elif intent == 'off_domain':
        phase1_response = general_responses(intent)
        state.memory.add_exchange(user_input, phase1_response, intent)
        if should_use_llm:
            try:
                return call_llm(user_input, context, phase1_response, intent,
                                state.domain_knowledge, state.no_llm)
            except Exception:
                return phase1_response
        return phase1_response

    return general_responses('general')


def evaluate_test_set(state):
    print('\n' + '=' * 60)
    print('EVALUATING ON TEST SET')
    print('=' * 60)
    test_data = parse_restaurant_xml(TEST_XML, 'test')
    print(f'Loaded {len(test_data.aspects)} test aspect annotations', flush=True)

    from sklearn.metrics import accuracy_score, f1_score, classification_report

    predictions = []
    for _, row in test_data.aspects.iterrows():
        feat = make_feature(row['text'], row['term_normalized'])
        vec = state.tfidf.transform([feat])
        pred = state.clf.predict(vec)[0]
        predictions.append(pred)

    y_true = test_data.aspects['polarity'].values
    acc = accuracy_score(y_true, predictions)
    f1_weighted = f1_score(y_true, predictions, average='weighted')

    print(f'\nAccuracy:      {acc:.4f}')
    print(f'Weighted-F1:   {f1_weighted:.4f}')
    print(f'\nClassification Report:')
    print(classification_report(y_true, predictions, digits=4))

    return {'accuracy': acc, 'weighted_f1': f1_weighted, 'samples': len(y_true)}


def run_fifty_question_test(state):
    import pandas as pd

    print('\n' + '=' * 60)
    print('RUNNING 50-QUESTION TEST HARNESS')
    print('=' * 60)

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
            response = chat_keyword(question, state)
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

    type_counts = Counter(r['type'] for r in test_results)
    print(f'\nQuestion breakdown:')
    for t, c in sorted(type_counts.items()):
        print(f'  {t}: {c}')
    return test_results


def interactive_chat(state):
    banner = f"""
============================================================
  RESTAURANTXPERT — Interactive Chat Mode
  {'LLM: ' + ('ENABLED' if LLM_ENABLED and not state.no_llm else 'OFF') if hasattr(state, 'no_llm') else 'LLM: ' + ('ENABLED' if LLM_ENABLED else 'OFF')}
============================================================
  I can help with:
    \u2022 Review: "The pizza was amazing"
    \u2022 Domain Q: "Is the food good?"
    \u2022 Tech Q: "How accurate are you?"
    \u2022 Help: "What can you do?"
    \u2022 Stats: "What do people complain about?"
    \u2022 Trends: "What are the most mentioned terms?"
  Type 'exit', 'quit', or 'bye' to end the session.
============================================================
"""
    print(banner)

    msg_count = 0
    intent_counts = Counter()

    while True:
        try:
            user_input = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n')
            break

        if not user_input:
            continue

        if user_input.lower() in ('exit', 'quit', 'bye'):
            print('Bot: Goodbye! Thanks for chatting.\n')
            break

        msg_count += 1
        intent = detect_intent(user_input, state.domain_knowledge, state.restaurant_terms_lem)
        intent_counts[intent] += 1

        should_use_llm = LLM_ENABLED and not state.no_llm
        if should_use_llm and intent not in ('greeting', 'farewell'):
            print('Bot: Thinking...', end='\r', flush=True)

        response = chat(user_input, state)
        if should_use_llm and intent not in ('greeting', 'farewell'):
            print(' ' * 40, end='\r', flush=True)
        print(f'Bot: {response}\n')

    if msg_count > 0:
        print(f'--- Session Summary: {msg_count} messages ---')
        for intent, count in intent_counts.most_common():
            print(f'  {intent}: {count}')


def main():
    _chat_only = '--chat-only' in sys.argv
    _test_only = '--test-only' in sys.argv
    _no_llm = '--no-llm' in sys.argv
    _show_test = not _chat_only

    state = init(no_llm=_no_llm)

    eval_results = evaluate_test_set(state)

    if _show_test:
        run_fifty_question_test(state)

    print(f'\nModel save location: {PROJECT_ROOT / "outputs" / "chatbot_model.pkl"}')
    print(f'Domain knowledge: {state.domain_knowledge["_overall_total"]} annotations aggregated')

    if not _test_only:
        interactive_chat(state)

    print('\nDone!')


if __name__ == '__main__':
    main()
