"""Intent detection — priority-ordered router for 7 user intent categories."""

import re
from .config import TOKEN_RE, nlp, LEMMATIZER
from .data_utils import clean_text


GREETINGS = {'hi', 'hello', 'hey', 'howdy', 'greetings', 'morning', 'evening',
             'hiya', 'hola', 'yo', 'sup', 'good afternoon', 'good evening'}
FAREWELL = {'bye', 'goodbye', 'quit', 'thanks', 'thank', 'see you', 'later',
            'farewell', 'cya', 'cheers', 'take care', 'exit', 'done', 'stop'}
GREETING_TOKENS = {'hi', 'hello', 'hey', 'howdy', 'greetings', 'morning', 'evening',
                   'hiya', 'hola', 'yo', 'sup'}
FAREWELL_TOKENS = {'bye', 'goodbye', 'quit', 'thanks', 'thank', 'later', 'farewell',
                   'cya', 'cheers', 'exit', 'done', 'stop'}
GREETING_PHRASES = ['good afternoon', 'good evening']
FAREWELL_PHRASES = ['see you', 'take care']

HELP_PATTERNS = ['help', 'capabilities', 'what can you do', 'how do you work',
                 'what do you do', 'what are you', 'what can i ask', 'how can you',
                 'what is your purpose', 'what are your functions', 'commands',
                 'options', 'features', 'what questions can', 'what kind of',
                 'how does this work', 'explain yourself',
                 'what else', 'what other', 'what more', 'is there anything else',
                 'what else can']

RESTAURANT_TERMS = {
    'food', 'meal', 'dish', 'menu', 'taste', 'flavor', 'cuisine', 'ingredient', 'portion',
    'pizza', 'pasta', 'sushi', 'steak', 'burger', 'sandwich', 'salad', 'soup', 'appetizer',
    'dessert', 'wine', 'cocktail', 'drink', 'coffee', 'seafood', 'chicken', 'beef', 'pork',
    'lamb', 'rice', 'noodle', 'bread', 'cake', 'cheese', 'fries', 'sauce', 'tiramisu',
    'risotto', 'lobster', 'roll', 'sashimi', 'desserts', 'drinks', 'cocktails', 'noodles',
    'restaurant', 'cafe', 'bistro', 'diner', 'eatery', 'bar', 'pub', 'brunch',
    'reservation', 'table', 'booking', 'seating',
    'service', 'staff', 'waiter', 'waitress', 'server', 'bartender', 'host', 'manager',
    'sommelier', 'waiters', 'servers', 'waitstaff',
    'price', 'bill', 'cost', 'expensive', 'cheap', 'value', 'money', 'overpriced', 'deal',
    'prices', 'budget', 'affordable',
    'ambience', 'atmosphere', 'decor', 'music', 'lighting', 'mood', 'vibe', 'setting',
    'scene', 'interior', 'environment', 'design',
    'delicious', 'tasty', 'yummy', 'disgusting', 'bland', 'fresh', 'stale', 'cold',
    'warm', 'hot', 'crispy', 'tender', 'juicy', 'dry', 'burnt', 'overcooked', 'raw',
    'friendly', 'rude', 'polite', 'slow', 'fast', 'quick', 'attentive', 'helpful',
    'noisy', 'quiet', 'loud', 'crowded', 'cozy', 'romantic', 'dirty', 'clean'
}

RESTAURANT_ABOUT_US = [
    'your restaurant', 'your menu', 'your food', 'you serve',
    'do you serve', 'do you have', 'what meals do you',
    'what food do you', 'what kind of restaurant',
    'tell me about your', 'about your restaurant',
    'what is your best', 'what is your most popular',
    'what is the best', 'most popular dish', 'best food',
    'best thing on the menu', 'what do you recommend',
    'what should i order', 'whats your best', "what's your best",
    'what is good here', 'whats good here', "what's good here",
    'i just want to know about your', 'want to know about your restaurant',
    'know about your restaurant', 'know about your place',
]

TECH_QUESTION_KEYWORDS = {
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
                    'fail', 'fails', 'struggles', 'what can\'t',
                    'what are your limitations', 'what do you struggle with',
                    'what are your weaknesses'],
    'sarcasm':    ['sarcasm', 'irony', 'sarcastic', 'ironic', 'handle sarcasm',
                   'sarcasm detection', 'how do you handle irony'],
}


def lemmatize_tokens(tokens):
    return {LEMMATIZER.lemmatize(t) for t in tokens}


def detect_tech_question_intent(text):
    text_lower = text.lower()

    for topic, keywords in TECH_QUESTION_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return topic

    if any(w in text_lower for w in ['how do you work', 'how you work', 'explain yourself',
                                      'describe yourself', 'what are you', 'who are you',
                                      'tell me about yourself', 'how do you operate',
                                      'explain your process', 'what is your purpose']):
        return 'how_it_works'

    if any(w in text_lower for w in ['compare', 'comparing', 'difference between',
                                      'better restaurant', 'which restaurant',
                                      'two restaurants', 'vs']):
        return 'compare'

    return None


def detect_intent(text, domain_knowledge, restaurant_terms_lem):
    if not text or not text.strip():
        return 'off_domain'

    text_lower = text.lower()
    tokens = set(re.findall(r'[a-z]+', text_lower))
    tokens_lem = lemmatize_tokens(tokens)

    tech_topic = detect_tech_question_intent(text)

    if tech_topic in ('model', 'accuracy', 'training', 'limitations', 'sarcasm'):
        return 'tech_questions'

    if any(p in text_lower for p in HELP_PATTERNS):
        return 'help'

    if tech_topic:
        return 'tech_questions'

    from .knowledge import classify_domain_query
    domain_cat = classify_domain_query(text, restaurant_terms_lem)
    if domain_cat:
        return 'domain_query'

    from .knowledge import COMPLAINT_KEYWORDS, POPULAR_KEYWORDS
    if any(k in text_lower for k in COMPLAINT_KEYWORDS):
        return 'domain_query'
    if any(k in text_lower for k in POPULAR_KEYWORDS):
        return 'domain_query'
    if any(w in text_lower for w in ['how is it', 'how are things', 'overall',
                                      'in general', 'what do you think', 'opinion']):
        return 'domain_query'

    if any(p in text_lower for p in RESTAURANT_ABOUT_US):
        return 'domain_query'

    if tokens & GREETING_TOKENS or any(p in text_lower for p in GREETING_PHRASES):
        return 'greeting'

    if tokens & FAREWELL_TOKENS or any(p in text_lower for p in FAREWELL_PHRASES):
        return 'farewell'

    if tokens_lem & restaurant_terms_lem:
        return 'restaurant_review'

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
