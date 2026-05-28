"""Domain knowledge base — aggregates training annotations into stats and answers
restaurant queries from pre-computed category-level statistics."""

import random
from collections import Counter, defaultdict

from .config import CATEGORIES, LEMMATIZER


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

QUERY_INTENT_WORDS = [
    'how is', 'how are', 'what do people', 'what about', 'tell me about',
    'is the', 'are the', 'do people', 'how about', 'what is the',
    'what are the', 'what do you', 'what is your', 'what about',
    'any good', 'people like', 'people say', 'people think',
    'how would you', 'anything to', 'can you tell',
    'what meals', 'what food', 'what kind', 'do you serve',
    'do you have', 'would like to know', 'i want to know',
    'could you tell', 'what sort of', 'how good is',
    'what kind of', 'best thing', 'most popular'
]

COMPLAINT_KEYWORDS = [
    'complaint', 'worst', 'bad things', 'disappointing', 'negative about',
    'most hated', 'what do people hate', 'why do people complain',
    'common complaints', 'biggest problem', 'biggest problems'
]

POPULAR_KEYWORDS = [
    'popular', 'most mentioned', 'common', 'frequent', 'discussed',
    'most liked', 'most loved', 'best thing', 'best things',
    'what is popular', 'what are popular'
]


def classify_domain_query(text, restaurant_terms_lem):
    import re
    text_lower = text.lower()
    tokens = set(re.findall(r'[a-z]+', text_lower))
    tokens_lem = {LEMMATIZER.lemmatize(t) for t in tokens}

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


def compute_domain_knowledge(train_xml_path):
    import xml.etree.ElementTree as ET
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


def answer_domain_query(category, domain_knowledge):
    stats = domain_knowledge.get(category)
    if not stats:
        return ("I can tell you about our food, service, prices, or atmosphere — "
                "which would you like to know about?")

    cat_keywords = DOMAIN_QUERY_PATTERNS.get(category, [])
    top_for_cat = [(t, c) for t, c in domain_knowledge['_top_terms']
                   if any(k in t for k in cat_keywords)][:3]
    top_str = ', '.join(f'{t}' for t, _ in top_for_cat) if top_for_cat else 'various items'

    if stats['positive_pct'] > 50:
        verdict = f'guests really enjoy the {category}'
    elif stats['positive_pct'] > 40:
        verdict = f'{category} feedback is a bit mixed'
    else:
        verdict = f'{category} gets a fair amount of criticism'

    templates = [
        f"Our guests really enjoy the {category} — about {stats['positive_pct']}% "
        f"of {stats['total']} mentions are positive. "
        f"The most talked-about are {top_str}.",

        f"People tend to say good things about our {category}: "
        f"{stats['positive_pct']}% positive out of {stats['total']} mentions. "
        f"{top_str} come up a lot in guest feedback.",

        f"Looking at guest feedback, {verdict} — "
        f"{stats['positive_pct']}% positive vs {stats['negative_pct']}% negative. "
        f"The standouts are {top_str}.",

        f"{stats['positive_pct']}% of guests mention {category} positively. "
        f"{top_str} get the most attention — that's {stats['total']} mentions total.",
    ]
    return random.choice(templates)


def answer_overall_query(domain_knowledge):
    overall_pos = round(100 * sum(
        domain_knowledge[c]['positive'] for c in CATEGORIES
        if c in domain_knowledge) / domain_knowledge['_overall_total'])
    top_5 = ', '.join(t for t, _ in domain_knowledge['_top_terms'][:5])
    templates = [
        f"Overall, our guests are pretty happy — about {overall_pos}% of all feedback is positive. "
        f"The most talked-about things are {top_5}.",

        f"Across all guest mentions, roughly {overall_pos}% are positive. "
        f"{top_5} get mentioned the most.",

        f"Most guests leave happy — {overall_pos}% positive overall from "
        f"{domain_knowledge['_overall_total']} mentions. Top topics: {top_5}.",
    ]
    return random.choice(templates)


def answer_complaints_query(domain_knowledge):
    categories_by_neg = sorted(
        [(c, domain_knowledge[c]) for c in CATEGORIES if c in domain_knowledge],
        key=lambda x: x[1]['negative_pct'], reverse=True
    )
    lines = ["Here's what guests tend to flag:"]
    for cat, stats in categories_by_neg:
        lines.append(f"  {cat}: {stats['negative_pct']}% negative mentions "
                     f"({stats['negative']} out of {stats['total']})")
    return '\n'.join(lines)


def answer_popular_query(domain_knowledge):
    top = domain_knowledge['_top_terms'][:8]
    terms = ', '.join(f'{t}' for t, c in top)
    return (f"The most mentioned things by our guests: {terms}. "
            f"Food gets the most love — "
            f"{domain_knowledge['food']['positive_pct']}% positive across "
            f"{domain_knowledge['food']['total']} mentions.")


def answer_restaurant_overview(domain_knowledge):
    food_stats = domain_knowledge.get('food', {})
    service_stats = domain_knowledge.get('service', {})
    ambience_stats = domain_knowledge.get('ambience', {})
    price_stats = domain_knowledge.get('price', {})

    top_5 = ', '.join(t for t, _ in domain_knowledge['_top_terms'][:5])

    templates = [
        f"We're known for our food — {food_stats.get('positive_pct', 70)}% of guests mention it positively! "
        f"Our service is well-regarded too ({service_stats.get('positive_pct', 54)}% positive), "
        f"and people love the {ambience_stats.get('positive_pct', 61)}% positive ambience. "
        f"Popular dishes include {top_5}. What catches your eye?",

        f"Let me give you the full picture: our food gets rave reviews ({food_stats.get('positive_pct', 70)}% positive), "
        f"the service team is appreciated ({service_stats.get('positive_pct', 54)}% positive), "
        f"the vibe gets compliments ({ambience_stats.get('positive_pct', 61)}% positive), "
        f"and guests feel they get good value ({price_stats.get('positive_pct', 56)}% positive). "
        f"Top mentions include {top_5}. What would you like to know more about?",

        f"Here's what our guests love: the food (our strongest point at {food_stats.get('positive_pct', 70)}% positive), "
        f"the atmosphere ({ambience_stats.get('positive_pct', 61)}% positive), "
        f"friendly service ({service_stats.get('positive_pct', 54)}% positive), "
        f"and fair prices ({price_stats.get('positive_pct', 56)}% positive). "
        f"Most talked-about: {top_5}. Anything specific you'd like to dive into?",
    ]
    return random.choice(templates)
