"""Response templates — pre-written responses for greetings, farewells,
help requests, off-domain redirects, and tech/self-knowledge questions.
Why pre-written instead of pure LLM: templates are fast, reliable, and
greetings don't need LLM processing (saves ~1.5s latency)."""

import random


# ============================================================
# GREETING, FAREWELL, HELP, OFF-DOMAIN
# ============================================================

def general_responses(intent):
    if intent == 'greeting':
        templates = [
            "Hey! Doing well, how about you? I'm here to help with everything about our restaurant — what guests love about the food, how the service rates, or what dishes people mention most. What would you like to know?",

            "I'm doing well, thanks for asking! I can share what our guests say about the food, service, price, and atmosphere — or break down a dining review for you. How can I help?",

            "Hi there! I'm your restaurant assistant. Doing well — ready to help you with anything from popular dishes to guest feedback on our service and ambience. What are you curious about?",

            "Hello! I'm doing well, how are you? I can answer questions about our restaurant, share what guests love most, or analyze a review. Where should we start?",

            "Doing great, thanks! I'm here to help with anything restaurant-related — what's popular, how our service rates, or analyzing your dining experience. What would you like to know?",
        ]
        return random.choice(templates)

    if intent == 'farewell':
        templates = [
            "Thanks for stopping by! Hope that was helpful. Come back anytime.",
            "Glad I could help! Enjoy your meal — and when you have more questions, I'll be here.",
            "Take care! If you ever want to know more about our food or service, just ask. Bye!",
            "Goodbye! Hope the insights were useful. See you next time!",
            "Thanks for chatting! Happy dining — reach out anytime you need restaurant info.",
        ]
        return random.choice(templates)

    if intent == 'help':
        templates = [
            "I can help with:\n  \u2022 What guests love about our food, service, price, atmosphere\n  \u2022 Popular dishes and most-mentioned items\n  \u2022 Breaking down a review you have\n  \u2022 How my system works behind the scenes\n\nTry: 'What's your best food?' or 'The pasta was amazing'",
            "Here's what I do:\n  \u2022 Answer questions about our restaurant from guest feedback\n  \u2022 Analyse a dining review you share with me\n  \u2022 Explain my model and accuracy if you're curious\n\nTry asking 'Is the food good here?' or 'What do people complain about?'",
            "Ask me anything about our restaurant! I can tell you:\n  \u2022 What dishes guests mention most and how they rate them\n  \u2022 How people feel about the service, price, and atmosphere\n  \u2022 How my analysis system works\n\nTry: 'What's popular?' or 'How accurate are you?'",
            "I'm your restaurant knowledge assistant. I can:\n  1. Share what guests say about our food, service, prices, and vibe\n  2. Analyse a review — 'The steak was overcooked but the ambience was lovely'\n  3. Explain how I work — 'What model do you use?'\n\nWhat would you like to try?",
        ]
        return random.choice(templates)

    if intent == 'off_domain':
        templates = [
            "I'm here for restaurant questions! Want to know what guests love about our food, or how our service rates? That's where I can really help.",
            "That's a bit outside my area — but I can tell you everything about our restaurant. Popular dishes, guest feedback, service quality... what interests you?",
            "Happy to help with anything restaurant-related! Ask about our food, what people say about the service, or share a review. That's my sweet spot.",
            "I'm best at answering restaurant questions. Curious what guests tend to say about our food? Or want me to break down a dining experience?",
            "Sorry, I stick to restaurant topics! But I've got plenty to share — popular dishes, service ratings, guest feedback. What would you like to know?",
        ]
        return random.choice(templates)

    return ("I'm not sure I understood that. Try typing a restaurant review like:\n"
            '  "The pasta was cold but the waiter was friendly"')


# ============================================================
# TECH / SELF-KNOWLEDGE RESPONSES
# ============================================================

TECH_QUESTION_RESPONSES = {
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
        "3,693 training annotations."
    ),
    'compare': (
        "I cannot compare two specific restaurants because my training data doesn't include "
        "restaurant identities — only anonymized review sentences. I can tell you general "
        "trends (e.g., food is 70% positive, service is 54% positive), but not which "
        "restaurant is better."
    ),
}


def answer_tech_question(topic):
    if topic in TECH_QUESTION_RESPONSES:
        return TECH_QUESTION_RESPONSES[topic]
    return ("I'd be happy to tell you about my system! I use Logistic Regression with "
            "TF-IDF features for sentiment analysis, trained on the SemEval-2014 corpus. "
            "Ask me about my model, accuracy, training data, or limitations.")
