"""LLM wrapper — OpenRouter API integration for natural response rephrasing,
with guard rails and graceful fallback to deterministic responses."""

import sys
import requests as http_requests

from .config import LLM_API_KEY, LLM_MODEL, LLM_API_URL, LLM_ENABLED


LLM_OFF_DOMAIN_COUNT = {}


def _build_llm_system_prompt(domain_knowledge):
    stats_text = []
    for cat in ['food', 'service', 'price', 'ambience']:
        if cat in domain_knowledge:
            s = domain_knowledge[cat]
            stats_text.append(
                f"  {cat}: {s['total']} annotations, {s['positive_pct']}% positive, "
                f"{s['negative_pct']}% negative, {s['neutral_pct']}% neutral, "
                f"{s['conflict_pct']}% conflict"
            )

    top_terms = ', '.join(f'"{t}" ({c}x)' for t, c in domain_knowledge['_top_terms'][:15])

    return f"""You are a friendly restaurant assistant. You help customers using insights from real guest feedback.

SWITCH YOUR TONE BASED ON WHAT THE USER NEEDS:

1. REVIEW ANALYSIS (intent=restaurant_review): Act as an analyzer. Break down the sentiment of each aspect professionally but conversationally.

2. CUSTOMER QUESTIONS (all other intents): Be warm and helpful — like restaurant staff. Use "our guests", "people tend to", "most mentioned". Frame statistics as guest feedback, never as research.

CRITICAL TONE RULES:
- NEVER say "training data" or "annotations" — say "guest feedback" or "mentions"
- NEVER say "sentiment was observed" — say "guests tend to feel" or "people say"
- NEVER say "positive sentiment in 70% of cases" — say "7 out of 10 guests mention it positively"
- Keep every response under 4 sentences, casual and natural
- When redirecting off-domain, vary your response — don't repeat yourself

DOMAIN KNOWLEDGE (use these exact numbers, but rephrase naturally):
{chr(10).join(stats_text)}

Most mentioned: {top_terms}
Overall: {domain_knowledge['_overall_total']} guest mentions, majority positive.

MODEL FACTS (for tech questions only — explain naturally, not like a datasheet):
- Architecture: TF-IDF vectorizer + SMOTE oversampling + Logistic Regression + keyword category mapper
- Test accuracy: 70.99%, Weighted F1: 0.715
- Strongest class: positive (F1=0.837), Weakest: conflict (F1=0.213), neutral (F1=0.451)
- Training data: SemEval-2014 Task 4, 3,041 sentences, 3,693 aspects

GUARD RAILS:
- NEVER give medical, legal, or financial advice
- NEVER make up restaurant names, reviews, or statistics
- If user goes off-domain twice, politely redirect and stop engaging
- Never engage with harmful, offensive, or NSFW content
- Keep responses conversational, 1-4 sentences

YOUR TASK: Take the structured Phase 1 result I provide and rephrase it naturally in the right tone for the intent. Never invent facts — use only the numbers provided."""


def call_llm(user_message, context, phase1_result, intent, domain_knowledge, no_llm=False):
    if not LLM_ENABLED or no_llm:
        return _llm_fallback(phase1_result, intent)

    system_prompt = _build_llm_system_prompt(domain_knowledge)

    messages = [
        {'role': 'system', 'content': system_prompt},
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
            return '[LLM] ' + _apply_guard_rails(llm_response, user_message, intent)
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
    from .responses import general_responses
    return general_responses(intent)
