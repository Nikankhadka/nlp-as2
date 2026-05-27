# Future Improvements — Knowledge-Grounded Expert Chatbot

**Updated:** 28 May 2026 — Based on live 72-question comprehensive test

---

## Test Results Summary (Actual Live Run)

| Mapper | Correct | Partial | Wrong | Passable |
|--------|---------|---------|-------|----------|
| Keyword | 45/72 (62.5%) | 4/72 (5.6%) | 23/72 (31.9%) | 49/72 (68.1%) |
| BART | 27/42 (64.3%) | 3/42 (7.1%) | 12/42 (28.6%) | 30/42 (71.4%) |

**Root cause analysis of 23 failures:** 13 = intent keyword set too small | 5 = no self-knowledge | 2 = help keywords | 3 = negation. BART addresses only 8 of 23 failures (and those 8 are all fixed by a bigger keyword set anyway). **BART is not needed.**

---

## Core Insight: Same Data, Two Purposes

The SemEval-2014 dataset (3,041 sentences, 3,693 aspect annotations, 3,713 category annotations) serves two distinct roles:

| Role | ABSA Model (Task 3) | Chatbot (Task 4) |
|------|--------------------|--------------------|
| **How data is used** | Each of 3,693 annotations trains a supervised classifier (token → sentiment) | All 3,693 annotations are aggregated into a domain knowledge base (distributions, frequencies) |
| **What it answers** | "What sentiment is 'cold pasta' in this review?" | "Is the food good at this restaurant?" |
| **Output** | `FOOD: negative (about: pasta)` | *"Based on 3,713 reviews, food is 70% positive. Pizza and sushi are most mentioned."* |
| **Model** | TF-IDF + SMOTE + Logistic Regression | Pre-computed statistics + intent router + conversational wrapper |

**Viva justification:** *"For my ABSA, I used the annotations to train a classifier. For the chatbot, I computed aggregate statistics from those same annotations to build a domain knowledge base. Same data, two systems, two different purposes."*

---

## Architecture: Current vs Target

### Current System (3 intent paths)

```
User → intent ─┬─ greeting/farewell/help → single template
                ├─ restaurant_query       → ABSA pipeline → single template
                └─ general                → "I am not sure"
```

**Problems:** 32% failure rate, 0/6 examiner questions answered, no domain Q&A, identical canned responses, no follow-up handling.

### Target System (6 intent paths)

```
User → intent ─┬─ greeting / farewell    → 4-5 varied templates
                ├─ restaurant_review      → ABSA pipeline → 4-5 varied templates
                ├─ domain_query           → stats lookup  → "Food is 70% positive..."
                ├─ examiner               → self-knowledge responses
                ├─ help / capabilities    → "I analyze restaurant reviews..."
                └─ off_domain             → domain redirect → "I specialize in..."
                                                              (optional LLM wraps all
                                                               responses naturally)
```

---

## Change Inventory: What Stays, Changes, Gets Added, Gets Removed

### Stays (Unchanged)

| Component | Why |
|-----------|-----|
| XML parser (`parse_restaurant_xml()`) | Still needed to compute aggregate domain stats |
| Text cleaning (`clean_text()`, `normalize_text()`) | Still needed for review analysis |
| Aspect extraction (`extract_aspects_spacy()`) | Core ABSA pipeline — works, stays as-is |
| Feature engineering (`make_feature()`) | Tags [ASPECT] span for the classifier |
| TF-IDF + LR sentiment model | Domain expertise engine — unchanged |
| Evaluation code (`evaluate_test_set()`) | Still needed for metrics |
| `chatbot_model.pkl` | Loads exactly the same |

### Modified

| Component | Change | Effort |
|-----------|--------|--------|
| `detect_intent()` | Expand keyword set 17→50+, add lemmatization, return 6 intent types | 30 min |
| `format_absa_response()` | 4-5 template variants instead of single hardcoded template | 20 min |
| `general_responses()` | Split into varied templates per intent type | 15 min |
| `chat()` | Route to 6 paths instead of 2 | 15 min |

### Added (New)

| Component | Purpose | Effort |
|-----------|---------|--------|
| `compute_domain_knowledge()` | Calculate aggregate stats once from 3,693 training annotations | 20 min |
| `classify_domain_query()` | Map natural questions → stat lookup | 30 min |
| Domain Q&A response templates | Structured answers from pre-computed stats | 15 min |
| `detect_examiner_intent()` | Keyword matching for self-knowledge questions | 15 min |
| Self-knowledge response dict | Pre-written facts about model, accuracy, data, limitations | 15 min |
| `ConversationMemory` | 3-turn sliding window for follow-ups | 20 min |
| LLM wrapper (Phase 2 only) | OpenRouter API + system prompt + guard rails | 4 hours |

### Removed

| Component | Why |
|-----------|-----|
| **BART zero-shot classifier** (1.6GB, `facebook/bart-large-mnli`) | 340x slower than keyword (0.68s vs 0.002s). Fixes only 8 of 23 failures — all 8 also fixed by expanded keyword set. Misclassifies "ambience"→"miscellaneous". Not needed. |
| `heads` parameter in `extract_aspects_spacy()` | Dead code — `head_lex` (203 words) is trained but never used in function body |

---

## Phase 1: Knowledge-Grounded Q&A (2 hours) — Grade: Good (7.5)

Makes the chatbot answer domain questions from its training data — not by classifying the question text, but by looking up pre-computed aggregate statistics.

### 1.1 Compute Domain Knowledge Base

```python
def compute_domain_knowledge(train_xml_path):
    """Aggregate all training annotations into a queryable knowledge base."""
    from collections import Counter, defaultdict

    root = ET.parse(train_xml_path).getroot()
    cat_polarity = defaultdict(list)
    term_counts = Counter()

    for s in root.findall('.//sentence'):
        for cat in (s.find('aspectCategories') or []):
            for c in (cat.findall('aspectCategory') if cat is not None else []):
                cat_polarity[c.get('category', '').lower()].append(
                    c.get('polarity', '').lower())
        for asp in (s.find('aspectTerms') or []):
            for a in (asp.findall('aspectTerm') if asp is not None else []):
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
        }

    knowledge['_overall_total'] = sum(len(p) for p in cat_polarity.values())
    knowledge['_top_terms'] = term_counts.most_common(20)

    return knowledge

# Example output after running on SemEval-2014 train data:
# {
#   'food':     {'total': 1232, 'positive': 867, 'negative': 209,
#                'positive_pct': 70, 'negative_pct': 17},
#   'service':  {'total': 597,  'positive': 324, 'negative': 218,
#                'positive_pct': 54, 'negative_pct': 37},
#   'price':    {'total': 321,  'positive': 179, 'negative': 115,
#                'positive_pct': 56, 'negative_pct': 36},
#   'ambience': {'total': 431,  'positive': 263, 'negative': 98,
#                'positive_pct': 61, 'negative_pct': 23},
#   '_top_terms': [('food',376), ('service',238), ('prices',65), ...]
# }
```

### 1.2 Domain Query Classifier

Maps natural language questions to knowledge base lookups:

```python
DOMAIN_QUERY_PATTERNS = {
    'food':     ['food', 'dish', 'meal', 'eat', 'pizza', 'pasta', 'sushi', 'taste'],
    'service':  ['service', 'staff', 'waiter', 'waitress', 'server', 'bartender'],
    'price':    ['price', 'cost', 'expensive', 'cheap', 'value', 'bill', 'money'],
    'ambience': ['ambience', 'atmosphere', 'decor', 'music', 'mood', 'vibe', 'lighting'],
    'complaints': ['complaint', 'worst', 'bad', 'disappointing', 'negative about'],
    'popular':  ['popular', 'most mentioned', 'common', 'frequent', 'discussed'],
}

def classify_domain_query(text, knowledge):
    """Determine which domain stat to return based on the question."""
    text_lower = text.lower()
    # Find which category the question is about
    for category, keywords in DOMAIN_QUERY_PATTERNS.items():
        if category in ('complaints', 'popular'):
            continue
        if any(k in text_lower for k in keywords):
            return categorie
    return None
```

### 1.3 Domain Q&A Response Templates

```python
def answer_domain_query(category, knowledge):
    """Generate response from pre-computed stats."""
    stats = knowledge.get(category)
    if not stats:
        return "I have data on food, service, price, and ambience. Which would you like to know about?"

    templates = [
        f"Based on {knowledge['_overall_total']} reviews in my training data, "
        f"{category} is rated positively {stats['positive_pct']}% of the time "
        f"and negatively {stats['negative_pct']}% of the time.",

        f"Across my training set, {category} gets positive marks "
        f"{stats['positive_pct']}% of the time ({stats['positive']} out of "
        f"{stats['total']} mentions) — with {stats['negative_pct']}% negative.",

        f"Looking at {stats['total']} {category} mentions in my data: "
        f"{stats['positive_pct']}% positive, {stats['negative_pct']}% negative, "
        f"{stats['neutral_pct']}% neutral.",
    ]
    import random
    return random.choice(templates)
```

### 1.4 Fix Intent Detection (Expanded Keywords + Lemmatization)

See full code in TEST_ERRORS.md Error #1. The fix expands the keyword set from 17 to 50+ words and adds WordNet lemmatization so "waiters" → "waiter", "prices" → "price", "desserts" → "dessert". This single fix recovers 13 of 23 failures.

### 1.5 Fix Help Intent

```python
HELP_PATTERNS = ['help', 'capabilities', 'what can you do', 'how do you work',
                 'what do you do', 'what are you']

def detect_intent(text):
    text_lower = text.lower()
    if any(p in text_lower for p in HELP_PATTERNS):
        return 'help'
```

### 1.6 Add Self-Knowledge Responses

```python
EXAMINER_KEYWORDS = {
    'model':      ['model', 'algorithm', 'classifier', 'logistic', 'tf-idf'],
    'accuracy':   ['accurate', 'accuracy', 'performance', 'f1', 'f1-score'],
    'training':   ['training', 'trained', 'data', 'dataset', 'semeval'],
    'limitations': ['limit', 'limitation', 'weakness', 'struggle', 'fail'],
    'sarcasm':    ['sarcasm', 'irony', 'sarcastic'],
}

EXAMINER_RESPONSES = {
    'model': ("I use a Logistic Regression classifier with TF-IDF features, "
              "trained on 3,693 aspect annotations from the SemEval-2014 dataset."),
    'accuracy': ("I achieve 70.99% accuracy on the test set — 84% F1 on positive, "
                 "45% F1 on neutral, 21% F1 on conflict."),
    'training': ("SemEval-2014 restaurant corpus: 3,041 review sentences with "
                 "manually annotated aspects, categories, and polarities."),
    'limitations': ("(1) I struggle with sarcasm, (2) I can't compare two restaurants, "
                    "(3) my vocabulary is from 2014 — newer food terms may be unfamiliar."),
    'sarcasm': ("Sarcasm is a known limitation. 'Oh great, another cold meal' "
                "can be misclassified as positive since I don't detect irony."),
}
```

### 1.7 What Phase 1 Conversation Looks Like (No LLM)

```
User: "Hello"
Chat:  "Hi! I'm a restaurant review analyst. What can I help you explore?"

User: "Is the food good here?"
Chat:  "Based on 3,713 reviews in my training data, food is rated positively 
       70% of the time and negatively only 17% of the time."

User: "What about the service?"
Chat:  [memory: previous topic was food]
       "Service is more polarizing — 54% positive but 37% negative. 
       Staff attitude and wait times are the biggest complaint areas."

User: "What model are you using?"
Chat:  "I use a Logistic Regression classifier with TF-IDF features, 
       trained on 3,693 aspect annotations from SemEval-2014."

User: "The pizza was amazing"
Chat:  [ABSA pipeline runs]
       "Glad you enjoyed it! Your review shows the pizza got a strong 
       positive mention — food quality seems to be the highlight here."

User: "What's the weather like?"
Chat:  "I specialize in restaurant review analysis — that's where I'm 
       most helpful. Would you like to analyze a dining experience?"
```

**Grade impact:** Goes from Satisfactory (5-6) to Good (7.5). Answers domain questions, handles examiner questions, varied responses, no more "I am not sure" for valid restaurant input. But responses are templated, not truly conversational.

---

## Phase 2: LLM Wrapper for Natural Conversation (4 hours) — Grade: Excellent (10)

Adds a lightweight LLM (OpenRouter free tier) that takes the pre-computed stats from Phase 1 and expresses them naturally — while keeping the ABSA model and domain knowledge base as the source of truth.

### Why Phase 2 → Excellent

| Factor | Phase 1 Only (Good 7.5) | Phase 1 + LLM (Excellent 10) |
|--------|------------------------|------------------------------|
| Domain Q&A | Templated "Based on 3,713 reviews..." | Natural: "Food tends to be well-received — about 70% positive across thousands of reviews" |
| Examiner questions | Pre-written scripted responses | Adapts to phrasing, explains reasoning, handles follow-up "but why?" |
| Off-domain handling | Rotates 4-5 templates | Contextual: "Weather's not my thing, but if you had a patio meal somewhere, I can analyze that!" |
| Follow-up engagement | Keyword-based "previous topic was food" | Full 3-turn memory, references earlier exchanges naturally |
| Unexpected questions | Fails if not in hardcoded patterns | LLM synthesizes from system prompt knowledge |

### 2.1 Key Design Principle

**The LLM never replaces your ABSA model.** It handles only conversation, formatting, and redirection. All facts come from your pre-computed statistics. The system prompt tells the LLM:

```
You format responses naturally. Facts come from the knowledge base, not from you.

DOMAIN KNOWLEDGE (use these exact numbers):
- Food: 70% positive, 17% negative, 7% neutral, 5% conflict (1,232 annotations)
- Service: 54% positive, 37% negative, 3% neutral, 6% conflict (597 annotations)
- Price: 56% positive, 36% negative, 3% neutral, 5% conflict (321 annotations)
- Ambience: 61% positive, 23% negative, 5% neutral, 11% conflict (431 annotations)
- Most mentioned: food (376), service (238), prices (65), staff (57), pizza (51)
- Overall dataset: 3,713 annotations, 58.7% positive, 22.6% negative

MODEL FACTS:
- Architecture: TF-IDF vectorizer + SMOTE oversampling + Logistic Regression
- Test accuracy: 70.99%, Weighted F1: 0.715
- Strongest class: positive (F1=0.837), Weakest: conflict (F1=0.213)
- Training data: SemEval-2014 Task 4, 3,041 sentences, 3,693 aspects

NEVER invent facts beyond these numbers. Never claim knowledge of specific restaurants.
```

### 2.2 LLM Integration — Critical Prompts Only

**System Prompt (condensed, full version in `LLM_INTEGRATION_PLAN.md`):**

```
You are RestaurantXpert, a restaurant review analysis chatbot.

You help users by: analyzing restaurant reviews for sentiment on FOOD, SERVICE,
PRICE, and AMBIENCE; answering questions about restaurant review trends using 
your training statistics; explaining how your ABSA system works.

FORMAT: 1-3 sentences. Conversational. Use the facts provided to you.

GUARD RAILS: Never give medical/legal/financial advice. Never make up restaurant 
facts. If a user goes off-domain 2x, politely redirect and stop engaging. Never 
engage with harmful content.

OFF-DOMAIN RESPONSE: "I specialize in restaurant reviews. [Redirect to domain]."
```

**Fallback strategy:** If OpenRouter API is unreachable during the viva, the Phase 1 deterministic responses take over immediately — no crash, no error message, just slightly less natural wording.

### 2.3 Intent Router (Full 6-Path Logic)

```python
def detect_intent(text):
    text_lower = text.lower()
    tokens = set(re.findall(r'[a-z]+', text_lower))

    # 1. Greetings/Farewells (unchanged)
    if tokens & GREETINGS: return 'greeting'
    if tokens & FAREWELL:  return 'farewell'

    # 2. Help (fixed — now catches "what can you do")
    if any(p in text_lower for p in HELP_PATTERNS):
        return 'help'

    # 3. Examiner questions (new)
    if detect_examiner_topic(text_lower):
        return 'examiner'

    # 4. Domain questions (new — "is food good?", "most complaints?")
    if classify_domain_query(text_lower):
        return 'domain_query'

    # 5. Restaurant reviews (expanded keywords + lemmatization)
    if has_restaurant_terms(text_lower, tokens):
        return 'restaurant_review'

    # 6. Off-domain
    return 'off_domain'
```

---

## What Phase 1+2 Conversation Looks Like

```
User: "Hello"
Chat:  "Hi! I'm RestaurantXpert — I analyze restaurant reviews and answer 
       questions about what people like and dislike in dining experiences. 
       Want to share a review?"

User: "The pizza was cold but the waiter was friendly"
Chat:  "Sounds like a mixed experience! The cold pizza was disappointing 
       [ABSA: FOOD negative], but the friendly waiter seems to have made up 
       for it [ABSA: SERVICE positive]. Service can really save a meal."

User: "What do people usually complain about?"
Chat:  "Across my training data of over 3,700 reviews, service is the biggest 
       pain point — 37% negative. Price follows at 36% negative. Food gets 
       complaints much less often, only 17% of the time."

User: "How accurate are you?"
Chat:  "I achieve about 71% accuracy on standard benchmarks. I'm strongest 
       at spotting positive sentiment at 84% F1, but I'll be honest — I 
       struggle with neutral reviews at 45% F1 and sarcasm still trips me up."

User: "What about the ambience?"
Chat:  [remembers previous topic was complaints]
       "Ambience gets fewer complaints — only 23% negative and 61% positive. 
       People seem to enjoy the atmosphere more than they dislike it, at 
       least in my training data."

User: "What's the capital of France?"
Chat:  "That's a bit outside my restaurant wheelhouse! I'd be much more useful 
       analyzing a French bistro experience if you have one to share."
```

---

## Why This Approach vs Alternatives

| Approach | Grade Ceiling | Risk in Viva | Justifiability | Effort |
|----------|--------------|-------------|----------------|--------|
| **Knowledge-grounded stats (Phase 1)** | Good (7.5) | LOW — deterministic, never wrong | STRONG — same data used for classification then aggregation | 2h |
| **+ LLM wrapper (Phase 2)** | Excellent (10) | LOW — LLM formats stats, doesn't invent facts | STRONG — LLM is conversational layer only; facts are pre-computed | +4h |
| Retrieval QA (embed→search→return) | Good (7.5) | MEDIUM — can retrieve irrelevant sentences | WEAKER — retrieval is essentially classification-by-similarity | 6-10h |
| Fine-tuned BERT QA | Good (7.5) | HIGH — hallucination risk on unseen questions | WEAKER — hard to justify as "different" from original classifier | 8-15h |

### Why Not Retrieval QA or Fine-Tuned BERT

1. **Reliability:** The viva is live. An embedding retrieving "The food was terrible" when asked "Is food good?" is embarrassing. A hallucinating BERT saying "Food is 93% positive" is worse. Computed stats are always correct.

2. **Justification:** An examiner can instantly reason: *"You trained a classifier on individual annotations, then aggregated those same annotations into distributions for Q&A."* Retrieval and BERT both retrain on the same data — conceptually closer to a second classifier than a genuinely different system.

3. **Separation of concerns:** With knowledge-grounded stats, the chatbot never makes up an answer. Every response is traceable to a specific count in the training data. That's examinable. That's viva-ready.

---

## Key Metrics: Before vs Phase 1 vs Phase 2

| Metric | Current | Phase 1 | Phase 2 |
|--------|---------|---------|---------|
| Passable rate (72 questions) | 68% | ~86% | ~95% |
| Restaurant intent detection | 57-88% by category | ~95% | ~95% |
| Domain Q&A ("Is food good?") | FAILS (classifies question as review) | Templated from stats | Natural from stats |
| Examiner questions handled | 0/6 | 5/6 (hardcoded) | 6/6 (natural) |
| Conversational variety | 1 template | 4-5 templates | Unlimited |
| Follow-up conversation | None | Keyword-based | 3-turn memory |
| Off-domain handling | "I am not sure" (identical) | 4-5 redirect templates | Contextual redirect |
| Response latency | 0.002s (keyword) | 0.002s | 0.002s + ~1.5s LLM |
| **Estimated rubric grade** | **Satisfactory (5-6)** | **Good (7.5)** | **Excellent (10)** |

---

## Implementation Roadmap

```
Phase 1: Knowledge-Grounded Q&A (2 hours)
├── compute_domain_knowledge()          → aggregate 3,693 annotations into stats
├── classify_domain_query()             → route "Is food good?" → FOOD stats
├── Expand keyword set 17→50+          → fix 13 intent failures
├── Add lemmatization                   → fix plural issues
├── Fix help intent                     → "What can you do?" works
├── Add examiner intent + responses     → 5 self-knowledge topics
└── Diversify response templates        → 4-5 variants per intent type

Phase 2: LLM Wrapper (4 hours)
├── Set up OpenRouter free account
├── Write system prompt with domain knowledge stats
├── Implement intent router (6 paths)
├── Add conversation memory (3 turns)
├── Add guard rails (pre/post LLM filters)
├── Test with 72-question suite
└── Fallback: if API down → Phase 1 responses

Polish (2 hours)
├── Rehearse viva demo (10 min)
├── Test with peers as examiners
├── Prepare backup if anything fails
└── Pre-warm models + verify paths
```

---

## Cost Breakdown

| Approach | Time | Cost | Grade | Risk |
|----------|------|------|-------|------|
| Current (no change) | 0h | $0 | Satisfactory (5-6) | HIGH — examiner will expose limitations |
| Phase 1 only | 2h | $0 | Good (7.5) | LOW — deterministic, all answers traceable |
| Phase 1 + 2 (recommended) | 6h | $0 (free tier) | Excellent (10) | LOW — LLM formats facts, doesn't invent them |

---

## Viva Presentation Outline

**10 minutes including questions**

### Part 1: Introduction (2 min)
- Domain: Restaurant review analysis
- Dataset: SemEval-2014 (3,041 sentences, 3,693 annotations, multi-restaurant)
- **Key framing:** *"Same dataset, two purposes — classification for ABSA, aggregation for the chatbot's domain knowledge"*
- Live: "Hello, I'm your restaurant review expert"

### Part 2: Approach & Scope (3 min)
- Show the 6-path intent router architecture
- Explain: *"When asked 'Is food good?', the chatbot doesn't classify the question — it looks up pre-computed statistics from 3,693 annotations"*
- Acknowledge limitations: sarcasm detection, neutral F1=0.45, conflict F1=0.21
- If Phase 2: *"An LLM wraps responses naturally — but all facts come from computed statistics"*

### Part 3: Testing — Live Demo (3 min)
Show RANGE — the examiner needs to see multiple types of interaction:
1. **Review analysis:** "The pizza was amazing" → ABSA + natural response
2. **Domain Q&A:** "Is the food good here?" → *"70% positive across 1,232 food mentions"*
3. **Examiner Q&A:** "How accurate are you?" → self-knowledge from stats
4. **Off-domain:** "What's the weather?" → graceful redirect
5. **Follow-up:** "What about service?" → memory + new stat
6. Show evaluation metrics (accuracy, F1, confusion matrix)

### Part 4: Conclusion (2 min)
- Future: Better neutral/conflict handling, BERT-based model, sarcasm detection
- The hybrid approach = domain accuracy (custom model) + conversational flexibility (LLM)
- Both systems grounded in the same SemEval data — two uses, one dataset

---

*See also: `COMPREHENSIVE_TEST_RESULTS.md` for full test log, `LLM_INTEGRATION_PLAN.md` for full prompts and integration code.*
