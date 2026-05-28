# RestaurantXpert Chatbot — Architecture Guide

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [Project Structure](#2-project-structure)
3. [Startup Sequence](#3-startup-sequence)
4. [Request Flow: Input → Response](#4-request-flow-input--response)
5. [Intent Detection — Deep Dive](#5-intent-detection--deep-dive)
6. [ABSA Pipeline — Deep Dive](#6-absa-pipeline--deep-dive)
7. [Domain Knowledge Base](#7-domain-knowledge-base)
8. [LLM Wrapping Layer](#8-llm-wrapping-layer)
9. [Conversation Memory](#9-conversation-memory)
10. [Key Design Patterns](#10-key-design-patterns)

---

## 1. What It Does

A chatbot that lets you do two things:

1. **Analyze a restaurant review** — paste a sentence, the bot extracts what you mentioned (food, service, etc.) and tells you the sentiment per aspect
2. **Ask questions about a restaurant** — "Is the food good?" → the bot answers from pre-computed stats on 3,693 real guest annotations

The LLM is **only a rephraser** — all facts come from deterministic Phase 1 computations. If the API is down, `--no-llm` keeps the bot working with zero accuracy loss.

---

## 2. Project Structure

```
absa/
├── run_chatbot.py                     ← Entry point (thin wrapper)
├── chatbot_architecture.md            ← This file
├── PRESENTATION.md                    ← Viva presentation guide
├── test_chatbot_comprehensive.py      ← 72-question test harness
├── .env                               ← OPENROUTER_API_KEY
│
├── chatbot/                           ← Main Python package
│   ├── app.py                         ← Orchestrator: init(), chat(), main()
│   ├── config.py                      ← Paths, API config, spaCy/NLTK loading
│   ├── intents.py                     ← 11-step intent routing
│   ├── absa.py                        ← Aspect extraction, training, formatting
│   ├── knowledge.py                   ← Domain stats + query answers
│   ├── responses.py                   ← Templates (greeting, help, tech Qs)
│   ├── memory.py                      ← 3-turn conversation memory
│   ├── llm.py                         ← OpenRouter wrapper + guard rails
│   └── data_utils.py                  ← XML parsing, text cleaning
│
├── data/
│   └── raw/
│       ├── Restaurants_Train_v2.xml   ← SemEval-2014 training (3,041 sentences)
│       └── Restaurants_Test_Gold.xml  ← SemEval-2014 test set
│
└── outputs/
    ├── chatbot_model.pkl              ← Serialized TF-IDF + LR + lexicons
    └── test_results_*.csv             ← Test harness outputs
```

### File Route Reference

| What | File |
|------|------|
| Entry point | `run_chatbot.py` |
| Initialization (+ training + knowledge build) | `chatbot/app.py:53` — `init()` |
| Main chat (Phase 1 + LLM wrapping) | `chatbot/app.py:160` — `chat()` |
| Keyword-only chat (no LLM, used by tests) | `chatbot/app.py:103` — `chat_keyword()` |
| Interactive REPL loop | `chatbot/app.py:412` — `interactive_chat()` |
| CLI argument parsing | `chatbot/app.py:470` — `main()` |
| Intent detection (11-step decision tree) | `chatbot/intents.py:116` — `detect_intent()` |
| Aspect extraction (spaCy + lexicon) | `chatbot/absa.py:50` — `extract_aspects_spacy()` |
| Category mapping (keyword classifier) | `chatbot/absa.py:91` — `predict_category_fast()` |
| Feature engineering (`[ASPECT]` tagging) | `chatbot/absa.py:129` — `make_feature()` |
| Sentiment training (TF-IDF + SMOTE + LR) | `chatbot/absa.py:146` — `train_model()` |
| Review analysis (extract → map → classify) | `chatbot/absa.py:264` — `analyse()` |
| ABSA response formatting (5 templates) | `chatbot/absa.py:176` — `format_absa_response()` |
| Knowledge base computation | `chatbot/knowledge.py:87` — `compute_domain_knowledge()` |
| Domain query classification | `chatbot/knowledge.py:53` — `classify_domain_query()` |
| Category query answering | `chatbot/knowledge.py:132` — `answer_domain_query()` |
| Restaurant overview (promotional) | `chatbot/knowledge.py:213` — `answer_restaurant_overview()` |
| LLM API call + guardrails | `chatbot/llm.py:70` — `call_llm()` |
| LLM system prompt builder | `chatbot/llm.py:17` — `_build_llm_system_prompt()` |
| Guardrail (forbidden topics, off-domain limit) | `chatbot/llm.py:124` — `_apply_guard_rails()` |
| Template responses (greeting, help, etc.) | `chatbot/responses.py:13` — `general_responses()` |
| Tech question answers | `chatbot/responses.py:112` — `answer_tech_question()` |
| Conversation memory (deque) | `chatbot/memory.py:8` — `ConversationMemory` |
| XML parsing into DataFrames | `chatbot/data_utils.py:43` — `parse_restaurant_xml()` |
| Text cleaning (contractions, lowercase) | `chatbot/data_utils.py:27` — `clean_text()` |
| Configuration (paths, env, spaCy/NLTK) | `chatbot/config.py` |

---

## 3. Startup Sequence

When you run `python3 run_chatbot.py`, here's what happens:

```
run_chatbot.py
  └─ chatbot/app.py:main()
       ├─ init()                           ← chatbot/app.py:53
       │   ├─ parse_restaurant_xml()       ← Loads SemEval-2014 XML → DataFrames
       │   ├─ build_extraction_lexicon()   ← Builds single_lex, multi_lex, head_lex
       │   │                                  from aspect terms appearing ≥2x
       │   ├─ train_model()                ← TF-IDF + SMOTE + Logistic Regression
       │   │                                  on 3,693 aspect annotations
       │   ├─ compute_domain_knowledge()   ← Aggregates all annotations →
       │   │                                  {food: {total, positive_pct, ...}, ...}
       │   └─ lemmatize_tokens()           ← Lemmatizes 120+ restaurant terms
       │
       ├─ evaluate_test_set()              ← Prints accuracy (70.99%), F1, etc.
       ├─ run_fifty_question_test()        ← 46 keyword-only test questions
       └─ interactive_chat()               ← The REPL loop
```

**What `init()` returns:** A `ChatbotState` dataclass (`chatbot/app.py:35`) bundling:
- `tfidf`, `clf` — the trained sentiment model
- `single_lex`, `multi_lex`, `head_lex` — extraction lexicons
- `domain_knowledge` — pre-computed stats dict
- `memory` — `ConversationMemory()` instance
- `restaurant_terms_lem` — lemmatized restaurant terms
- `no_llm` — boolean flag

This single object is passed to every chat function — no global state.

---

## 4. Request Flow: Input → Response

Here's how one user message is processed:

```
                         USER TYPES: "The pasta was cold but the waiter was friendly"

    ┌────────────────────────────────────────────────────────────────────────────┐
    │  STEP 1: INTENT DETECTION    chatbot/intents.py:116                        │
    │                                                                             │
    │  Runs 11 checks in priority order, stops at first match:                    │
    │                                                                             │
    │  [1] Tech topics (specific) — no match                                     │
    │  [2] Help patterns — no match                                              │
    │  [3] Tech topics (generic) — no match                                      │
    │  [4] Domain query — no match                                               │
    │  [5] Complaint/popular — no match                                          │
    │  [6] Restaurant about-us — no match                                        │
    │  [7] Greeting — no match                                                   │
    │  [8] Farewell — no match                                                   │
    │  [9] Review tokens — lemmatized {"pasta","cold","waiter","friendly"}       │
    │      intersect restaurant_terms_lem → MATCH → 'restaurant_review'          │
    └────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
    ┌────────────────────────────────────────────────────────────────────────────┐
    │  STEP 2: PHASE 1 — DETERMINISTIC PROCESSING   chatbot/app.py:178-182       │
    │                                                                             │
    │  Intent is 'restaurant_review', so call:                                    │
    │                                                                             │
    │  analyse()   chatbot/absa.py:264                                           │
    │  ├─ extract_aspects_spacy()   chatbot/absa.py:50                            │
    │  │   ├─ spaCy POS: finds nouns "pasta", "waiter"                           │
    │  │   ├─ spaCy chunks: catches multi-word phrases                            │
    │  │   └─ Lexicon: supplements domain terms spaCy might miss                  │
    │  │   → ["pasta", "waiter"]                                                 │
    │  │                                                                          │
    │  ├─ For each aspect:                                                        │
    │  │   ├─ predict_category_fast("pasta")  → "food"  (keyword match)          │
    │  │   ├─ make_feature("the pasta was cold...", "pasta")                      │
    │  │   │   → "the [ASPECT] pasta [/ASPECT] was cold..."                      │
    │  │   ├─ tfidf.transform() → numeric vector                                 │
    │  │   ├─ clf.predict()     → "negative"                                     │
    │  │                                                                          │
    │  │   ├─ predict_category_fast("waiter") → "service"                        │
    │  │   ├─ make_feature("...waiter was friendly", "waiter")                    │
    │  │   └─ clf.predict() → "positive"                                         │
    │  │                                                                          │
    │  └─ Returns: [                                                            │
    │       {aspect:"pasta",  category:"food",   sentiment:"negative"},          │
    │       {aspect:"waiter", category:"service", sentiment:"positive"}           │
    │     ]                                                                       │
    │                                                                             │
    │  format_absa_response()   chatbot/absa.py:176                                │
    │  → randomly picks 1 of 5 formatting templates                               │
    │  → "I found these aspects: PASTA (food) negative, WAITER (service) positive"│
    └────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
    ┌────────────────────────────────────────────────────────────────────────────┐
    │  STEP 3: PHASE 2 — LLM WRAPPING (if enabled)    chatbot/llm.py:70          │
    │                                                                             │
    │  call_llm(user_input, context, phase1_response, intent, knowledge, no_llm)  │
    │  ├─ Builds system prompt with domain stats + tone rules                     │
    │  ├─ Injects conversation memory as context                                  │
    │  ├─ Sends Phase 1 result as user message                                   │
    │  ├─ POSTs to OpenRouter API (non-streaming, 15s timeout)                    │
    │  ├─ Applies guard rails (forbidden topics, off-domain limit)                │
    │  └─ Returns: "[LLM] Your review mentions two things: the pasta came out     │
    │              cold (so food was negative), but the waiter was friendly        │
    │              (service positive). A mixed experience overall!"                │
    └────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
                    Bot: [LLM] Your review mentions two things...
```

### Important: LLM is skipped for greetings/farewells

`chatbot/app.py:170-174` — Template responses for "hello" and "goodbye" are already natural and save ~1.5s of API latency.

### If LLM fails

`chatbot/llm.py:119-121` — Falls back to `_llm_fallback()` which returns the raw Phase 1 response (still factually accurate).

---

## 5. Intent Detection — Deep Dive

**File:** `chatbot/intents.py:116` — `detect_intent()`

### The priority order is critical:

```
1. Specific tech topics  ── "What model do you use?" → 'tech_questions'
                              "What are your limitations?" → 'tech_questions'
                              (checked FIRST so "limitations" catches before
                               generic "what are you" in step 2)

2. Help patterns        ── "Help" / "what can you do?" → 'help'
                              (checked AFTER specific tech to avoid false positives)

3. Generic tech topics  ── "How do you work?" / "compare A vs B" → 'tech_questions'

4. Domain query         ── "Is the food good?" → 'domain_query'
                              (two-stage filter: must have query phrase + restaurant term)

5. Complaint/popular    ── "What do people hate?" → 'domain_query'

6. About-us             ── "Tell me about your restaurant" → 'domain_query'

7. Greeting             ── "Hey there!" → 'greeting'
                              (uses token SET INTERSECTION, not substring —
                               prevents "yo" in "you" from false matching)

8. Farewell             ── "Goodbye" → 'farewell'

9. Review (lexicon)     ── "The pasta was cold" → 'restaurant_review'
                              (lemmatized tokens against 120+ RESTAURANT_TERMS —
                               "waiters" → "waiter", "prices" → "price")

10. Review (spaCy fallback) ── Catches terms not in our vocabulary
                              "The ramen was amazing" — has nouns + "was" pattern
                              + not a question → 'restaurant_review'

11. Off-domain          ── Everything else → 'off_domain'
```

### Why `lemmatize_tokens()` matters

Without lemmatization, "waiters" wouldn't match the lexicon entry "waiter," and a review that says "The waiters were rude" would be classified as `off_domain` instead of `restaurant_review`.

### Why the spaCy fallback exists

The 120-term lexicon was built from SemEval-2014 data. Terms like "ramen," "boba," or "brunch" might not be on the list. The spaCy fallback catches sentences with `NOUN + was/is/were + opinion` structure, as long as it's not structured like a question.

### Example: "What are your limitations?"

```
Step 1: tech_topic = detect_tech_question_intent()
         → matches "limitations" keyword
         → since topic IS in ('model','accuracy','training','limitations','sarcasm')
         → RETURN 'tech_questions' immediately
```

If this were checked AFTER help (step 2), "what are you" in the help patterns list would catch it first and misclassify it.

---

## 6. ABSA Pipeline — Deep Dive

**File:** `chatbot/absa.py`

The ABSA pipeline has 4 stages. Here's the walkthrough with the example:

> _"The pasta was cold and the waiters were rude but the cheesecake was amazing"_

### Stage A: Aspect Extraction (`extract_aspects_spacy`, line 50)

Three methods run in **sequence** and results are unioned:

| # | Method | What it finds | Example output |
|---|--------|---------------|----------------|
| 1 | **spaCy POS** | Nouns/proper nouns that are not stopwords, not opinion words, >2 chars | `pasta`, `waiters`, `cheesecake` |
| 2 | **spaCy chunks** | Multi-word noun phrases | (none in this example; would catch "wait staff") |
| 3 | **Lexicon lookup** | 2-3 token n-grams from `multi_lex`, single tokens from `single_lex` | supplements any terms spaCy missed |

**Result:** `["pasta", "waiters", "cheesecake"]`

The lexicon is built at startup from training data:
- `single_lex`: single-word terms appearing ≥2x
- `multi_lex`: 2-3 word phrases appearing ≥2x
- `head_lex`: head nouns appearing ≥3x

### Stage B: Category Mapping (`predict_category_fast`, line 91)

A **keyword-based deterministic classifier** — no model, no GPU, no disk:

| Aspect | Matched keyword | Category |
|--------|----------------|----------|
| `pasta` | "pasta" in food keywords | `food` |
| `waiters` | "waiters" in service keywords | `service` |
| `cheesecake` | no match → fallback | `miscellaneous` |

**Why keywords instead of BART/BERT?**
- 340x faster (no model load time, no inference)
- 100% transparent (every decision is a visible keyword match)
- Uses no GPU/disk
- Same accuracy for 5 well-defined restaurant categories

### Stage C: Feature Engineering (`make_feature`, line 129)

Wraps the aspect term in `[ASPECT]...[/ASPECT]` tags within the full review:

```
"the pasta was cold and the waiters were rude but the cheesecake was amazing"

→ For aspect "pasta":
  "the [ASPECT] pasta [/ASPECT] was cold and the waiters were rude but the cheesecake was amazing"

→ For aspect "waiters":
  "the pasta was cold and the [ASPECT] waiters [/ASPECT] were rude but the cheesecake was amazing"
```

**Why this matters:** Without tags, TF-IDF just sees bags of words. With tags, it learns that "cold" near a food `[ASPECT]` is different from "cold" near "atmosphere" `[ASPECT]`. The model learns context-specific patterns.

### Stage D: Sentiment Classification (`analyse`, line 264)

1. `tfidf.transform(feature)` — converts tagged text to a numeric vector
2. `clf.predict(vector)`` — Logistic Regression outputs one of: `positive`, `negative`, `neutral`, `conflict`

**Training happens at startup** (`train_model`, line 146):
- **TF-IDF**: `ngram_range=(1,2)`, `max_features=5000`, `sublinear_tf=True`
- **SMOTE**: Oversamples minority classes (conflict has only ~45 examples out of 3,693)
- **Logistic Regression**: `max_iter=1000`

### Response Formatting (`format_absa_response`, line 176)

Picks **randomly** from 5 templates for variety. Example outputs for the same review:

| Template | Output style |
|----------|-------------|
| v1 | Group by category, show dominant sentiment per category |
| v2 | Overall tone first, then breakdown |
| v3 | Per-aspect listing with emojis |
| v4 | Split into "things you liked / didn't like" |
| v5 | Summary with sentiment counts per category |

---

## 7. Domain Knowledge Base

**File:** `chatbot/knowledge.py`

### Computation (line 87)

At startup, `compute_domain_knowledge()` parses the SemEval-2014 XML and aggregates all 3,693 annotations:

```python
{
    'food': {
        'total': 1132,
        'positive': 792,   'positive_pct': 70,
        'negative': 196,   'negative_pct': 17,
        'neutral': 136,    'neutral_pct': 12,
        'conflict': 8,     'conflict_pct': 1
    },
    'service': { ... },
    'price': { ... },
    'ambience': { ... },
    '_overall_total': 3693,
    '_top_terms': [('food', 312), ('service', 289), ('prices', 145), ...]
}
```

Every percentage is traceable to an exact annotation count — zero hallucination.

### Answering Questions

| User asks | Function called | Example output |
|-----------|----------------|----------------|
| "Is the food good?" | `answer_domain_query('food', ...)` | "70% of 1,132 food mentions are positive. The standouts are sushi, pasta, steak." |
| "How's the overall vibe?" | `answer_overall_query(...)` | "About 65% of all guest feedback is positive. Top mentioned: food, service, prices." |
| "What do people complain about?" | `answer_complaints_query(...)` | Categories sorted by negative%: service 22%, price 18%, food 17%, ambience 12% |
| "What's popular?" | `answer_popular_query(...)` | Top 8 most-mentioned terms with counts |
| "Tell me about your restaurant" | `answer_restaurant_overview(...)` | Promotional pitch covering ALL categories like real staff |

All functions use `random.choice()` over 3-4 template variants for natural variety.

### Two-Stage Query Filter (`classify_domain_query`, line 53)

A user question must pass TWO checks to be routed as a domain query:
1. Contains a **query-intent phrase** (e.g., "how is", "what about", "tell me about") OR has `?` with domain keywords
2. Contains a **restaurant-related term** (e.g., "food", "service", "waiter")

This prevents "What is the weather?" from being treated as a restaurant question — it has a query phrase but no restaurant term.

---

## 8. LLM Wrapping Layer

**File:** `chatbot/llm.py`

### Architecture: LLM as wrapper, not brain

```
User input → Phase 1 (deterministic facts) → Phase 2 (LLM rephrases naturally) → Response
                 ↑                                 ↑
          100% factual, no guessing         "Can you say this more naturally?"
```

### API Call (`call_llm`, line 70)

- **Provider:** OpenRouter API
- **Model:** `z-ai/glm-4.5-air`
- **Method:** Non-streaming POST to `https://openrouter.ai/api/v1/chat/completions`
- **Parameters:** `max_tokens=300`, `temperature=0.7`, `timeout=15s`

**What's sent to the LLM:**
1. **System prompt** — tells the LLM what tone to use based on intent, embeds domain statistics, lists guard rules
2. **Context** — previous turn from `ConversationMemory` (injected as an assistant message)
3. **User message** — the original input + the Phase 1 structured result + instructions to rephrase

**What the system prompt tells the LLM to do for tone switching:**
```
intent = restaurant_review → act as analyzer (professional but conversational)
intent = domain_query       → act as restaurant staff (warm, "our guests", "people say")
intent = tech_questions     → explain naturally, no datasheet tone
intent = off_domain         → polite redirect
```

**Tone translation (what the LLM converts):**
```
Phase 1: "food: 70% positive, service: 54% positive, ambience: 61% positive"
  ↓ LLM rephrases ↓
Phase 2: "Our guests really love the food — 7 out of 10 mention it positively!"

Phase 1: "Training data: SemEval-2014, accuracy: 70.99%, F1: 0.715"
  ↓ LLM rephrases ↓
Phase 2: "I was trained on the SemEval-2014 restaurant review corpus and achieve about 71% accuracy."
```

### Guard Rails (`_apply_guard_rails`, line 124)

1. **Forbidden topics:** Blocks medical, legal, financial advice, self-harm, illegal content
2. **2-strike off-domain limit:** If a user asks off-topic twice, delivers a firm redirect ("I've mentioned this before — I'm a restaurant review analyst")
3. **No invented facts:** System prompt explicitly forbids making up restaurant names, reviews, or statistics

### Graceful Degradation

If the API is down or timed out → `_llm_fallback()` returns the raw Phase 1 response:

```
LLM ON:  [LLM] I found three aspects in your review...
LLM OFF: Here is what I found: FOOD: negative (pasta), SERVICE: positive (waiter)...
         ↑ still factually identical ↑
```

The `[LLM]` prefix in the output tells the user whether the LLM is active.

### "Thinking..." Indicator

In `interactive_chat()` (`chatbot/app.py:452-458`):
- Prints `"Bot: Thinking..."` before the API call
- Clears it when the response arrives
- Skipped for greetings/farewells (no API call needed)

---

## 9. Conversation Memory

**File:** `chatbot/memory.py:8` — `ConversationMemory`

### Structure

- **3-turn deque** (`deque(maxlen=3)`) storing recent exchanges
- Each entry: `{user, bot, intent, topic}`
- **`last_topic`** — tracks the most recent conversation topic for follow-ups

### How follow-up routing works

```
User: Is the food good?
  → detect_intent → 'domain_query' → answer_domain_query('food')
  → memory.last_topic = 'food'

User: What about the service?
  → detect_intent → 'domain_query'
  → text contains "what about" AND memory.last_topic IS set
  → try classify_domain_query on new text → no match
  → fall back to answer_domain_query('food')
```

### How context is used for LLM

```python
memory.get_context()
# → "User just asked: 'Is the food good?'. The intent was domain_query.
#    I responded about: food."
```

This string is injected as an assistant message in the LLM call so it remembers the conversation flow.

---

## 10. Key Design Patterns

### Pattern 1: Two-Phase Response Generation
All facts come from **Phase 1** (deterministic). The **Phase 2** LLM only rephrases naturally. Hallucination is structurally impossible — the LLM can only reformat, never invent.

### Pattern 2: LLM as Wrapper, Not Brain
The LLM never makes routing decisions, never classifies sentiment, never extracts aspects. It's a "stylistic layer" applied after all computation is done. If the API is down, `--no-llm` keeps the bot working with identical factual accuracy.

### Pattern 3: Priority-Ordered Intent Decision Tree
Specific patterns checked before general ones. This prevents "What are your limitations?" (which has "what are you" in it) from being caught by the generic help pattern. Token set intersection for greetings prevents substring false positives.

### Pattern 4: Pre-Computed Knowledge Base
All 3,693 annotations are aggregated once at startup. Every answer is a dictionary lookup with randomized template selection. No database queries, no model inference — just fast key lookup.

### Pattern 5: spaCy + Lexicon Hybrid Extraction
Aspect extraction combines NLP grammar (spaCy POS/chunks) with a domain-specific learned lexicon. Catches both general nouns and specialized restaurant vocabulary that spaCy might not know (tiramisu, sashimi, risotto).

### Pattern 6: Single Dataclass State
`ChatbotState` bundles all runtime objects into one dataclass passed to every function. No global variables, clean dependency injection.

### Pattern 7: Randomized Templates for Naturalness
Both ABSA responses (5 formats) and domain query answers (3-4 templates each) use `random.choice()` to avoid robotic repetition.

### Pattern 8: Graceful Degradation
If the LLM API fails, the chatbot falls back to deterministic Phase 1 responses with zero factual accuracy loss. The `[LLM]` prefix makes the active mode visible.

### Pattern 9: Same Data, Two Purposes
The SemEval-2014 training XML serves dual purposes:
1. **Training the sentiment classifier** (aspect terms → TF-IDF features → LR model)
2. **Populating the knowledge base** (aspect/category annotations → aggregated stats for Q&A)

---

## Quick Stats Reference

| What | Value |
|------|-------|
| Dataset | SemEval-2014 Task 4, 3,041 sentences, 3,693 aspect annotations |
| Sentiment model | TF-IDF + SMOTE + Logistic Regression |
| Accuracy | 70.99%, Weighted F1 0.715 |
| Category mapping | Keyword rule engine (340x faster than BART) |
| LLM | OpenRouter API, `z-ai/glm-4.5-air`, non-streaming |
| Extraction | spaCy POS + noun chunks + learned lexicon (hybrid) |
| Memory | 3-turn deque with topic tracking |
| Intent paths | 7 (greeting, farewell, help, restaurant_review, domain_query, tech_questions, off_domain) |
| Chatbot test pass rate | 100% (72/72 questions, 0 wrong) |
