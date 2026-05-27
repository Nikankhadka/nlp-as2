# Understanding the Restaurant Review Chatbot

### What is ABSA?

A normal sentiment tool says a review is "mixed." ABSA (Aspect-Based Sentiment Analysis) gives you per-aspect scores: "Food: C-, Service: A" instead of just "Overall: B."

Example: *"The pasta was cold but the waiter was incredibly friendly."*
- **pasta** → Food / Negative
- **waiter** → Service / Positive

---

### How the Chatbot Works (Big Picture)

```
User types a message
        │
        ▼
  detect_intent()          ← chatbot/intents.py
  "What does user want?"
        │
   ┌────┼────┬──────┬──────────┬──────────┬──────────┐
   ▼    ▼    ▼      ▼          ▼          ▼          ▼
 greet fare help  domain    review     tech      off_domain
       well      _query   _analysis  _questions
                   │          │          │
                   ▼          ▼          ▼
                stats    ABSA runs   pre-written
                lookup   on review   answers
                            │
                    ┌───────┴───────┐
                    │ 1. Extract    │
                    │    aspects    │
                    │ 2. Map to     │
                    │    category   │
                    │ 3. Classify   │
                    │    sentiment  │
                    │ 4. Format     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  LLM WRAPPER │  ← chatbot/llm.py
                    │  (optional)  │
                    └──────┬───────┘
                           │
                           ▼
                    Response to user
```

---

### File-by-File Guide

#### `run_chatbot.py` — Entry Point

Thin CLI launcher. All logic lives in the `chatbot/` package.

#### `chatbot/config.py` — Shared Setup

Loads environment variables, sets project paths, loads spaCy and NLTK.
Everything in one place so other modules don't repeat setup.
- Paths: `TRAIN_XML`, `TEST_XML`, `PROJECT_ROOT`
- API: `LLM_API_KEY`, `LLM_MODEL`, `LLM_ENABLED`
- NLP: `nlp` (spaCy), `STOPWORDS`, `LEMMATIZER`

#### `chatbot/data_utils.py` — Data Loading & Cleaning

Converts SemEval-2014 XML into clean DataFrames.
- `parse_restaurant_xml()` → creates 3 DataFrames: sentences, aspects, categories
- `clean_text()` → expands contractions, lowercases, removes URLs/HTML
- `normalize_text()` → collapses whitespace

**Why:** XML files are messy. The parser puts everything into neat tables so the rest of the code works with clean DataFrames.

#### `chatbot/absa.py` — The ABSA Engine

The core review analysis pipeline. Everything that touches a review goes through here.

| Function | What it does |
|----------|-------------|
| `extract_aspects_spacy()` | Finds aspects using spaCy POS + noun chunks + learned lexicon |
| `predict_category_fast()` | Maps terms to food/service/price/ambience using keywords |
| `make_feature()` | Wraps aspect in [ASPECT] tags so classifier knows the target word |
| `train_model()` | Trains TF-IDF + SMOTE + Logistic Regression on SemEval data |
| `format_absa_response()` | Picks one of 5 templates to present results |
| `analyse()` | Full pipeline: extract → map → classify → return results |

**Why keywords for categories:** 5 categories on restaurant vocabulary is a vocabulary problem, not a deep learning problem. Keywords are 340x faster than BART and 100% explainable.

**Why [ASPECT] tags:** Without them, "cold" is just a word. With them, the model learns "cold near food [ASPECT]" is different from "cold near drink [ASPECT]".

#### `chatbot/knowledge.py` — Domain Knowledge Base

Pre-computes statistics from all training annotations and answers customer questions.

| Function | What it does |
|----------|-------------|
| `compute_domain_knowledge()` | Counts positive/negative/neutral/conflict per category |
| `classify_domain_query()` | Figures out which category a question is about |
| `answer_domain_query()` | Answers "Is the food good?" with exact stats |
| `answer_overall_query()` | Gives bird's-eye view across all categories |
| `answer_complaints_query()` | Lists categories by negativity percentage |
| `answer_popular_query()` | Shows most-mentioned terms |
| `answer_restaurant_overview()` | Promotes ALL categories (food, service, ambience, value) |

**Why pre-compute:** When a user asks "Is the food good?", we look up exact counts. Food is 867 positive out of 1,232 total = exactly 70%. An LLM might hallucinate. Every percentage is traceable.

#### `chatbot/intents.py` — Intent Detection

Routes user input to the right handler before any processing happens.

| Function | What it does |
|----------|-------------|
| `detect_intent()` | 11-step priority-ordered decision tree |
| `detect_tech_question_intent()` | Checks if user asks about the bot itself |

**Priority order:** specific tech topics → help → remaining tech → domain queries → greetings → farewells → reviews → off-domain.

**Why priority matters:** "What are your limitations?" has no restaurant words — if we checked for reviews first, it would be rejected. Specific intents must be caught before general ones.

#### `chatbot/responses.py` — Response Templates

Pre-written responses for greetings, farewells, help, off-domain redirects, and technical questions.

| Function | What it does |
|----------|-------------|
| `general_responses()` | Returns templates for greeting/farewell/help/off_domain |
| `answer_tech_question()` | Returns pre-written answers about model, accuracy, training, etc. |

**Why pre-written for tech questions:** The facts don't change. Pre-writing gives polished, accurate answers every time — no LLM risk of hallucinating wrong accuracy numbers.

**Greeting improvement:** New templates start with "I'm doing well, how are you?" instead of jumping straight to capabilities. More natural.

#### `chatbot/memory.py` — Conversation Memory

Tracks the last 3 exchanges and the most recent topic discussed.

**Why:** Lets the bot handle follow-ups. If someone asks about food then says "What about the service?", the memory fills in that they're switching topics.

#### `chatbot/llm.py` — LLM Wrapper

Wraps Phase 1 deterministic responses in natural language via OpenRouter.

| Function | What it does |
|----------|-------------|
| `call_llm()` | Sends Phase 1 result + context to OpenRouter API |
| `_build_llm_system_prompt()` | Tells the LLM what tone to use and what facts it can reference |
| `_apply_guard_rails()` | Blocks forbidden topics, counts off-domain attempts |

**Why wrapper not brain:** The LLM formats responses naturally, but ALL facts come from deterministic computations. If the API is down, `--no-llm` keeps the bot working with zero factual loss.

**Thinking indicator:** Prints "Thinking..." while the API call is in progress, so users know something is happening during the 1.5s delay.

#### `chatbot/app.py` — Main Application

Orchestrates everything: initialisation, chat routing, evaluation, and interactive mode.

| Function | What it does |
|----------|-------------|
| `init()` | Loads data, trains model, computes knowledge, returns ChatbotState |
| `chat()` | Full chat with LLM wrapping |
| `chat_keyword()` | Fast keyword-only chat (used by test harness) |
| `evaluate_test_set()` | Runs classifier on held-out test data |
| `interactive_chat()` | Terminal REPL loop |
| `main()` | CLI entry point with flags: `--test-only`, `--chat-only`, `--no-llm` |

**ChatbotState:** A dataclass that holds all trained objects (tfidf, clf, lexicons, knowledge, memory) and gets passed to every chat function. No global state.

---

### Training Pipeline (in detail)

```
Raw XML → parse_restaurant_xml() → 3,693 aspect annotations
    │
    ▼
clean_text() + make_feature() → [ASPECT]-tagged feature strings
    │
    ▼
TF-IDF Vectorizer → sparse numeric matrix (5,000 features)
    │
    ▼
SMOTE oversampling → balanced classes (conflict goes from 45 → ~2,200)
    │
    ▼
Logistic Regression → trained classifier
    │
    ▼
Saved as chatbot_model.pkl (for test harness)
```

**Why SMOTE:** Without it, the model would learn "always predict positive" and get 95% accuracy — but it would be useless for negative and conflict.

---

### Glossary

| Term | Simple definition |
|------|-------------------|
| **TF-IDF** | Scores words by how often they appear in THIS sentence vs ALL sentences. "Pizza" gets high score; "the" gets low score. |
| **SMOTE** | Creates synthetic training examples for rare classes by blending existing ones. |
| **Logistic Regression** | Assigns a weight to each word for each outcome. Sums weights and predicts the highest-scoring class. |
| **spaCy** | Library that reads text like a grammarian — knows nouns, verbs, adjectives, and how they relate. |
| **Lemmatisation** | Reducing words to base form: "waiters" → "waiter", "prices" → "price". |
| **Aspect** | The specific thing being reviewed — pasta, waiter, atmosphere. |
| **Polarity** | Sentiment direction: positive, negative, neutral, conflict. |
| **Intent** | What the user wants: review analysis, ask a question, say hello, learn about the bot. |
| **OpenRouter** | API service providing access to LLM models (we use `z-ai/glm-4.5-air`). |
| **F1 Score** | Balanced measure of classifier performance (harmonic mean of precision and recall). |

---

### Why These Design Choices

| Decision | Reason | What we avoided |
|----------|--------|-----------------|
| Keywords not BART | 340x faster, transparent, same accuracy | 1.6GB model, 340x latency, black-box decisions |
| Pre-computed stats | Every answer traceable, no hallucination | Unverifiable LLM facts |
| LLM as wrapper | Facts are deterministic; LLM handles only phrasing | Hallucinated statistics |
| Keyword category mapper | 5 categories on known vocabulary is a vocabulary problem | GPU-dependent neural classifier |
| 6-path intent system | Covers all examiner interaction types | "I am not sure" to everything else |
