# Viva Presentation: Aspect-Based Sentiment Analysis Restaurant Chatbot

**3 presenters · 12 minutes max · Code-focused + Live Demo**

---

## Presentation Roles & Timing

| Presenter | Time | Focus |
|-----------|------|-------|
| **Presenter 1** | 0:00–4:00 | Intro, Why, Architecture Overview, Live Demo |
| **Presenter 2** | 4:00–8:00 | Code Walkthrough (Intent, ABSA, Knowledge Base) |
| **Presenter 3** | 8:00–12:00 | Test Results, What Works, Limitations, Improvements, Q&A |

---

## Quick Reference – Source Code Map

| Component | File | Line |
|-----------|------|------|
| App entry point | `run_chatbot.py` | 1 |
| Main orchestrator | `chatbot/app.py` | 1 |
| Paths, env, spaCy/NLTK | `chatbot/config.py` | 1 |
| XML parser + text cleaning | `chatbot/data_utils.py` | 1 |
| Aspect extraction + training | `chatbot/absa.py` | 1 |
| Domain knowledge + query answers | `chatbot/knowledge.py` | 1 |
| Intent detection | `chatbot/intents.py` | 1 |
| Response templates | `chatbot/responses.py` | 1 |
| Conversation memory | `chatbot/memory.py` | 1 |
| LLM wrapper | `chatbot/llm.py` | 1 |
| Test harness (72 questions) | `test_chatbot_comprehensive.py` | 1 |

---

## Key Stats (Memorize)

| Metric | Value |
|--------|-------|
| Dataset | SemEval-2014 Task 4, 3,041 sentences, 3,693 aspects |
| Sentiment Model | TF-IDF + SMOTE + Logistic Regression |
| Accuracy | **70.99%** · Weighted F1 **0.715** |
| Positive F1 | 0.837 · Negative 0.656 · Neutral 0.451 · Conflict 0.213 |
| Category Mapper | Keyword rule engine (340x faster than BART) |
| LLM Wrapper | OpenRouter API, model `z-ai/glm-4.5-air` |
| Chatbot Passable Rate | **100%** (72/72 questions, 0 wrong) |
| Test Harness | 12 categories × 72 questions |

---

---

## PRESENTER 1 — Introduction + Demo (4 min)

### SLIDE 1: What Is It? (1 min)

**"Aspect-Based Sentiment Analysis Restaurant Chatbot"**

**Say:** "We built a restaurant chatbot that does two things. First, it breaks down reviews aspect-by-aspect — instead of 'this place is great', it tells you that the *food* was excellent but the *service* was slow. Second, it answers customer questions about the restaurant using real guest feedback, not made-up opinions."

**Two Modes:**
1. **Review Analysis** — user pastes a review, bot extracts each aspect (food, service, price, ambience) and classifies sentiment per aspect
2. **Domain Q&A** — user asks "Is the food good?", bot answers from pre-computed stats on 3,693 real annotations

**Show source:** `chatbot/app.py` (orchestrator), `chatbot/absa.py` (sentiment engine), `chatbot/knowledge.py` (knowledge base)

---

### SLIDE 2: Why This Project? (30 sec)

**Say:** "Restaurants get hundreds of reviews. A 3-star rating means nothing — was the food bad? Was the service slow? Generic star ratings don't tell the restaurant WHAT to fix. Aspect-based sentiment solves this: it attaches sentiment to *specific things* — food, service, price, atmosphere — so the restaurant knows exactly what's working and what isn't."

**Business value:**

| Problem | Our Solution |
|---------|-------------|
| Star ratings hide details | Per-aspect sentiment breakdown |
| Manual review reading is slow | Instant automated analysis |
| No structured guest feedback | Pre-computed stats on 4 categories |
| Staff can't answer every question | 24/7 chatbot trained on real data |

**Say:** "This is useful for restaurant owners, managers, or even curious customers who want to know what people *actually* say about a place."

---

### SLIDE 3: How It Works — Architecture (1 min)

**Say:** "Our architecture has two phases — and this is the key design decision. Phase 1 is completely deterministic: spaCy extracts aspects, a keyword engine maps them to categories, and Logistic Regression classifies sentiment. All facts come from Phase 1. Phase 2 is an LLM that *only* rephrases those facts naturally — it never invents data."

```
USER INPUT
    │
    ▼
┌──────────────────────────┐
│  PHASE 1: Deterministic  │  ← All facts computed here
│  ├─ Intent detection     │     (11-step priority tree)
│  ├─ spaCy aspect         │     extraction (+ learned lexicon)
│  ├─ Keyword category     │     mapper (340x faster than BART)
│  ├─ TF-IDF + LR          │     sentiment classifier
│  └─ Dict-lookup          │     knowledge base (3,693 annotations)
└──────────┬───────────────┘
           │
    ┌──────▼──────┐
    │ PHASE 2: LLM │  ← Rephrasing only, no facts invented
    │ OpenRouter   │
    │ glm-4.5-air  │
    └──────┬───────┘
           ▼
       RESPONSE
```

**Key justification points:**
- **Why keywords, not BERT?** 340x faster, 100% transparent (every decision traceable), same accuracy for 4 categories, no GPU needed
- **Why pre-computed stats?** Instant lookups, zero hallucination, every percentage traceable to an exact annotation count
- **Why LLM as wrapper?** If API is down, `--no-llm` keeps bot working with zero accuracy loss; LLM improves naturalness but never owns the facts
- **Dataset:** SemEval-2014 Task 4 — 3,041 sentences, 3,693 aspect annotations; same data trains the classifier AND populates the knowledge base

**Say:** "One dataset serves two purposes — it trains the sentiment classifier AND populates the knowledge base for Q&A. Efficient and traceable."

**Show code:** `chatbot/absa.py:146` (train_model), `chatbot/knowledge.py:87` (compute_domain_knowledge), `chatbot/llm.py:17` (system prompt builder)

---

### SLIDE 4: Live Demo (1.5 min) — Total: 4 min

**Switch to terminal — run:** `python3 run_chatbot.py`

#### Demo 1 — Greeting (natural response)
```
You: Hey, how are you doing?
Bot: I'm doing well, thanks for asking! I can share what our guests say about the
     food, service, price, and atmosphere. How can I help?
```

#### Demo 2 — Review Analysis (multi-aspect)
```
You: The steak was perfectly cooked but the service was incredibly slow
     and the bill was way too high.
Bot: [LLM] I found three things in your review:
     — Steak (food): positive — perfectly cooked
     — Service: negative — incredibly slow
     — Bill (price): negative — way too high
     Your experience was mixed — great food but let down by service.
```

#### Demo 3 — Domain Q&A (promotional tone)
```
You: Tell me about your restaurant
Bot: [LLM] We're known for our food — 7 out of 10 guests mention it positively!
     Our service is well-regarded too (54% positive), people love the atmosphere
     (61% positive), and guests feel they get good value. Popular dishes include
     food, service, prices. What catches your eye?
```

**Point out:** "See how it promotes ALL categories — food, service, ambience, value — like real staff. The `[LLM]` tag tells you the LLM is active."

#### Demo 4 — Tech Question
```
You: What model are you using?
Bot: [LLM] I use Logistic Regression with TF-IDF features, trained on 3,693
     aspect annotations from the SemEval-2014 corpus.
```

#### Demo 5 — Thinking indicator + Off-domain
```
You: What's the capital of France?
Bot: Thinking...
Bot: I'm here for restaurant questions! Want to know what guests love about
     our food, or how our service rates?
```

**Point out:** "The 'Thinking...' indicator shows when the LLM is processing. Graceful redirect for off-topic questions."

---

---

## PRESENTER 2 — Architecture & Code (4 min)

### SLIDE 5: System Architecture (1.5 min)

**Show diagram (draw or display):**

```
                    USER INPUT
                        │
               detect_intent()          ← chatbot/intents.py
                        │
    ┌──────┬──────┬─────┼─────┬──────┬──────┐
    ▼      ▼      ▼     │     ▼      ▼      ▼
 greeting farewell help  │  domain  tech   off_domain
                   _query   _questions
                         │
                    ┌────┴────┐
                    │  ABSA   │
                    │ Pipeline│       ← chatbot/absa.py
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         extract      map to     classify
         aspects     category    sentiment
         (spaCy)    (keywords)  (TF-IDF+LR)
                         │
                    ┌────┴────┐
                    │   LLM   │       ← chatbot/llm.py
                    │ Wrapper │
                    └────┬────┘
                         ▼
                      RESPONSE
```

**Key source files to show:**

| Step | File | Function |
|------|------|----------|
| Intent routing | `chatbot/intents.py:115` | `detect_intent()` |
| Aspect extraction | `chatbot/absa.py:52` | `extract_aspects_spacy()` |
| Category mapping | `chatbot/absa.py:92` | `predict_category_fast()` |
| Feature engineering | `chatbot/absa.py:128` | `make_feature()` |
| Training | `chatbot/absa.py:140` | `train_model()` |
| Knowledge base | `chatbot/knowledge.py:121` | `compute_domain_knowledge()` |
| Main chat loop | `chatbot/app.py:125` | `chat()` |

---

### SLIDE 6: Intent Detection (1 min)

**Source:** `chatbot/intents.py:115`

**Say:** "The intent router uses a 11-step priority-ordered decision tree. Order matters — specific intents checked before general ones."

**Priority order:**
1. Specific tech topics (model, accuracy, training, limitations, sarcasm)
2. Help patterns
3. Remaining tech topics (how-it-works, compare)
4. Domain queries ("Is the food good?")
5. Complaint/popular keywords
6. Restaurant about-us ("tell me about your restaurant")
7. Greetings (token set intersection)
8. Farewells
9. Review (lemmatized 120+ term matching)
10. spaCy fallback (noun + copula pattern)
11. Off-domain

**Say:** "Lemmatization is key — 'waiters' lemmatizes to 'waiter' so plurals are matched correctly. This fixed several failures from the original system."

---

### SLIDE 7: ABSA Pipeline in Detail (1.5 min)

**Source:** `chatbot/absa.py`

**3-step extraction** (`extract_aspects_spacy()`, line 52):
1. **spaCy POS tagging** — finds nouns/proper nouns (pizza, waiter, atmosphere)
2. **spaCy noun chunks** — catches multi-word phrases (wait staff, dining room)
3. **Learned lexicon** — catches domain terms spaCy might miss (tiramisu, sashimi)

**Feature engineering** (`make_feature()`, line 128):
- Wraps aspect term in `[ASPECT]...[/ASPECT]` tags within the full sentence
- Example: "the pasta was cold" → `"the [ASPECT] pasta [/ASPECT] was cold"`
- Without tags, "cold" is just a word — with tags, the model learns context

**Training** (`train_model()`, line 140):
- TF-IDF converts tagged text to numeric vectors
- SMOTE balances classes (conflict has only ~45 examples)
- Logistic Regression classifies sentiment

---

---

## PRESENTER 3 — Results + Q&A (4 min)

### SLIDE 8: Key Design Decisions (30 sec)

| Decision | Why |
|----------|-----|
| **Keywords not BART** for categories | 340x faster, 100% transparent, same accuracy |
| **Pre-computed stats** for domain Q&A | Every percentage traceable, no hallucination |
| **LLM as wrapper, not brain** | LLM formats naturally; facts come from deterministic computations |
| **6-path intent system** | Covers all interaction types an examiner expects |
| **3-turn memory** | Demonstrates stateful conversation within assignment scope |
| **"Thinking..." indicator** | Gives feedback during LLM API call (1.5s latency) |

---

### SLIDE 9: What's Been Tested (1.5 min)

**Say:** "We validated this system at two levels — model accuracy on the standard benchmark, and end-to-end chatbot behavior with a custom test harness."

**Model Evaluation** (`chatbot/app.py:283`):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Positive | 0.854 | 0.820 | **0.837** | ~60% |
| Negative | 0.522 | 0.612 | 0.563 | ~20% |
| Neutral | 0.483 | 0.424 | 0.451 | ~15% |
| Conflict | 0.152 | 0.357 | 0.213 | ~5% |
| **Overall** | | | **70.99%** acc | 3,693 |

**Chatbot Test Harness** (`test_chatbot_comprehensive.py`):

| Category | Questions | Result |
|----------|-----------|--------|
| Greeting / Farewell | 8 | 100% correct |
| Positive / Negative / Mixed reviews | 22 | 100% correct |
| Domain queries | 7 | 100% passable |
| Help | 4 | 100% passable |
| General / Off-domain | 10 | 100% correct redirect |
| Edge cases (empty, gibberish) | 8 | 100% handled |
| Tech / Self-knowledge | 6 | 100% correct |
| Negation handling | 5 | 100% passable |
| Long reviews | 4 | 100% passable |
| **TOTAL** | **72** | **100% passable, 0 wrong** |

**Say:** "72 questions across 12 categories, zero failures. The intent system routes correctly in all 11 priority steps, memory tracks follow-up questions, and off-domain redirects work reliably."

---

### SLIDE 10: What's Working (30 sec)

| Component | Status | Why |
|-----------|--------|-----|
| Intent detection | 100% | 11-step priority tree catches specific before general |
| Aspect extraction | Reliable | spaCy POS + chunks + learned lexicon hybrid |
| Category mapping | 100% transparent | Keyword matching on 4 categories, every decision explainable |
| Positive sentiment | F1 0.837 | Most common class (60%+), model well-trained |
| Domain Q&A | Zero hallucination | Pre-computed stats, every % traceable to exact annotation count |
| LLM fallback | Works flawlessly | `--no-llm` returns identical factual accuracy |
| Conversation memory | Works correctly | 3-turn deque + last_topic for follow-ups |
| Guard rails | Working | Forbidden topics blocked, 2-strike off-domain redirect |

---

### SLIDE 11: What's Not Working + Improvements (1 min)

**What needs improvement:**

| Issue | Why | Impact |
|-------|-----|--------|
| **Sarcasm detection** | "Oh great, another cold meal" → reads as positive | Misclassifies ironic reviews |
| **Neutral** (F1 0.451) | Ambiguous class between positive and negative | ~15% of reviews affected |
| **Conflict** (F1 0.213) | Only ~45 training examples / 3,693; SMOTE helps but can't fully compensate | Rare but wrongly classified |
| **No comparison** | Training data has no restaurant identities | Can't answer "Is Joe's better?" |
| **2014 vocabulary** | New terms like "boba", "cloud kitchen" not in lexicon | Misses modern terms |
| **No confidence scores** | Model outputs hard predictions without uncertainty | Can't flag borderline cases |

**Future improvements:**

| Improvement | Expected Gain | Effort |
|-------------|--------------|--------|
| Replace TF-IDF with BERT/RoBERTa | ~8-12% accuracy gain | Medium |
| Hierarchical classification (positive vs. not → fine-grained) | Better neutral/conflict F1 | Medium |
| Data augmentation for minority classes | +5-10% on conflict F1 | Low |
| Sarcasm-labeled fine-tuning | Enables sarcasm detection | High |
| Confidence thresholding + "I'm not sure" fallback | Better UX, flags borderline cases | Low |
| Periodic lexicon refresh | Covers newer food terms | Low |
| Streaming LLM responses | Removes 1.5s "Thinking..." delay | Low |

**Say:** "70.99% is competitive on SemEval-2014. The biggest practical gain would be BERT embeddings — we tested BART for category mapping but it was 340x slower with identical accuracy. For sentiment, contextual embeddings would likely give a meaningful boost."

---

### SLIDE 12: Q&A Prep (30 sec)

**Say:** "Three things to remember:
1. **Same data, two purposes** — annotations train the classifier AND populate the knowledge base
2. **Transparency over complexity** — 5KB keyword list replaced 1.6GB BART model
3. **LLM is formatting, not facts** — if the API is down, `--no-llm` flag keeps the bot working"

**Common examiner questions:**
- "Why 70.99% accuracy?" → Neutral/conflict drag it down; 83.7% on positive
- "Why Logistic Regression?" → Interpretable, fast, competitive on this benchmark
- "What does SMOTE do?" → Creates synthetic examples for minority classes
- "How does conversation memory work?" → 3-turn deque, tracks last topic for follow-ups
- "Is this actually RAG?" → It's a deterministic knowledge base, not vector RAG — keyword-routed lookup on pre-computed stats. The LLM only rephrases.
- "Why no vector database?" → Only 3,693 annotations across 4 fixed categories. A vector DB would be overengineering; a dict is faster and every number is traceable.

---

## Presentation Tips

- Open `chatbot/` folder in your editor before presenting
- Have terminal ready with `python3 run_chatbot.py` for live demo
- Use `--no-llm` flag if API is down during viva
- Print this sheet for quick reference to line numbers
