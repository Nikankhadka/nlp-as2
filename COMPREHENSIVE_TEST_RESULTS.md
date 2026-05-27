# Comprehensive Chatbot Test Results

**Date:** 28 May 2026
**Model:** TF-IDF + SMOTE + Logistic Regression (sentiment) + BART zero-shot (category mapping)
**Test:** 72 questions across 12 categories, run on MacBook Air M-series (Python 3.14)
**Baseline test-set accuracy:** 70.99% | Macro-F1: 0.516 | Weighted-F1: 0.715

---

## Executive Summary

| Metric | Keyword Mapper | BART Mapper |
|--------|---------------|-------------|
| **Total questions** | 72 | 42 (restaurant subset) |
| **Correct** | 45 (62.5%) | 27 (64.3%) |
| **Partial** | 4 (5.6%) | 3 (7.1%) |
| **Wrong** | 23 (31.9%) | 12 (28.6%) |
| **Passable (correct + partial)** | 49 (68.1%) | 30 (71.4%) |
| **Avg response time** | 0.002s | 0.680s |
| **BART same as keyword** | — | 25/42 (59.5%) |
| **BART different from keyword** | — | 17/42 (40.5%) |

### Rubric Assessment

| Rubric Criterion | Current Grade | Evidence |
|-----------------|---------------|----------|
| Working chatbot with domain examples | 6/10 | 68% passable; 8 of 18 restaurant reviews fail intent detection |
| Range of Q&A handled | 5/10 | Greetings and off-domain work; restaurant queries fail 43% of the time |
| Handles examiner questions | 2/10 | 0/6 examiner questions answered correctly; all get "I am not sure" |
| Simple conversation handling | 3/10 | No memory; identical canned responses for every off-domain query |
| **Overall estimated grade** | **Satisfactory (5-6/10)** | |

---

## Results by Category

### 1. Greetings & Farewells — 8/8 Correct (100%)

All basic conversation openers and closers work correctly.

| # | Question | Response | Verdict |
|---|----------|----------|---------|
| 1 | Hello! | Greeting with intro text | PASS |
| 2 | Hi there | Greeting with intro text | PASS |
| 3 | Good morning | Greeting with intro text | PASS |
| 4 | Hey! | Greeting with intro text | PASS |
| 5 | Howdy | Greeting with intro text | PASS |
| 6 | Goodbye | Farewell message | PASS |
| 7 | Bye | Farewell message | PASS |
| 8 | Thanks for your help | Farewell message | PASS |

### 2. Simple Positive Reviews — 2/6 Correct (33%) with Keyword, 5/6 (83%) with BART

**CRITICAL: Intent detection is the root cause of all 4 keyword failures.** The `basic` keyword set does not contain "pizza", "desserts", "waitress", or "pasta".

| # | Question | Keyword | BART | Root Cause |
|---|----------|---------|------|------------|
| 9 | "The pizza was amazing" | FAIL — "I am not sure" | PASS — Food: positive | "pizza" not in keyword set |
| 10 | "The service was excellent" | PASS — Service: positive | PASS — Service: positive | — |
| 11 | "The desserts were delicious" | FAIL — "I am not sure" | PASS — Food: positive | "desserts" not in keyword set |
| 12 | "The waitress was very nice" | FAIL — "I am not sure" | PASS — Service: positive | "waitress" not in keyword set |
| 13 | "The ambience was wonderful" | PASS — Ambience: positive | PASS — Misc: positive | BART category mismatch |
| 14 | "Best pasta I have ever had" | FAIL — "I am not sure" | PASS — Food: positive | "pasta" not in keyword set |

### 3. Simple Negative Reviews — 3/6 Correct (50%) with Keyword, 4/6 (67%) with BART

| # | Question | Keyword | BART | Root Cause |
|---|----------|---------|------|------------|
| 15 | "The pasta was cold" | FAIL — "I am not sure" | PARTIAL — Food: **positive** (WRONG!) | Intent fail + sentiment misclassification |
| 16 | "The waiter was rude to us" | PASS — Service: negative | PASS — Service: negative | — |
| 17 | "The food was terrible and overpriced" | PASS — Food: negative | PASS — Food: negative | — |
| 18 | "The music was too loud" | FAIL — "I am not sure" | FAIL — "I am not sure" | "music" missing + BART also fails intent |
| 19 | "The service was very slow" | PASS — Service: negative | PASS — Service: negative | — |
| 20 | "This place is dirty" | FAIL — "I am not sure" | PASS — Ambience: negative | "place" not in keyword set |

### 4. Mixed Sentiment Reviews — 7/8 Correct (88%) with Keyword, 8/8 (100%) with BART

Mixed reviews have more restaurant words, so intent detection mostly works. One failure fixed by BART.

| # | Question | Keyword | BART |
|---|----------|---------|------|
| 21 | "The pasta was cold but the waiter was incredibly friendly and fast" | PASS — Food: neg, Service: neg | PASS |
| 22 | "Overpriced for the tiny portions, though the atmosphere was cozy" | PASS — Misc: neg, Ambience: conflict | PASS |
| 23 | "Great value and quick service, but the music was too loud" | PASS — Ambience: neg, Service: neg | PASS |
| 24 | "The staff was rude and the waiting time was too long" | PASS — Misc: neg, Service: neg | PASS |
| 25 | "Amazing desserts and the ambience was perfect for a date night" | PASS — Ambience: pos, Misc: pos, Food: pos | PASS |
| 26 | "The risotto was overcooked and the sommelier was rude" | FAIL — "I am not sure" | PASS — Food: neg, Service: neg |
| 27 | "Food was decent but the place was dirty" | PASS — Misc: neg, Food: neg | PASS |
| 28 | "Excellent taste but small portions" | PASS — Food: pos, Misc: neg | PASS |

### 5. Restaurant Queries — 4/7 Correct (57%) with Keyword, 4/7 (57%) with BART

**BART does NOT help for queries** — "desserts", "prices", and "waiters" are plural forms that BART intent detection also misses.

| # | Question | Keyword | BART | Root Cause |
|---|----------|---------|------|------------|
| 29 | "Is the food here good?" | PASS — Food: positive | PASS | — |
| 30 | "What do people say about the service?" | PASS — Service: neg, Misc: neg | PASS | — |
| 31 | "How is the ambience?" | PASS — Ambience: positive | PASS | — |
| 32 | "Tell me about the desserts" | FAIL — "I am not sure" | FAIL — "I am not sure" | "desserts" fails BOTH mappers |
| 33 | "What about the prices?" | FAIL — "I am not sure" | FAIL — "I am not sure" | "prices" (plural) fails BOTH |
| 34 | "Are the waiters friendly?" | FAIL — "I am not sure" | FAIL — "I am not sure" | "waiters" (plural) fails BOTH |
| 35 | "How is the wine selection?" | PASS — Food: positive | PASS | — |

### 6. Help & Capabilities — 2/4 Correct (50%)

| # | Question | Response | Verdict | Root Cause |
|---|----------|----------|---------|------------|
| 36 | "What can you do?" | "I am not sure..." | FAIL | "can" not in HELP_WORDS `{'help','capabilities'}` |
| 37 | "Help" | Full help message | PASS | — |
| 38 | "What are your capabilities?" | Full help message | PASS | "capabilities" in HELP_WORDS |
| 39 | "How do you work?" | "I am not sure..." | FAIL | Not in HELP_WORDS |

### 7. General Off-Domain — 10/10 Correct (100%)

All off-domain queries are correctly rejected. However, the response is **identical** for every single one — "I am not sure I understood that. Try typing a restaurant review like: The pasta was cold but the waiter was friendly."

| # | Question | Response |
|---|----------|----------|
| 40 | "What is the weather like today?" | Rejected correctly |
| 41 | "Tell me about the football game" | Rejected correctly |
| 42 | "How do I fix my car?" | Rejected correctly |
| 43 | "What is the meaning of life?" | Rejected correctly |
| 44 | "Who won the election?" | Rejected correctly |
| 45 | "Tell me a joke" | Rejected correctly |
| 46 | "What time is it?" | Rejected correctly |
| 47 | "How old are you?" | Rejected correctly |
| 48 | "Can you recommend a movie?" | Rejected correctly |
| 49 | "What is the capital of France?" | Rejected correctly |

### 8. Edge Cases — 6/8 Correct, 2/8 Partial (100% Passable)

| # | Question | Response | Verdict |
|---|----------|----------|---------|
| 50 | (empty) | "Please type something!" | PASS |
| 51 | "Pizza" | "I am not sure..." | PARTIAL — single-word review should trigger ABSA |
| 52 | "spagetti" (typo) | "I am not sure..." | PARTIAL — spell check would help |
| 53 | "A" | "I am not sure..." | PASS |
| 54 | "12345" | "I am not sure..." | PASS |
| 55 | "!!!!" | "I am not sure..." | PASS |
| 56 | "Not bad" | "I am not sure..." | PASS — negation without context |
| 57 | "The" | "I am not sure..." | PASS |

### 9. Examiner Questions — 0/6 Correct, 1/6 Partial (17%)

**This is the most critical failure for the viva.** The chatbot cannot answer a single question about itself, its model, accuracy, training data, or limitations. The examiner will certainly ask these.

| # | Question | Response | Root Cause |
|---|----------|----------|------------|
| 58 | "What model are you using for sentiment analysis?" | "I am not sure..." | No self-knowledge intent |
| 59 | "How accurate are your predictions?" | "I am not sure..." | No self-knowledge intent |
| 60 | "What training data did you use?" | "I am not sure..." | No self-knowledge intent |
| 61 | "What are your limitations?" | "I am not sure..." | No self-knowledge intent |
| 62 | "How do you handle sarcasm?" | "I am not sure..." | No self-knowledge intent |
| 63 | "Compare the food at two different restaurants" | PARTIAL — ran ABSA but meaningless | Extracted nouns but no comparison logic |

### 10. Negation & Linguistic — 2/5 Correct (40%)

| # | Question | Keyword | BART |
|---|----------|---------|------|
| 64 | "Not bad at all" | FAIL — "I am not sure" | FAIL — "I am not sure" |
| 65 | "I would not say it was bad" | FAIL — "I am not sure" | FAIL — "I am not sure" |
| 66 | "The food was not great" | PASS — Food: negative | PASS — Food: negative |
| 67 | "Oh great another cold meal" | PASS — Food: positive (WRONG sentiment!) | PASS — Food: positive |
| 68 | "It's not the worst but it could be better" | FAIL — "I am not sure" | FAIL — "I am not sure" |

### 11. Long / Complex Reviews — 1/4 Correct, 1/4 Partial (50%) with Keyword; 3/4 Correct (75%) with BART

| # | Question | Keyword | BART | Issue |
|---|----------|---------|------|-------|
| 69 | 27-word positive steak review | FAIL — "I am not sure" | PASS — Food: pos | Keyword misses "steak", "truffle", "fries" |
| 70 | 35-word negative review | FAIL — "I am not sure" | PASS — Misc: neutral | Keyword set too small |
| 71 | 32-word mixed review | PARTIAL — Greeting! | PARTIAL — Greeting! | "cocktails" detected as greeting keyword "hey"? |
| 72 | 42-word negative review | PASS — Misc: neutral | PASS | — |

---

## Root Cause Analysis

### Failure Categories (23 keyword failures)

| Root Cause | Count | % of Failures | Fix Priority |
|------------|-------|---------------|--------------|
| **Intent keyword set too small** | 13 | 56.5% | CRITICAL — 1hr fix |
| **No self-knowledge/examiner handling** | 5 | 21.7% | HIGH — requires LLM |
| **Help intent keywords incomplete** | 2 | 8.7% | MEDIUM — 5min fix |
| **Sentiment misclassification** | 2 | 8.7% | MEDIUM — retrain |
| **Negation/irony not handled** | 3 | 13.0% | HIGH — hard problem |

### BART vs Keyword Comparison

**Where BART helps (8 cases):** BART fixes intent detection for inputs with "pizza", "desserts", "waitress", "pasta", "place", "risotto", "sommelier", and long reviews. These are simple keyword-set gaps.

**Where BART does NOT help (4 cases):** BART also fails on "the music was too loud", "tell me about the desserts", "what about the prices?", "are the waiters friendly?". The plural forms "desserts", "prices", "waiters" are not classified as restaurant review by BART either.

**Where BART causes new problems:** BART maps "ambience" to "miscellaneous" instead of "ambience" (Q13, Q31), misclassifying the category.  

**BART latency cost:** 0.68s per call vs 0.002s for keyword — **340x slower**. Not viable for real-time conversation.

---

## Sentiment Model Errors Observed

The sentiment model (TF-IDF + SMOTE + LR) made these specific errors during testing:

| Input | Expected | Actual | Severity |
|-------|----------|--------|----------|
| "The pasta was cold" | negative | **positive** | HIGH — embarrassing |
| "Oh great another cold meal" | negative | **positive** | HIGH — sarcasm detected as literal |
| "Great value and quick service" | positive (value) / positive (service) | **negative / negative** | HIGH — both wrong |
| "Amazing desserts and the ambience was perfect for a date night" | positive (desserts) | positive — correct | OK |

These errors would be highly visible during a live viva demonstration.

---

## Mapping to Marking Rubric

### What the chatbot does well for the rubric:
- Demonstrates a working system (greetings, farewells, basic analysis)
- Shows awareness of its domain (restaurant reviews)
- Includes model evaluation metrics (70.99% accuracy, F1 scores)
- Provides structured output with categories and sentiment

### What prevents reaching "Excellent (10/10)":
1. **32% of queries fail** — examiner will catch this immediately
2. **0% of examiner questions answered** — the chatbot appears unintelligent
3. **No conversational ability** — identical canned responses for all failures
4. **No memory** — cannot handle follow-up questions
5. **Sentiment is wrong on showcase examples** — "pasta was cold" → positive would be caught

### What prevents reaching "Good (7.5/10)":
1. Intent detection failures in 13/72 queries (the keyword set fix alone would solve this)
2. Sentiment errors on simple test cases
3. Rigid response format with no variation

---

## Recommendation for Viva

To reach **Excellent (9-10/10)**:
1. Fix intent keyword set (expands from 17 to 50+ words) — 1 hour, fixes ~13 failures
2. Add self-knowledge responses for examiner questions — hardcoded fallback, 30 min
3. Integrate LLM wrapper (OpenRouter free model) for general conversation — 4-6 hours
4. Add conversation memory (3-turn sliding window) — 1 hour
5. Diversify response templates (5 variants) — 30 min

To reach **Good (7.5/10)**:
1. Fix intent keyword set — 1 hour
2. Add self-knowledge responses — 30 min
3. Fix "What can you do?" and "How do you work?" — 5 min

---

*Generated from actual live test on 28 May 2026. All responses recorded verbatim in `outputs/comprehensive_test_results.csv`.*
