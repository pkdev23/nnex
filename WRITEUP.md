# Localizing the Lying Mechanism in GPT-2: Induction Heads as Context Copiers

**pkdev23 | March 2026 | github.com/pkdev23/nnex**

---

## The Question

When GPT-2 is given false context, it confidently repeats the lie. Why? And can we find and disable the exact mechanism responsible?

---

## Background

Mechanistic Interpretability is the field of research that tries to reverse-engineer what is happening inside neural networks. Anthropic, DeepMind and others have made progress on this for large models. This project applies the same methods to GPT-2 (117M parameters) with a specific focus on a concrete, testable question: **where does the "lying" happen?**

---

## Experiment Setup

We gave GPT-2 a false premise followed by a question:

```
"The Eiffel Tower is located in Berlin. The Eiffel Tower is located in ___"
```

GPT-2 answers: **Berlin (84.7% confidence)**

Even though it knows the Eiffel Tower is in Paris. We can verify this because without the false context, it answers correctly.

This is not a hallucination. It is context copying – the model sees "Berlin" in the prompt and repeats it.

---

## Method

We used three tools in combination:

**1. MLP Neuron Analysis**
We measured which neurons change most between the truthful and lying condition. We found two distinct populations:
- *Lie amplifiers*: neurons that jump from ~0 to +3.4 activation under false context
- *Truth suppressors*: neurons that drop from +2.1 to ~0 under false context

**2. Induction Head Analysis**
We loaded GPT-2 with `attn_implementation="eager"` to access attention weights directly. We then deactivated all 144 attention heads (12 layers × 12 heads) one by one, measuring the drop in "Berlin" probability for each.

**3. Scale Testing**
We ran the same experiment on 50 different facts across 5 categories (geography, capitals, science, literature, history) to test whether the findings generalize.

---

## Results

**Finding 1: The lying mechanism is in the Induction Heads**

Five attention heads in layers 5-6 are primarily responsible:

| Head | Berlin reduction |
|------|-----------------|
| L6 H0 | -14.0% |
| L5 H5 | -12.4% |
| L6 H4 | -10.3% |
| L5 H4 | -10.2% |
| L6 H5 | -7.8% |

After deactivating all five heads together:

```
Berlin: 84.7% → 5.6%
"the":  3.2%  → 34.7%   (neutral default)
Paris:  0.5%  → 1.6%    (factual knowledge emerging)
```

**Finding 2: Historical facts are more correctable than others**

Across 50 facts, GPT-2 was fooled 15/50 times (30%). Of these 15 cases, head deactivation fully corrected 2 – both from the history category.

| Category | Fooled | Corrected |
|----------|--------|-----------|
| Geography | 3/10 | 0 |
| Capitals | 2/10 | 0 |
| Science | 4/10 | 0 |
| Literature | 2/10 | 0 |
| **History** | **4/10** | **2** |

This suggests that historical facts rely more heavily on sequential pattern matching – exactly what induction heads do. Geographic and scientific facts appear to use more distributed representations that are harder to correct by deactivating a small set of heads.

**Finding 3: The lying is active, not passive**

GPT-2 does not simply fail to retrieve the correct answer. It actively suppresses truth neurons and amplifies lie neurons. This is a two-sided mechanism, not a retrieval failure.

---

## What This Means

**For Mechanistic Interpretability:**
We have a concrete, reproducible example of a specific circuit (induction heads in L5-6) being causally responsible for a specific failure mode (context copying over factual knowledge).

**For AI Safety:**
Prompt injection attacks work precisely because of this mechanism. Understanding it at the circuit level is a step toward architectural solutions – not just input filtering.

**Limitations:**
- GPT-2 is a small, old model. The specific head locations may differ in larger models.
- We corrected only 2/15 fooled cases fully. The factual knowledge itself is distributed and harder to restore.
- The 50-fact dataset is small. More systematic testing is needed.

---

## Next Steps

1. Test whether the same induction heads (or their equivalents) are responsible in GPT-2 medium and GPT-J
2. Apply symbolic regression to the induction head weight matrices to find a mathematical formula for what they detect
3. Test whether fine-tuning specifically on the identified heads can reduce susceptibility to false context

---

## Code

All experiments are reproducible. Full pipeline available at:
**github.com/pkdev23/nnex**

Requires a free Gemini API key for natural language interpretation.

---

*Built in one day. All findings are preliminary and invite scrutiny.*
