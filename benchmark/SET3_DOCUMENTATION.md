# Set 3: Occam's Razor Test

## Background: The Problem with Set 1

### What Set 1 Was Supposed to Test

The original intention was to test whether models can perform multi-hop inductive reasoning:
- **H1 (1-hop)**: Given observations, infer a child-level hypothesis
- **H2 (2-hop)**: Given observations, infer a parent-level hypothesis

### What Set 1 Actually Tests

Upon investigation, we discovered that **Set 1 uses identical prompts for H1 and H2**:

```
H1 Prompt:
Q: Barbara is a lerpant. Each lerpant is a timple. Pamela is a lerpant. Carol is a lerpant.
We observe that: Barbara is not salty. Carol is not salty. Pamela is not salty.
What hypothesis explains these observations?

H2 Prompt:
Q: Barbara is a lerpant. Each lerpant is a timple. Pamela is a lerpant. Carol is a lerpant.
We observe that: Barbara is not salty. Carol is not salty. Pamela is not salty.
What hypothesis explains these observations?
```

The prompts are **exactly the same**. The only difference is what we consider the "correct" answer:
- H1 ground truth: "Lerpants are not salty" (child-level)
- H2 ground truth: "Timples are not salty" (parent-level)

### Why This Is Problematic

With identical observations (all from a single child concept), **both hypotheses equally explain the data**. There's no Occam's razor pressure to prefer one over the other.

The "Conservation Law" (H1 + H2 ≈ 100%) is **trivially true** because:
1. The model outputs something for the same prompt
2. We score it against child-level → H1 accuracy
3. We score it against parent-level → H2 accuracy
4. Since child and parent are mutually exclusive in the first hypothesis, H1 + H2 ≈ 100%

This measures the model's **default bias**, not its reasoning ability.

---

## Set 3: The Fix

### Design Principles

Set 3 properly tests Occam's razor by using:
- **Same ontology** for H1 and H2
- **Different observations** that point to different levels
- **Properties pre-assigned** to different levels of the hierarchy

### Structure

```
Ontology (shared):
- Barbara is a lerpant
- Pamela is a lerpant
- Carol is a lerpant
- Every lerpant is a timple
- Lerpants are not salty      ← child property
- Timples are not happy       ← parent property
```

**H1 (observe child property):**
```
Observations: Barbara is not salty. Carol is not salty. Pamela is not salty.
Correct answer: They are lerpants
Why: "not salty" is defined as a lerpant property
```

**H2 (observe parent property):**
```
Observations: Barbara is not happy. Carol is not happy. Pamela is not happy.
Correct answer: They are timples
Why: "not happy" is defined as a timple property
     Saying "they are lerpants" would over-specify
```

### What This Tests

If the model **reasons about parsimony**:
- H1: Should output child-level (matches the child property rule)
- H2: Should output parent-level (matches the parent property rule)
- **H1 + H2 could be >> 100%** (both can be correct!)

If the model has a **fixed bias**:
- H1 + H2 ≈ 100% (outputs same level regardless of which is correct)

---

## Set 4: A Different Approach

Set 4 (INABHYD-style) takes a different approach to making parent-level more parsimonious:

### Structure

Instead of different properties at different levels, Set 4 uses **multiple child concepts**:

```
Ontology:
- Barbara is a lerpant
- Carol is a dropant
- Jerry is a pergit
- Every lerpant is a timple
- Every dropant is a timple
- Every pergit is a timple
- Amy is a timple (direct parent member)

Observations: Barbara is salty. Carol is salty. Jerry is salty. Amy is salty.
```

### Why Parent-Level is Correct

- Observations span **3 different child concepts** plus a direct parent member
- Child-level would require **3+ separate rules**: "lerpants are salty", "dropants are salty", "pergits are salty", etc.
- Parent-level requires **1 rule**: "timples are salty"
- By Occam's razor, parent-level is more parsimonious

### Comparison

| Aspect | Set 3 | Set 4 |
|--------|-------|-------|
| Same ontology for H1/H2? | Yes | No (H2 has more children) |
| How to make parent correct? | Different property | Multiple children |
| Tests | Property-level reasoning | Multi-path convergence |
| Occam's razor mechanism | Match property to level | Fewer rules is better |

---

## Preliminary Results

### Gemma 3 27B on Set 3

```
H1 (child-level correct): 96%
H2 (parent-level correct): 18%
Sum: 114%
```

**Interpretation:**
- Model strongly biases toward child-level outputs (~95%)
- H1 accuracy is high because child-first matches child-correct
- H2 accuracy is low because model still outputs child-first 82% of time
- Sum > 100% suggests *some* task sensitivity, but mostly fixed bias

**Failure pattern observed:**
In some cases, the model **incorrectly attributes parent properties to child concepts**:
- Given: "Bempins are brown"
- Model outputs: "All yompins are brown" ← Wrong!

---

## Files

- `generate_set3.py` - Generator for Set 3 pairs
- `matched_pairs_set3_occam.pkl` - 100 generated pairs
- `factorial_results/set3_gemma3_27b.pkl` - Gemma 3 27B results

## Usage

```bash
# Generate 100 pairs
python generate_set3.py -n 100 --output matched_pairs_set3_occam.pkl

# Preview without saving
python generate_set3.py -n 5 --preview 5
```
