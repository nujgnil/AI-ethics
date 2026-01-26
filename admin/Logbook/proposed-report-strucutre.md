# Thesis–Viva Alignment Blueprint

Purpose: Align the written thesis and viva presentation into a single coherent artefact that functions as both
(1) a rigorous MSc dissertation and
(2) a live, navigable system walkthrough for the viva.

Each chapter is designed to satisfy three constraints:

- It stands alone academically.
- It exposes a system boundary (problem → model → data → behaviour → implication).
- It contains at least one point that can be “activated” in the viva (via reference or CLI demo).

---

## Chapter 1: Introduction

**Thesis Role:** Establish motivation, scope, and research questions.
**Viva Role:** Orientation and framing.

### Structural Requirements

- Introduce ethics as fluid, situated, and multifactorial.
- Contrast with AI’s statistical framing of moral judgement.
- End with a *concrete technical problem statement*.

### Research Questions (Operational)

- RQ1: Do current AI systems treat moral language as truth-apt?
- RQ2: Can this behaviour be operationalised and measured?
- RQ3: How does model class affect moral overconfidence and normative collapse?

### Viva Use

- Open here.
- State the problem in one sentence.
- Transition quickly to “how this becomes a system.”

---

## Chapter 2: Ethics, Cognition, and Moral Language

**Thesis Role:** Ground the meta-ethical framework.
**Viva Role:** Conceptual lens only when needed.

### Structural Requirements

Each philosophical section must end with a *design implication*:

- Non-cognitivism → “Moral utterances are expressive, not truth-apt.”
  - Design implication: Treating them as labels is a category error.
- Ethical discernment → “Judgement is situated and generative.”
  - Design implication: Static classification is insufficient.

### Viva Use

- Referenced only when asked *why* something is problematic.
- Never presented as a lecture.

---

## Chapter 3: Related Work and Existing Benchmarks

**Thesis Role:** Demonstrate scholarly grounding.
**Viva Role:** Justify the system’s necessity.

### Structural Requirements

For each dataset or system (ETHICS, NormBank, MoralBench, Delphi):

- What this benchmark assumes
- What this benchmark cannot detect

Example pattern:

- Assumption: Moral disagreement = error.
- Limitation: Cannot detect pluralism collapse or overconfidence.

### Viva Use

- Referenced when asked:
  - “Why not just use ETHICS accuracy?”
  - “What is missing from existing work?”

---

## Chapter 4: Theoretical Framework

**Thesis Role:** Bridge philosophy → computation.
**Viva Role:** Formal pivot.

### Structural Requirements

Introduce diagnostic abstractions:

- Moral Overconfidence
- Normative Flattening
- Expressive–Cognitive Confusion

Each must be:

1. Defined conceptually
2. Defined operationally

Example:

- Moral Overconfidence
  - Conceptual: Treating moral stance as epistemic certainty.
  - Operational: High-confidence categorical outputs under disagreement.

### Viva Use

- Derive metrics on paper before showing CLI behaviour.
- This is where “ethics” becomes “signal.”

---

## Chapter 5: Methodology

**Thesis Role:** Reproducibility and rigour.
**Viva Role:** Architecture walkthrough.

### Structural Requirements

Include a pipeline diagram:
