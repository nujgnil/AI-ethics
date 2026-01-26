# Viva / Defence Runbook

Format: Report-led + CLI Demonstration
Goal: Present the project as a technical system with empirical behaviour, where philosophical implications emerge from observed results rather than rhetoric.

---

## Overview

This defence is structured as a live technical walkthrough:

- The report is the authoritative artefact:

  - problem formulation
  - theory
  - architecture
  - methodology
  - results
  - interpretation
- The CLI is the empirical proof:

  - shows the pipeline exists
  - demonstrates model behaviour in real time
  - grounds claims in observable output

The flow alternates between:

Report → CLI → Report → CLI → Report

The examiner experiences the project as a system audit, not a talk.

---

## Phase 1 — Orientation (Report-led, 3–4 minutes)

Open the report (PDF or printed).

Say:

I will structure this as a technical walkthrough of the system described in the report,
and at key points I will switch to the CLI to demonstrate its behaviour live.

Navigate to:

- Chapter 1: Introduction
  - Problem Statement
  - Research Questions

Summarise in one sentence:

This project tests whether AI systems treat moral language as factual,
and operationalises that behaviour as something measurable.

No demo yet.
This anchors why the system exists.

---

## Phase 2 — System Architecture (Report-led, 3–4 minutes)

Navigate to:

- Chapter 5: Methodology
- Chapter 6: Experimental Framework

Point to:

- Pipeline diagram or description:
  - Prompt class
  - Model
  - Response
  - Scoring

Explain verbally:

- Moral prompts are grouped by task type.
- Each model response is analysed for categorical assertion vs plural framing.
- These are aggregated into diagnostic metrics.

Transition:

I will now show what this looks like when run.

---

## Phase 3 — Live CLI Demonstration (5–6 minutes)

Open terminal or notebook.

Run:

```bash
python run_probe.py --type disagreement --model distilbert
```
