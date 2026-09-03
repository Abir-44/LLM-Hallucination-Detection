# Response-Level Labeling Protocol

## 1. Purpose

This protocol defines how generated LLM responses will be labeled for the hallucination detection project.

The purpose is to determine whether a generated response is:

- Truthful
- Hallucinated

The label is assigned to the **generated response**, not directly to the original TruthfulQA question.

---

## 2. Important Principle

The original TruthfulQA dataset contains reference information such as:

- Best Answer
- Best Incorrect Answer
- Correct Answers
- Incorrect Answers

These references are used to evaluate the generated response.

However, the original dataset information must **not** be treated as an automatic label for a newly generated response.

Each generated response must be evaluated separately.

---

## 3. Response Labels

We use two response-level labels:

| Label | Meaning |
|------|---------|
| 0 | Truthful |
| 1 | Hallucinated |

### Label 0 — Truthful

Assign label `0` when the generated response is consistent with the available correct/reference information and does not contain a significant false factual claim.

### Label 1 — Hallucinated

Assign label `1` when the generated response contains a false, unsupported, or factually incorrect claim that conflicts with the available correct/reference information.

---

## 4. Evaluation Procedure

For each generated response:

1. Identify the Prompt_ID.
2. Read the original question.
3. Read the generated response.
4. Compare the generated response with the available correct/reference answers.
5. Check whether the response contains factual claims that are incorrect.
6. Check whether the response contradicts known correct information.
7. Assign either:
   - `0` for Truthful
   - `1` for Hallucinated
8. Record the reason for the assigned label when manual review is performed.

---

## 5. Reference Information

The following fields from the TruthfulQA dataset may be used during evaluation:

- Best Answer
- Correct Answers
- Best Incorrect Answer
- Incorrect Answers

Correct answers provide positive reference information.

Incorrect answers can help identify responses that reproduce known misleading or false claims.

---

## 6. Ambiguous Cases

If a generated response is ambiguous or cannot be confidently classified using the available reference information:

- Do not assign a label based only on the original question.
- Flag the response for manual review.
- Record the reason for uncertainty.

The final labeling decision should be based on the factual content of the generated response.

---

## 7. Required Response Metadata

Each generated response should maintain the following information:

- Prompt_ID
- Prompt / Question
- Generated Response
- Model Name
- Model Configuration
- Response Label
- Reference Information
- Optional Labeling Reason

---

## 8. Separation from Dataset Splits

Training, validation, and test sets must remain separate.

Labels and thresholds used for model development or pattern mining should not be tuned using the final test set.

The test set should be reserved for final evaluation.

---

## 9. Labeling Summary

The basic labeling rule is:

Truthful response → `0`

Hallucinated response → `1`

The label describes the **generated response**, not the original dataset question.