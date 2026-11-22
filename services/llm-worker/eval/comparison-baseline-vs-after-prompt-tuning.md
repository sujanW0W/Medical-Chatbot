# Before/After: prompt tightening for groundedness

## Why this rerun

The `baseline` eval run surfaced a safety-relevant failure mode, not just a quality score to shrug at: on two of the eight `web_search`-routed queries, the pipeline was given weak or stale retrieved/web context and responded by confidently fabricating specific medical details anyway. Inventing a 2026 COVID-19 booster schedule from context that was actually dated 2021, and generating a detailed, entirely ungrounded answer about GLP-1 research when the web search step had errored. The system prompt at the time (`src/prompts.py`) only said to answer "based on the provided context," with nothing telling the model what to do when that context didn't actually support an answer.

## What we changed, and what we targeted

The `system_prompt` was rewritten to do two things explicitly:

1. Forbid adding facts, figures, drug names, or statistics not stated in the provided context, even if the model "knows" them from pretraining.
2. Instruct the model to say it doesn't have reliable information, rather than guess, when context is empty, erroneous, or insufficient.

Target: raise `judge_faithfulness` (does the answer stay grounded in what was actually retrieved?) without tanking `judge_answer_relevance` (does it still actually answer the question?).

## Results

| Metric | `baseline` | `after-prompt-tuning` | Δ |
|---|---|---|---|
| Router accuracy | 100% | 100% | — |
| Avg retrieval relevance | 0.671 | 0.671 | — (retrieval is untouched by this change) |
| **Avg judge faithfulness (1–5)** | **4.05** | **4.70** | **+0.65** |
| Avg judge answer relevance (1–5) | 4.80 | 4.65 | −0.15 |
| Avg latency (s) | 28.94 | 25.8 | not attributable to the prompt change |

Full per-query data: [`results-baseline.json`](results-baseline.json) [`eval-report-baseline.md`](eval-report-baseline.md) vs. [`results-after-prompt-tuning.json`](results-after-prompt-tuning.json) [`eval-report-after-prompt-tuning.md`](eval-report-after-prompt-tuning.md).

Each labeled `results-*.json` embeds the exact `system_prompt` text that produced it, so the two runs are self-documenting without cross-referencing git history.

## We got what we targeted, to some extent

The two cases that motivated this experiment now behave as intended:

- **COVID-19 booster schedule** — previously a confident, fabricated 2026 schedule. After tuning: *"I don't have reliable information on the current recommended COVID-19 booster schedule... the provided context includes outdated information from 2021... consult official sources like the CDC."*
- **GLP-1 latest research** (context had errored) — faithfulness improved 3 → 4, with the answer now sticking closer to the (thin) context it actually had instead of inventing trial names wholesale.

### Honest trade-offs — this did not fully solve it

- **It can be overly conservative.** On `appendicitis-symptoms-treatment`, the after-tuning answer declined to list classic symptoms the context did in fact contain, dropping `answer_relevance` 5 → 3. Stricter grounding sometimes costs completeness on borderline cases.
- **It didn't eliminate fabrication.** `weight-loss-drugs-newest` actually got *worse* (faithfulness 3 → 2) — the model invented a specific drug name under the new prompt too. The instruction makes confident fabrication rarer, not impossible.

**Conclusion:** net improvement in the direction that matters most for a medical-information chatbot, fabricating a 2026 vaccine schedule from 2021 data is a worse failure than giving an overly hedged answer, and the mechanism is clearly visible in the actual model outputs, not just the score.
