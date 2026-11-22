"""
Evaluation harness for the multi-agent RAG pipeline.

Runs a fixed set of golden queries through the real LangGraph pipeline
(src.orchestrator.graph) and scores:
  - router accuracy    : did it pick retrieve vs. web_search correctly?
  - retrieval relevance: reuses RetrievalAgent._calculate_relevance (already
                          computed per-query, never surfaced anywhere before)
  - answer quality     : an LLM-judge (reference-free) rating faithfulness
                          and relevance of the final answer, 1-5 each
  - latency            : wall-clock time per query

Usage (inside the llm-worker container, where deps + env vars are present):
    python eval/run_eval.py                       # overwrites results.json / eval-report.md
    python eval/run_eval.py --label baseline       # writes results-baseline.json / eval-report-baseline.md
    python eval/run_eval.py --label after-prompt-tuning

Use --label for any run you want to keep around for a before/after
comparison (e.g. a prompt change) instead of letting the next run overwrite
it. The current src/prompts.py system_prompt text is saved into the labeled
results.json alongside the scores, so each labeled run is self-describing.
"""
import argparse
import json
import os
import re
import statistics
import time
from datetime import datetime

from src.llm_utils import strip_reasoning
from src.orchestrator import graph, llm
from src.prompts import system_prompt as current_system_prompt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_QUERIES_PATH = os.path.join(BASE_DIR, "golden_queries.json")

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating the quality of an AI medical assistant's answer. "
    "Rate the answer on two dimensions, each from 1 (poor) to 5 (excellent):\n"
    "1. faithfulness: is the answer grounded in the provided context, with no "
    "fabricated claims not supported by it?\n"
    "2. answer_relevance: does the answer actually address the user's question?\n\n"
    "Respond with ONLY a JSON object, no markdown fences, no other text, in "
    "exactly this shape:\n"
    '{"faithfulness": <int 1-5>, "answer_relevance": <int 1-5>, "rationale": "<one sentence>"}'
)


def build_context_summary(result: dict, max_chars_per_item: int = 500) -> str:
    """Render whatever context the pipeline actually used, for the judge to see."""
    parts = []

    retrieved_docs = result.get("retrieved_docs")
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs, start=1):
            parts.append(
                f"[KB doc {i}] {doc.page_content[:max_chars_per_item]}")

    web_results = result.get("web_results")
    if web_results:
        for i, r in enumerate(web_results, start=1):
            snippet = r.get("snippet", "")[:max_chars_per_item]
            parts.append(f"[Web result {i}] {r.get('title', '')}: {snippet}")

    return "\n\n".join(parts) if parts else "(no context retrieved)"


def judge_response(query: str, context: str, answer: str) -> dict:
    user_content = (
        f"Question: {query}\n\n"
        f"Context provided to the assistant:\n{context}\n\n"
        f"Assistant's answer:\n{answer}"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = llm.invoke(messages)
    raw = strip_reasoning(response.content)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
        return {
            "faithfulness": parsed.get("faithfulness"),
            "answer_relevance": parsed.get("answer_relevance"),
            "rationale": parsed.get("rationale"),
        }
    except (json.JSONDecodeError, AttributeError):
        return {
            "faithfulness": None,
            "answer_relevance": None,
            "rationale": f"[judge parse failed] raw={raw[:200]}",
        }


def run_single_query(item: dict) -> dict:
    query = item["query"]
    expected_route = item["expected_route"]

    record = {
        "id": item["id"],
        "query": query,
        "expected_route": expected_route,
        "actual_route": None,
        "route_correct": None,
        "retrieval_relevance": None,
        "answer": None,
        "judge_faithfulness": None,
        "judge_answer_relevance": None,
        "judge_rationale": None,
        "latency_seconds": None,
        "error": None,
    }

    start = time.time()
    try:
        result = graph.invoke(
            {"messages": [{"role": "user", "content": query}]})
    except Exception as e:
        record["error"] = f"graph invocation failed: {e}"
        record["latency_seconds"] = round(time.time() - start, 2)
        return record
    record["latency_seconds"] = round(time.time() - start, 2)

    # The retrieval agent sets last_tool="retriever" (not "retrieve") -
    # normalize here so golden_queries.json can stay human-readable.
    actual_route = "retrieve" if result.get(
        "last_tool") == "retriever" else "web_search"
    record["actual_route"] = actual_route
    record["route_correct"] = actual_route == expected_route

    if actual_route == "retrieve":
        record["retrieval_relevance"] = result.get("relevance_score")

    answer = result["messages"][-1].content
    record["answer"] = answer

    try:
        context = build_context_summary(result)
        judged = judge_response(query, context, answer)
        record["judge_faithfulness"] = judged["faithfulness"]
        record["judge_answer_relevance"] = judged["answer_relevance"]
        record["judge_rationale"] = judged["rationale"]
    except Exception as e:
        record["error"] = f"judge failed: {e}"

    return record


def compute_aggregates(records: list) -> dict:
    total = len(records)
    route_judged = [r for r in records if r["route_correct"] is not None]
    relevance_scores = [r["retrieval_relevance"]
                        for r in records if r["retrieval_relevance"] is not None]
    faithfulness_scores = [r["judge_faithfulness"] for r in records if isinstance(
        r["judge_faithfulness"], (int, float))]
    answer_relevance_scores = [r["judge_answer_relevance"] for r in records if isinstance(
        r["judge_answer_relevance"], (int, float))]
    latencies = [r["latency_seconds"]
                 for r in records if r["latency_seconds"] is not None]

    return {
        "total_queries": total,
        "errors": sum(1 for r in records if r["error"] is not None),
        "router_accuracy": round(sum(1 for r in route_judged if r["route_correct"]) / len(route_judged), 3) if route_judged else None,
        "avg_retrieval_relevance": round(statistics.mean(relevance_scores), 3) if relevance_scores else None,
        "avg_judge_faithfulness": round(statistics.mean(faithfulness_scores), 2) if faithfulness_scores else None,
        "avg_judge_answer_relevance": round(statistics.mean(answer_relevance_scores), 2) if answer_relevance_scores else None,
        "avg_latency_seconds": round(statistics.mean(latencies), 2) if latencies else None,
    }


def write_markdown_report(aggregates: dict, records: list, report_path: str, label: str | None = None) -> None:
    router_accuracy_display = (
        f"{aggregates['router_accuracy'] * 100:.1f}%"
        if aggregates["router_accuracy"] is not None
        else "n/a"
    )

    title_suffix = f" — {label}" if label else ""
    lines = [
        f"# RAG Pipeline Evaluation Report{title_suffix}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total queries | {aggregates['total_queries']} |",
        f"| Errors | {aggregates['errors']} |",
        f"| Router accuracy | {router_accuracy_display} |",
        f"| Avg retrieval relevance (in-KB queries) | {aggregates['avg_retrieval_relevance']} |",
        f"| Avg judge faithfulness (1-5) | {aggregates['avg_judge_faithfulness']} |",
        f"| Avg judge answer relevance (1-5) | {aggregates['avg_judge_answer_relevance']} |",
        f"| Avg latency (s) | {aggregates['avg_latency_seconds']} |",
        "",
        "## Per-query results",
        "",
        "| Query | Expected | Actual | Match | Retrieval Relevance | Faithfulness | Answer Relevance | Latency (s) | Notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in records:
        if r["route_correct"] is True:
            match = "✅"
        elif r["route_correct"] is False:
            match = "❌"
        else:
            match = "—"

        retrieval_rel = f"{r['retrieval_relevance']:.2f}" if r["retrieval_relevance"] is not None else "—"
        faith = r["judge_faithfulness"] if r["judge_faithfulness"] is not None else "—"
        ans_rel = r["judge_answer_relevance"] if r["judge_answer_relevance"] is not None else "—"
        latency = r["latency_seconds"] if r["latency_seconds"] is not None else "—"
        notes = str(r["error"] or r.get("judge_rationale") or "")
        if len(notes) > 180:
            notes = notes[:180] + "…"

        query_cell = r["query"].replace("|", "\\|")
        notes_cell = notes.replace("|", "\\|")

        lines.append(
            f"| {query_cell} | {r['expected_route']} | {r['actual_route'] or '—'} | "
            f"{match} | {retrieval_rel} | {faith} | {ans_rel} | {latency} | {notes_cell} |"
        )

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RAG eval harness.")
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label (e.g. 'baseline', 'after-prompt-tuning') to keep this "
             "run's output files alongside others instead of overwriting them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = f"-{args.label}" if args.label else ""
    results_path = os.path.join(BASE_DIR, f"results{suffix}.json")
    report_path = os.path.join(BASE_DIR, f"eval-report{suffix}.md")

    with open(GOLDEN_QUERIES_PATH) as f:
        golden_queries = json.load(f)

    records = []
    for i, item in enumerate(golden_queries, start=1):
        print(f"[{i}/{len(golden_queries)}] running: {item['id']}")
        records.append(run_single_query(item))

    aggregates = compute_aggregates(records)

    with open(results_path, "w") as f:
        json.dump({
            "label": args.label,
            "generated_at": datetime.now().isoformat(),
            "system_prompt": current_system_prompt,
            "aggregates": aggregates,
            "records": records,
        }, f, indent=2)

    write_markdown_report(aggregates, records, report_path, label=args.label)

    print("\n=== Summary ===")
    print(json.dumps(aggregates, indent=2))
    print(f"\nReport written to {report_path}")
    print(f"Raw results written to {results_path}")


if __name__ == "__main__":
    main()
