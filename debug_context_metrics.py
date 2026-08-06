"""
Fast diagnostic for the context_precision=0.0 / context_recall=0.0 issue.

Runs ONE question, prints the actual retrieved chunks side-by-side with
the reference answer so you can eyeball whether retrieval is the problem,
then calls context_precision/context_recall directly (not through the
full evaluate() pipeline) so you get an answer in ~2-4 minutes instead
of waiting through the whole 8-question run again.

Usage: python debug_context_metrics.py
"""

import asyncio

from ragas.metrics import ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.dataset_schema import SingleTurnSample
from langchain_ollama import ChatOllama

from rag_pipeline import build_rag_chain
from eval_ragas import run_pipeline, JUDGE_MODEL, TEST_SET


def main():
    item = TEST_SET[0]  # just the first question

    print("Building RAG chain...")
    chain = build_rag_chain()

    print(f"\nQuestion: {item['question']}")
    output = run_pipeline(chain, item["question"])

    print(f"\n{'=' * 60}\nREFERENCE (ground truth you wrote)\n{'=' * 60}")
    print(item["ground_truth"])

    print(f"\n{'=' * 60}\nSYSTEM ANSWER\n{'=' * 60}")
    print(output["answer"])

    print(f"\n{'=' * 60}\nRETRIEVED CONTEXTS ({len(output['contexts'])} chunks)\n{'=' * 60}")
    for i, c in enumerate(output["contexts"]):
        print(f"\n--- chunk {i} ---")
        print(c[:400])

    # >>> Look at the printed contexts above before reading the scores below.
    # If they're clearly ABOUT the paper's actual content (methodology,
    # results, etc.) and roughly match what the reference talks about,
    # then 0.0 scores below point to a metric/schema issue, not retrieval.
    # If the chunks look irrelevant or empty, retrieval itself is the bug.

    sample = SingleTurnSample(
        user_input=item["question"],
        response=output["answer"],
        retrieved_contexts=output["contexts"],
        reference=item["ground_truth"],
    )

    judge_llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0))
    context_precision = ContextPrecision(llm=judge_llm)
    context_recall = ContextRecall(llm=judge_llm)

    async def score():
        p = await context_precision.single_turn_ascore(sample)
        r = await context_recall.single_turn_ascore(sample)
        return p, r

    print(f"\n{'=' * 60}\nSCORING (this is the slow part, ~2-4 min)\n{'=' * 60}")
    precision, recall = asyncio.run(score())
    print(f"context_precision: {precision}")
    print(f"context_recall:    {recall}")


if __name__ == "__main__":
    main()