"""
Before/after comparison: baseline (plain vector search) retriever vs. the
hybrid (BM25 + vector + cross-encoder rerank) retriever.
"""

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_pipeline import build_rag_chain
from eval_ragas import TEST_SET, QUICK_TEST_SIZE, JUDGE_MODEL, run_pipeline, build_eval_dataset


def run_eval_for(use_hybrid: bool, active_test_set: list, judge_llm, judge_embeddings, run_config, metrics):
    """Build a chain with the given retriever mode and score it."""
    label = "hybrid" if use_hybrid else "baseline"
    print(f"\n{'#' * 60}")
    print(f"# Building chain: {label} retriever")
    print(f"{'#' * 60}")

    chain = build_rag_chain(use_hybrid=use_hybrid)

    print(f"Running {len(active_test_set)} questions through the {label} pipeline...")
    dataset = build_eval_dataset(chain, active_test_set)

    print(f"\nScoring {label} results with RAGAS...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    df = results.to_pandas()
    out_path = f"ragas_results_{label}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    return df


def main():
    active_test_set = TEST_SET[:QUICK_TEST_SIZE] if QUICK_TEST_SIZE else TEST_SET
    if QUICK_TEST_SIZE:
        print(f"QUICK_TEST_SIZE={QUICK_TEST_SIZE} — running a sanity check, not the full comparison.\n")

    judge_llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
    run_config = RunConfig(timeout=480, max_workers=1, max_retries=2, max_wait=30)
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
    metric_names = [m.name for m in metrics]

    baseline_df = run_eval_for(False, active_test_set, judge_llm, judge_embeddings, run_config, metrics)
    hybrid_df = run_eval_for(True, active_test_set, judge_llm, judge_embeddings, run_config, metrics)

    print(f"\n{'=' * 70}")
    print("BEFORE / AFTER COMPARISON — average scores")
    print(f"{'=' * 70}")
    print(f"  {'metric':20s}  {'baseline':>10s}  {'hybrid':>10s}  {'delta':>10s}")
    for metric in metric_names:
        base_avg = baseline_df[metric].mean()
        hybrid_avg = hybrid_df[metric].mean()
        delta = hybrid_avg - base_avg
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:20s}  {base_avg:10.3f}  {hybrid_avg:10.3f}  {sign}{delta:.3f}")

    print(
        "\nPositive delta = hybrid retriever scored higher on that metric. "
        "context_precision and context_recall are the ones most directly "
        "attributable to the retriever change — faithfulness/answer_relevancy "
        "can shift too since a different retrieved context changes what the "
        "LLM has to work with, but the retriever isn't the only factor there."
    )


if __name__ == "__main__":
    main()
