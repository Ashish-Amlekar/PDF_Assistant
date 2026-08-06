"""
Evaluate the RAG pipeline's retrieval + answer quality using RAGAS.
"""

from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_pipeline import build_rag_chain

# The model used to JUDGE answers (separate from the mistral model your
# app uses to generate them). A more instruction-tuned model produces
# valid structured JSON far more reliably, which matters a lot for the
# faithfulness and context_precision metrics. Swap this if you pull a
# different model — must already exist locally via `ollama pull <name>`.
JUDGE_MODEL = "qwen2.5:7b-instruct"  # fallback to "mistral" if you don't want to pull anything new

# Sanity-check switch: when the judge model or config changes, a full run
# can take hours before you find out it still doesn't work. Set this to a
# small number (e.g. 2) to run just the first N questions first — takes a
# few minutes instead of an hour+ — then bump it back to None once you've
# confirmed real (non-NaN) scores are coming back.
QUICK_TEST_SIZE = None  # set to None to run the full TEST_SET

# ---------------------------------------------------------------------
# 1. Your golden test set — EDIT THIS to match whatever PDF(s) you've
#    actually processed. Each entry needs a question and the correct
#    reference answer (write these yourself by reading the source PDF —
#    that's what makes it "ground truth").
# ---------------------------------------------------------------------
TEST_SET = [
    {
        "question": "What was the main research question of the paper?",
        "ground_truth": (
            "Whether the size-based stratification the authors previously observed in "
            "drying binary colloidal mixtures also occurs in more complex ternary and "
            "polydisperse suspensions, and how the mixture composition affects the "
            "resulting layering pattern."
        ),
    },
    {
        "question": "What methodology did the authors use?",
        "ground_truth": (
            "Brownian dynamics (Langevin dynamics) computer simulations of colloidal "
            "particles drying beneath a descending air-water interface, using the LAMMPS "
            "simulation code for the ternary mixtures and a custom in-house code for the "
            "polydisperse mixtures. Particles interacted through a short-range repulsive "
            "Yukawa potential."
        ),
    },
    {
        "question": "What were the key findings or results?",
        "ground_truth": (
            "Larger particles consistently migrate toward the bottom of the drying film "
            "while smaller particles accumulate near the top. Ternary mixtures with large "
            "enough size ratios separate into three distinct layers. Polydisperse "
            "suspensions show a continuous size gradient, with a larger mean diameter near "
            "the bottom and a smaller mean diameter near the top, consistent with a "
            "proposed power-law model relating particle displacement to particle diameter."
        ),
    },
    {
        "question": "What limitations did the authors acknowledge?",
        "ground_truth": (
            "The simulations neglect hydrodynamic interactions between particles and "
            "treat the solvent as uniform, without allowing its temperature or other "
            "properties to vary during drying. The theoretical model only accounts for "
            "concentration gradients and relies on several simplifying approximations, so "
            "the authors say a more complete theory based on collective diffusion is "
            "needed."
        ),
    },
    {
        "question": "What term did the authors coin for the segregation mechanism, and what does it mean?",
        "ground_truth": (
            "They call it colloidal diffusiophoresis: the motion of one colloidal species "
            "in response to a concentration gradient of another colloidal species, which "
            "they describe as a special case of cross-diffusion."
        ),
    },
    {
        "question": "What relationship does the proposed model predict between particle size and segregation velocity?",
        "ground_truth": (
            "A power-law relationship: the difference in velocity between a particle of a "
            "given diameter and a particle of average diameter scales with diameter raised "
            "to the power (2 minus alpha), where alpha comes from how the sedimentation "
            "coefficient depends on particle size."
        ),
    },
    {
        "question": "Why did the authors need a custom in-house simulation code for the polydisperse mixtures?",
        "ground_truth": (
            "Because LAMMPS did not support particle-level polydispersity at the time, so "
            "they developed their own Brownian dynamics code to model particle diameters "
            "drawn from a continuous Gaussian distribution."
        ),
    },
    {
        "question": "How many layers formed in mixture C3, and what did each layer contain?",
        "ground_truth": (
            "Three well-defined layers formed: a top layer containing only the smallest "
            "particles, a middle layer containing small and intermediate-size particles, "
            "and a bottom layer containing all three particle species."
        ),
    },
    # Add more of your own for a more reliable score. 5-10 is a bare
    # minimum; 20+ starts to give you numbers you can actually trust.
]


def run_pipeline(chain, question: str) -> dict:
    """
    Run one question through the RAG chain and pull out exactly what
    RAGAS needs: the full answer and the FULL text of each retrieved
    chunk (not the 150-char preview that ask_question() shows in the UI).
    """
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "contexts": [doc.page_content for doc in result["source_documents"]],
    }


def build_eval_dataset(chain, test_set: list) -> EvaluationDataset:
    """Run every question in test_set through the pipeline and assemble
    the dataset RAGAS expects."""
    rows = []
    for item in test_set:
        print(f"  Running: {item['question']}")
        output = run_pipeline(chain, item["question"])
        rows.append({
            "user_input": item["question"],
            "response": output["answer"],
            "retrieved_contexts": output["contexts"],
            "reference": item["ground_truth"],
        })
    return EvaluationDataset.from_list(rows)


def main():
    unfilled = [t for t in TEST_SET if "REPLACE ME" in t["ground_truth"]]
    if unfilled:
        print(
            f"WARNING: {len(unfilled)} test question(s) still have placeholder "
            "ground_truth answers. Edit TEST_SET in this file with real answers "
            "from your PDF before trusting the scores.\n"
        )

    print("Building RAG chain (loads vector store + hybrid retriever)...")
    chain = build_rag_chain()

    active_test_set = TEST_SET[:QUICK_TEST_SIZE] if QUICK_TEST_SIZE else TEST_SET
    mode_note = f" (QUICK_TEST_SIZE={QUICK_TEST_SIZE} — sanity check, not a full run)" if QUICK_TEST_SIZE else ""
    print(f"\nRunning {len(active_test_set)} questions through the pipeline...{mode_note}")
    dataset = build_eval_dataset(chain, active_test_set)

    # Wrap the local Ollama LLM + embeddings so RAGAS uses them as the
    # judge instead of defaulting to OpenAI. JUDGE_MODEL is separate from
    # the "mistral" model your app uses to generate answers.
    judge_llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    # RAGAS defaults to firing many judging calls at once (fine for a fast
    # hosted API). A local Ollama model can really only serve one request
    # at a time — concurrent requests just queue up behind each other,
    # which is what caused most of the earlier timeouts. max_workers=1
    # fully serializes calls; timeout=300 caps a single attempt.
    #
    # max_retries/max_wait matter A LOT here: RAGAS's default retry count
    # is high, and each retry waits longer than the last (exponential
    # backoff) — that's exactly what turned one failing job into a
    # 5+ hour hang last run. Capping retries to 2 with a short max wait
    # means a genuinely-stuck job fails in minutes, not hours, and you
    # actually see the NaN instead of staring at a frozen progress bar.
    local_run_config = RunConfig(timeout=300, max_workers=1, max_retries=2, max_wait=30)

    # context_precision structurally struggles with small local judges (see
    # the module docstring) — it asks for a strict JSON verdict per
    # retrieved chunk, and 7B-class models rarely produce valid JSON on the
    # first try. If it's still all-NaN after switching JUDGE_MODEL, comment
    # it out of this list rather than burning another hour re-confirming it.
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

    print("\nScoring with RAGAS (this calls the local LLM once or twice per "
          "question per metric, so it's slower than a normal chat answer — "
          "expect this to take a while since calls are now fully serialized)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=local_run_config,
    )

    df = results.to_pandas()
    metric_names = [m.name for m in metrics]

    print("\n" + "=" * 60)
    print("RESULTS (per question)")
    print("=" * 60)
    print(df[["user_input"] + metric_names].to_string(index=False))

    print("\n" + "=" * 60)
    print("AVERAGE SCORES")
    print("=" * 60)
    for metric in metric_names:
        print(f"  {metric:20s}: {df[metric].mean():.3f}")

    out_path = "ragas_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
