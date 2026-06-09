import json
import os
import logging
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent.parent


def compare_reranker(
    test_data_file: str,
    original_rerank_model: str,
    finetuned_rerank_model: str,
    top_k: int = 3,
    output_dir: str = None,
):
    from sentence_transformers import CrossEncoder
    from src.retriever import OpsRetriever
    from eval.eval_retrievers import RetrievalMetrics
    import pandas as pd

    with open(test_data_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    pdf_path = str(_BASE_DIR / "data" / "文档2.pdf") if (_BASE_DIR / "data" / "文档2.pdf").exists() else None
    retriever = OpsRetriever(pdf_path=pdf_path)

    if not retriever.ensemble:
        logger.error("混合检索器不可用，无法评估")
        return

    logger.info(f"加载原始 Reranker: {original_rerank_model}")
    original_reranker = CrossEncoder(original_rerank_model)

    logger.info(f"加载微调 Reranker: {finetuned_rerank_model}")
    try:
        finetuned_reranker = CrossEncoder(finetuned_rerank_model)
    except Exception as e:
        logger.error(f"微调 Reranker 加载失败: {e}")
        return

    results = {"original": {}, "finetuned": {}}

    for label, reranker in [("original", original_reranker), ("finetuned", finetuned_reranker)]:
        agg = {m: [] for m in ["recall", "precision", "hit_rate", "mrr", "ndcg", "map"]}
        per_query = []

        for i, item in enumerate(test_data, 1):
            query = item["query"]
            relevant_ids = item.get("relevant_ids", [])

            if not relevant_ids:
                continue

            try:
                docs = retriever.ensemble.invoke(query)
                docs = retriever._deduplicate(docs)
                if not docs:
                    continue

                pairs = [(query, d.page_content) for d in docs]
                scores = reranker.predict(pairs)
                ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                top_docs = [doc for doc, _ in ranked[:top_k]]
                retrieved_ids = [d.metadata.get("doc_id", "") for d in top_docs]
            except Exception as e:
                logger.warning(f"[{label}] 查询 '{query[:30]}' 失败: {e}")
                continue

            metrics = RetrievalMetrics.all_metrics(retrieved_ids, relevant_ids)
            for k, v in metrics.items():
                agg[k].append(v)

            per_query.append({
                "query": query,
                "reranker": label,
                **metrics,
            })

        if any(agg.values()):
            results[label] = {k: sum(v) / len(v) for k, v in agg.items() if v}
            results[label]["num_queries"] = len(per_query)

        logger.info(f"[{label}] 评估完成: {len(per_query)} 条查询")

    df_original = pd.DataFrame([results.get("original", {})], index=["Original_Reranker"])
    df_finetuned = pd.DataFrame([results.get("finetuned", {})], index=["Finetuned_Reranker"])
    df_compare = pd.concat([df_original, df_finetuned])

    print("\n" + "=" * 70)
    print("  Reranker 微调前后对比报告")
    print("=" * 70)
    print(df_compare.to_string())
    print("=" * 70)

    if "ndcg" in df_compare.columns:
        orig_ndcg = df_compare.loc["Original_Reranker", "ndcg"]
        ft_ndcg = df_compare.loc["Finetuned_Reranker", "ndcg"]
        delta = ft_ndcg - orig_ndcg
        pct = (delta / orig_ndcg * 100) if orig_ndcg > 0 else 0
        print(f"\n  NDCG 变化: {orig_ndcg:.4f} → {ft_ndcg:.4f} ({'+' if delta > 0 else ''}{delta:.4f}, {pct:+.1f}%)")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"reranker_compare_{ts}.csv")
        df_compare.to_csv(csv_path, encoding="utf-8-sig")
        logger.info(f"对比报告已保存: {csv_path}")

        _generate_compare_html(df_compare, output_dir, ts)

    return df_compare


def compare_llm(
    test_data_file: str,
    api_llm,
    lora_infer,
    retriever,
    top_k: int = 3,
    output_dir: str = None,
):
    from eval.eval_retrievers import RetrievalMetrics
    import pandas as pd

    with open(test_data_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = {"api_llm": {}, "lora_llm": {}}

    for label, llm_or_infer in [("api_llm", api_llm), ("lora_llm", lora_infer)]:
        answer_quality = []
        retrieval_metrics = []

        for i, item in enumerate(test_data, 1):
            query = item["query"]
            relevant_ids = item.get("relevant_ids", [])
            ground_truth = item.get("ground_truth", "")

            try:
                docs = retriever.get_ensemble_rerank_docs(query, top_k=top_k)
                contexts = [d.page_content for d in docs]
                context_str = "\n\n".join(contexts)

                if label == "api_llm":
                    prompt = f"根据以下资料回答问题：\n{context_str}\n\n问题：{query}"
                    response = llm_or_infer.invoke(prompt)
                    answer = response.content.strip()
                else:
                    answer = llm_or_infer.generate(query)

                has_answer = len(answer) > 20
                has_format = any(kw in answer for kw in ["故障现象", "排查", "修复", "命令"])
                answer_quality.append({
                    "query": query,
                    "has_answer": has_answer,
                    "has_format": has_format,
                    "answer_len": len(answer),
                })

            except Exception as e:
                logger.warning(f"[{label}] 查询失败: {e}")
                answer_quality.append({
                    "query": query,
                    "has_answer": False,
                    "has_format": False,
                    "answer_len": 0,
                })

        if answer_quality:
            total = len(answer_quality)
            results[label] = {
                "answer_rate": sum(1 for q in answer_quality if q["has_answer"]) / total,
                "format_rate": sum(1 for q in answer_quality if q["has_format"]) / total,
                "avg_answer_len": sum(q["answer_len"] for q in answer_quality) / total,
                "num_queries": total,
            }

    df_api = pd.DataFrame([results.get("api_llm", {})], index=["API_LLM"])
    df_lora = pd.DataFrame([results.get("lora_llm", {})], index=["LoRA_LLM"])
    df_compare = pd.concat([df_api, df_lora])

    print("\n" + "=" * 70)
    print("  LLM 微调前后对比报告")
    print("=" * 70)
    print(df_compare.to_string())
    print("=" * 70)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"llm_compare_{ts}.csv")
        df_compare.to_csv(csv_path, encoding="utf-8-sig")
        logger.info(f"对比报告已保存: {csv_path}")

    return df_compare


def _generate_compare_html(df, output_dir, timestamp):
    html_path = os.path.join(output_dir, f"compare_report_{timestamp}.html")

    metrics = [c for c in df.columns if c != "num_queries"]

    rows = ""
    for method in df.index:
        cells = f"<td><strong>{method}</strong></td>"
        for col in metrics:
            val = df.loc[method, col]
            pct = val * 100 if val <= 1.0 else val
            color = "#22C55E" if "Finetuned" in method or "LoRA" in method else "#3B82F6"
            cells += f'<td><div class="bar-container"><div class="bar" style="width:{min(pct, 100)}%;background:{color}"></div><span class="bar-text">{val:.4f}</span></div></td>'
        rows += f"<tr>{cells}</tr>"

    header = "".join(f"<th>{c}</th>" for c in metrics)

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8"><title>微调对比报告</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 40px; }}
h1 {{ color: #60a5fa; text-align: center; }}
h2 {{ color: #94a3b8; margin-top: 30px; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th, td {{ padding: 12px 16px; border: 1px solid #334155; text-align: left; }}
th {{ background: #1e293b; color: #94a3b8; }}
.bar-container {{ position: relative; width: 100%; height: 24px; background: #1e293b; border-radius: 4px; overflow: hidden; }}
.bar {{ height: 100%; border-radius: 4px; }}
.bar-text {{ position: absolute; right: 8px; top: 2px; font-size: 12px; color: #e2e8f0; font-weight: 600; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
.badge-green {{ background: #166534; color: #4ade80; }}
.badge-blue {{ background: #1e3a5f; color: #60a5fa; }}
</style></head><body>
<h1>微调前后对比报告</h1>
<p style="text-align:center;color:#64748b;">生成时间: {timestamp}</p>
<table><tr><th>Method</th>{header}</tr>{rows}</table>
<p style="text-align:center;color:#64748b;font-size:13px;">Generated by SmartOps Eval</p>
</body></html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML对比报告: {html_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="微调前后对比评估")
    parser.add_argument("--type", choices=["reranker", "llm", "both"], default="reranker")
    parser.add_argument("--test-data", default=str(_BASE_DIR / "eval" / "test_queries.json"))
    parser.add_argument("--original-reranker", default=str(_BASE_DIR / "model" / "bge-reranker-v2-m3"))
    parser.add_argument("--finetuned-reranker", default=str(_BASE_DIR / "model" / "lora-reranker" / "best_reranker"))
    parser.add_argument("--output-dir", default=str(_BASE_DIR / "eval" / "reports"))
    args = parser.parse_args()

    if args.type in ["reranker", "both"]:
        if not Path(args.finetuned_reranker).exists():
            logger.error(f"微调 Reranker 不存在: {args.finetuned_reranker}")
            logger.info("请先运行: python src/finetune.py --mode reranker")
        else:
            compare_reranker(
                args.test_data,
                args.original_reranker,
                args.finetuned_reranker,
                output_dir=args.output_dir,
            )

    if args.type in ["llm", "both"]:
        logger.info("LLM 对比需要 API 和 LoRA 模型同时可用")
        logger.info("请确保: 1) API 可用  2) LoRA 模型已训练并配置")
