import json
import math
import logging
import os
import sys
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent.parent


class RetrievalMetrics:

    @staticmethod
    def recall(retrieved: List[str], relevant: List[str]) -> float:
        if not relevant:
            return 0.0
        return len(set(retrieved) & set(relevant)) / len(relevant)

    @staticmethod
    def precision(retrieved: List[str], relevant: List[str]) -> float:
        if not retrieved:
            return 0.0
        return len(set(retrieved) & set(relevant)) / len(retrieved)

    @staticmethod
    def hit_rate(retrieved: List[str], relevant: List[str]) -> float:
        return 1.0 if set(retrieved) & set(relevant) else 0.0

    @staticmethod
    def mrr(retrieved: List[str], relevant: List[str]) -> float:
        relevant_set = set(relevant)
        for rank, rid in enumerate(retrieved, 1):
            if rid in relevant_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg(retrieved: List[str], relevant: List[str]) -> float:
        if not relevant:
            return 0.0
        relevant_set = set(relevant)
        dcg = sum(1.0 / math.log2(rank + 1) for rank, rid in enumerate(retrieved, 1) if rid in relevant_set)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), len(retrieved)) + 1))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def map_score(retrieved: List[str], relevant: List[str]) -> float:
        if not relevant:
            return 0.0
        relevant_set = set(relevant)
        hits = 0
        precision_sum = 0.0
        for rank, rid in enumerate(retrieved, 1):
            if rid in relevant_set:
                hits += 1
                precision_sum += hits / rank
        return precision_sum / len(relevant)

    @classmethod
    def all_metrics(cls, retrieved: List[str], relevant: List[str]) -> Dict[str, float]:
        return {
            "recall": cls.recall(retrieved, relevant),
            "precision": cls.precision(retrieved, relevant),
            "hit_rate": cls.hit_rate(retrieved, relevant),
            "mrr": cls.mrr(retrieved, relevant),
            "ndcg": cls.ndcg(retrieved, relevant),
            "map": cls.map_score(retrieved, relevant),
        }


class RetrievalEvaluator:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def _get_retriever_methods(self, retriever) -> Dict[str, callable]:
        methods = {
            "BM25_Only": retriever.get_bm25_docs,
            "Vector_Only": retriever.get_vector_docs,
            "Ensemble+Rerank": retriever.get_ensemble_rerank_docs,
        }
        if hasattr(retriever, "retriever_and_rerank_with_scores"):
            methods["Ensemble+Rerank+Scores"] = lambda q, top_k=self.top_k: [
                doc for doc, _ in retriever.retriever_and_rerank_with_scores(q, top_k)
            ]
        return methods

    def run(self, test_data: List[Dict], retriever) -> pd.DataFrame:
        logger.info("初始化检索器...")
        methods = self._get_retriever_methods(retriever)

        all_results = {}
        per_query_details = []

        for name, func in methods.items():
            logger.info(f"评估: {name}")
            agg = defaultdict(list)

            for i, item in enumerate(test_data, 1):
                query = item["query"]
                relevant_ids = item.get("relevant_ids", [])

                try:
                    docs = func(query, top_k=self.top_k)
                    retrieved_ids = [d.metadata.get("doc_id", "") for d in docs]
                except Exception as e:
                    logger.warning(f"  [{name}] 查询 '{query[:30]}...' 失败: {e}")
                    retrieved_ids = []

                metrics = RetrievalMetrics.all_metrics(retrieved_ids, relevant_ids)

                for k, v in metrics.items():
                    agg[k].append(v)

                per_query_details.append({
                    "method": name,
                    "query": query,
                    **metrics,
                })

                if i % 10 == 0:
                    logger.info(f"  进度: {i}/{len(test_data)}")

            all_results[name] = {k: sum(v) / len(v) for k, v in agg.items() if v}

        df = pd.DataFrame(all_results).T.round(4)
        return df, pd.DataFrame(per_query_details)

    def run_with_kg(self, test_data: List[Dict], retriever) -> pd.DataFrame:
        kg_available = hasattr(retriever, "kg") and retriever.kg and retriever.kg.is_available

        methods = self._get_retriever_methods(retriever)

        if kg_available:
            methods["KG_Only"] = lambda q, top_k=self.top_k: self._kg_retrieve(retriever, q)
            methods["Ensemble+Rerank+KG"] = lambda q, top_k=self.top_k: self._ensemble_with_kg(retriever, q, top_k)

        all_results = {}
        for name, func in methods.items():
            logger.info(f"评估: {name}")
            agg = defaultdict(list)

            for i, item in enumerate(test_data, 1):
                query = item["query"]
                relevant_ids = item.get("relevant_ids", [])

                try:
                    docs = func(query, top_k=self.top_k) if "KG" not in name else func(query)
                    if isinstance(docs, list) and docs and hasattr(docs[0], "metadata"):
                        retrieved_ids = [d.metadata.get("doc_id", "") for d in docs]
                    else:
                        retrieved_ids = []
                except Exception as e:
                    logger.warning(f"  [{name}] 查询失败: {e}")
                    retrieved_ids = []

                metrics = RetrievalMetrics.all_metrics(retrieved_ids, relevant_ids)
                for k, v in metrics.items():
                    agg[k].append(v)

            all_results[name] = {k: sum(v) / len(v) for k, v in agg.items() if v}

        return pd.DataFrame(all_results).T.round(4)

    def _kg_retrieve(self, retriever, query: str) -> List:
        from langchain_core.documents import Document
        if not retriever.kg or not retriever.kg.is_available:
            return []
        context = retriever.kg.format_graph_context(query, depth=2)
        if not context:
            return []
        return [Document(page_content=context, metadata={"source": "知识图谱", "doc_id": "kg_context"})]

    def _ensemble_with_kg(self, retriever, query: str, top_k: int) -> List:
        docs = retriever.get_ensemble_rerank_docs(query, top_k)
        kg_context = self._kg_retrieve(retriever, query)
        if kg_context and docs:
            return kg_context + docs
        return docs


def print_report(df: pd.DataFrame, title: str = "检索模型对比报告"):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)

    if "ndcg" in df.columns:
        best = df["ndcg"].idxmax()
        print(f"\n  NDCG 最优方法: {best} ({df.loc[best, 'ndcg']:.4f})")
    if "mrr" in df.columns:
        best = df["mrr"].idxmax()
        print(f"  MRR  最优方法: {best} ({df.loc[best, 'mrr']:.4f})")


def save_report(df: pd.DataFrame, per_query_df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "retrieval_eval_report.csv")
    detail_path = os.path.join(output_dir, "retrieval_eval_detail.csv")

    df.to_csv(summary_path, encoding="utf-8-sig")
    per_query_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    logger.info(f"汇总报告: {summary_path}")
    logger.info(f"逐条详情: {detail_path}")


def generate_html_report(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "retrieval_eval_report.html")

    metrics_cols = [c for c in df.columns if c in ["recall", "precision", "hit_rate", "mrr", "ndcg", "map"]]

    bar_colors = {
        "recall": "#3B82F6",
        "precision": "#22C55E",
        "hit_rate": "#F59E0B",
        "mrr": "#EF4444",
        "ndcg": "#8B5CF6",
        "map": "#EC4899",
    }

    rows_html = ""
    for method in df.index:
        cells = f"<td><strong>{method}</strong></td>"
        for col in metrics_cols:
            val = df.loc[method, col]
            pct = val * 100
            color = bar_colors.get(col, "#6B7280")
            cells += f'<td><div class="bar-container"><div class="bar" style="width:{pct}%;background:{color}"></div><span class="bar-text">{val:.4f}</span></div></td>'
        rows_html += f"<tr>{cells}</tr>"

    header_cells = "".join(f"<th>{c}</th>" for c in metrics_cols)

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8"><title>检索评估报告</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 40px; }}
h1 {{ color: #60a5fa; text-align: center; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th, td {{ padding: 12px 16px; border: 1px solid #334155; text-align: left; }}
th {{ background: #1e293b; color: #94a3b8; font-weight: 600; }}
tr:hover {{ background: #1e293b; }}
.bar-container {{ position: relative; width: 100%; height: 24px; background: #1e293b; border-radius: 4px; overflow: hidden; }}
.bar {{ height: 100%; border-radius: 4px; transition: width 0.5s; }}
.bar-text {{ position: absolute; right: 8px; top: 2px; font-size: 12px; color: #e2e8f0; font-weight: 600; }}
</style></head><body>
<h1>检索模型对比报告 (Top-{df.index.name or 'K'})</h1>
<table><tr><th>Method</th>{header_cells}</tr>{rows_html}</table>
<p style="text-align:center;color:#64748b;font-size:13px;">Generated by SmartOps Eval</p>
</body></html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML报告: {html_path}")


def load_test_data(path: str) -> List[Dict]:
    if not os.path.exists(path):
        logger.error(f"测试文件不存在: {path}")
        logger.info("请先运行 export_chunks.py 生成标注文件，或手动创建 test_queries.json")
        logger.info("格式: [{\"query\": \"...\", \"relevant_ids\": [\"doc_id_1\", \"doc_id_2\"]}]")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"加载 {len(data)} 条测试数据")
    return data


if __name__ == "__main__":
    test_data_path = os.path.join(os.path.dirname(__file__), "test_queries.json")
    output_dir = os.path.join(os.path.dirname(__file__), "reports")

    test_data = load_test_data(test_data_path)

    from src.retriever import OpsRetriever
    pdf_path = str(_BASE_DIR / "data" / "文档2.pdf") if (_BASE_DIR / "data" / "文档2.pdf").exists() else None
    retriever = OpsRetriever(pdf_path=pdf_path)

    evaluator = RetrievalEvaluator(top_k=3)
    df, detail_df = evaluator.run(test_data, retriever)

    print_report(df, f"检索模型对比报告 (Top-3)")
    save_report(df, detail_df, output_dir)
    generate_html_report(df, output_dir)

    kg_available = hasattr(retriever, "kg") and retriever.kg and retriever.kg.is_available
    if kg_available:
        logger.info("检测到知识图谱，运行含知识图谱的评估...")
        df_kg = evaluator.run_with_kg(test_data, retriever)
        print_report(df_kg, "含知识图谱的检索对比报告")
