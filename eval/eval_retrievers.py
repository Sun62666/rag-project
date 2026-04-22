# eval_retrievers.py
import json
from typing import List, Dict
from collections import defaultdict
import pandas as pd
from src.retriever import OpsRetriever  # 确保路径正确
from pathlib import Path
parent_path = Path(__file__).parent
grandparent_path = parent_path.parent
PDF_PATH = str(grandparent_path / "data" / "文档2.pdf")

class RetrievalEvaluator:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def _calc_metrics(self, retrieved_ids: List[str], relevant_ids: List[str]) -> Dict[str, float]:
        retrieved_set, relevant_set = set(retrieved_ids), set(relevant_ids)
        tp = len(retrieved_set & relevant_set)

        recall = tp / len(relevant_set) if relevant_set else 0.0  # 召回率，在所有真实文档中找到相关文档
        precision = tp / self.top_k if self.top_k > 0 else 0.0  # 精确率，在所有文档中找到相关文档
        hit = 1.0 if tp > 0 else 0.0  #Hit Rate（命中率），在k个检索中，至少包含一个相关文档的概率（二值化指标）

        mrr = 0.0   # 均值倒数秩/平均倒数排名（Mean Reciprocal Rank），第一个相关文档在检索结果中排名的倒数，只取排名最靠前的那个（即好结果是否排在前面）
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_set:
                mrr = 1.0 / rank
                break
        return {"recall": recall, "precision": precision, "mrr": mrr, "hit_rate": hit}

    def run(self, test_data: List[Dict], pdf_path: str) -> pd.DataFrame:
        print("🚀 正在初始化检索器（仅首次加载较慢）...")
        retriever = OpsRetriever(pdf_path=pdf_path)

        retrievers_map = {
            "BM25_Only": retriever.get_bm25_docs,
            "Vector_Only": retriever.get_vector_docs,
            "Ensemble+Rerank": retriever.get_ensemble_rerank_docs
        }

        results = {}
        for name, func in retrievers_map.items():
            print(f"\n🔍 正在评估: {name} ...")
            agg = defaultdict(list)
            for i, item in enumerate(test_data, 1):
                docs = func(item["query"], top_k=self.top_k)
                # print(f"\n{name}返回文档: {docs}")
                retrieved_ids = [d.metadata.get("doc_id", "") for d in docs]
                # print(f"\n{name}返回的相关检索id: {retrieved_ids}")
                metrics = self._calc_metrics(retrieved_ids, item["relevant_ids"])
                # print(f"\n{name}返回的指标包括（recall、precision、mrr、hit_rate）: {metrics}")
                for k, v in metrics.items():
                    agg[k].append(v)
                    # print(f'\nk: {k},v: {v}')

                if i % 10 == 0: print(f"  进度: {i}/{len(test_data)}")
            # print(f"\nagg: {agg}")
            results[name] = {k: sum(v) / len(v) for k, v in agg.items()}
            # print(f"\n{name}: results: {results} ")
        df = pd.DataFrame(results).T.round(4)
        print(f"df: {df}")
        return df


if __name__ == "__main__":
    # 1. 加载测试集
    with open("test_queries.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)  # 将json格式解析为字典或列表

    # 2. 配置 PDF 路径（与 main.py 保持一致）
    # import os
    #
    # pdf_path = os.path.join(os.path.dirname(__file__), "data", "文档2.pdf")
    pdf_path = PDF_PATH

    # print(test_data)
    # 3. 执行评估
    evaluator = RetrievalEvaluator(top_k=3)
    report = evaluator.run(test_data, pdf_path)

    print("\n" + "=" * 50)
    print("📈 检索模型对比报告 (Top-3)")
    print("=" * 50)
    print(report)

    # 保存报告
    report.to_csv("retrieval_eval_report.csv", encoding="utf-8-sig")
    print("\n💾 报告已保存至 retrieval_eval_report.csv")
