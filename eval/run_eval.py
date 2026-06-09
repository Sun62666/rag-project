import json
import os
import sys
import logging
from typing import List, Dict, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent.parent


def build_ragas_dataset(
    test_queries_file: str,
    retriever,
    llm,
    max_samples: Optional[int] = None,
    top_k: int = 3,
) -> dict:
    from langchain_core.documents import Document

    logger.info("加载测试集...")
    with open(test_queries_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]

    questions = []
    ground_truths = []
    contexts_list = []
    answers = []

    for i, item in enumerate(test_data, 1):
        query = item["query"]
        logger.info(f"[{i}/{len(test_data)}] 处理: {query}")

        try:
            docs = retriever.get_ensemble_rerank_docs(query, top_k=top_k)

            if not docs:
                logger.warning(f"  未检索到文档，跳过")
                continue

            contexts = [doc.page_content for doc in docs]

            kg_context = ""
            if hasattr(retriever, "kg") and retriever.kg and retriever.kg.is_available:
                try:
                    kg_context = retriever.kg.format_graph_context(query, depth=2)
                except Exception:
                    pass

            full_context = contexts[:]
            if kg_context:
                full_context.insert(0, f"[知识图谱] {kg_context}")

            prompt = f"""你是一个运维专家，请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请说明"根据现有资料无法回答"。

参考资料：
{chr(10).join([f"[{j + 1}] {ctx}" for j, ctx in enumerate(full_context)])}

用户问题：{query}

请给出简洁、准确的回答（100字以内）："""

            response = llm.invoke(prompt)
            answer = response.content.strip()

            questions.append(query)
            contexts_list.append(contexts)
            answers.append(answer)
            ground_truths.append(item.get("ground_truth", ""))

            logger.info(f"  完成")

        except Exception as e:
            logger.warning(f"  失败: {e}")
            continue

    logger.info(f"成功处理 {len(questions)} 条数据")

    data = {
        "question": questions,
        "ground_truth": ground_truths,
        "contexts": contexts_list,
        "answer": answers,
    }

    return data, len(questions)


def run_ragas_eval(dataset_dict: dict, llm, embeddings, output_dir: str):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision

    dataset = Dataset.from_dict(dataset_dict)

    has_ground_truth = any(gt.strip() for gt in dataset_dict["ground_truth"])

    if has_ground_truth:
        metrics = [faithfulness, answer_relevancy, context_precision]
        logger.info("检测到 ground_truth，评估: faithfulness + answer_relevancy + context_precision")
    else:
        metrics = [faithfulness, answer_relevancy]
        logger.info("未检测到 ground_truth，仅评估: faithfulness + answer_relevancy")

    logger.info("开始 RAGAS 评估...")

    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )
        df = result.to_pandas()
    except Exception as e:
        logger.error(f"RAGAS 评估失败: {e}")
        logger.info("尝试仅评估 faithfulness...")
        try:
            result = evaluate(
                dataset,
                metrics=[faithfulness],
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=False,
            )
            df = result.to_pandas()
        except Exception as e2:
            logger.error(f"评估再次失败: {e2}")
            return None

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "ragas_report.csv")
    df.to_csv(report_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("  RAGAS 端到端评估报告")
    print("=" * 60)

    numeric_cols = [c for c in df.columns if c in ["faithfulness", "answer_relevancy", "context_precision"]]
    if numeric_cols:
        print(f"\n平均得分:")
        print(df[numeric_cols].mean().round(4).to_string())

    print(f"\n详细结果:")
    print(df.to_string())
    print(f"\n报告已保存至: {report_path}")

    return df


def auto_generate_ground_truth(
    test_queries_file: str,
    retriever,
    llm,
    output_file: str,
    top_k: int = 3,
):
    logger.info("自动生成 ground_truth...")

    with open(test_queries_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    for item in test_data:
        if item.get("ground_truth", "").strip():
            continue

        query = item["query"]
        try:
            docs = retriever.get_ensemble_rerank_docs(query, top_k=top_k)
            contexts = [doc.page_content for doc in docs]

            prompt = f"""根据以下参考资料，用一句话回答问题。如果资料中没有相关信息，回答"无法回答"。

参考资料：
{chr(10).join([f"[{j+1}] {ctx}" for j, ctx in enumerate(contexts)])}

问题：{query}

参考答案："""
            response = llm.invoke(prompt)
            item["ground_truth"] = response.content.strip()
        except Exception as e:
            logger.warning(f"生成 ground_truth 失败: {e}")
            item["ground_truth"] = ""

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    logger.info(f"ground_truth 已保存至: {output_file}")


if __name__ == "__main__":
    from src.core.config import get_settings
    from src.retriever import OpsRetriever
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import DashScopeEmbeddings

    cfg = get_settings()
    llm = ChatOpenAI(model=cfg.LLM_MODEL, base_url=cfg.BASE_URL, api_key=cfg.DASHSCOPE_API_KEY, temperature=0.1)
    emb = DashScopeEmbeddings(model=cfg.EMBED_MODEL, dashscope_api_key=cfg.DASHSCOPE_API_KEY)

    eval_dir = os.path.dirname(__file__)
    test_queries_file = os.path.join(eval_dir, "test_queries.json")
    output_dir = os.path.join(eval_dir, "reports")

    if not os.path.exists(test_queries_file):
        logger.error(f"找不到测试文件: {test_queries_file}")
        sys.exit(1)

    pdf_path = str(_BASE_DIR / "data" / "文档2.pdf") if (_BASE_DIR / "data" / "文档2.pdf").exists() else None
    retriever = OpsRetriever(pdf_path=pdf_path)

    has_ground_truth = False
    with open(test_queries_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    has_ground_truth = any(item.get("ground_truth", "").strip() for item in test_data)

    if not has_ground_truth:
        logger.info("未检测到 ground_truth，正在自动生成...")
        gt_file = os.path.join(eval_dir, "test_queries_with_gt.json")
        auto_generate_ground_truth(test_queries_file, retriever, llm, gt_file)
        test_queries_file = gt_file

    print("=" * 60)
    print("  RAGAS 端到端评估")
    print("=" * 60)

    dataset_dict, num_samples = build_ragas_dataset(
        test_queries_file=test_queries_file,
        retriever=retriever,
        llm=llm,
        max_samples=5,
    )

    if num_samples == 0:
        logger.error("没有成功处理任何数据，退出")
        sys.exit(1)

    run_ragas_eval(dataset_dict, llm, emb, output_dir)
