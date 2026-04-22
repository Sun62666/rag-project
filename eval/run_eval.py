import json
import os
import sys
import pandas as pd
from ragas import experiment
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from src.config import Config
from src.retriever import OpsRetriever
from dotenv import load_dotenv

load_dotenv("../Key.env")
cfg = Config()
llm = ChatOpenAI(model=cfg.LLM_MODEL, base_url=cfg.BASE_URL, api_key=cfg.DASHSCOPE_API_KEY, temperature=0.1)
emb = DashScopeEmbeddings(model=cfg.EMBED_MODEL, dashscope_api_key=cfg.DASHSCOPE_API_KEY)


def build_ragas_dataset(test_queries_file, pdf_path, max_samples=None):
    print("📂 加载测试集...")
    with open(test_queries_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]

    print(f" 初始化检索器（首次加载较慢）...")
    retriever = OpsRetriever(pdf_path=pdf_path)

    questions = []
    ground_truths = []
    contexts_list = []
    answers = []

    for i, item in enumerate(test_data, 1):
        query = item["query"]
        print(f"\n[{i}/{len(test_data)}] 处理: {query}")

        try:
            docs = retriever.get_ensemble_rerank_docs(query, top_k=3)

            if not docs:
                print(f"  ⚠️ 未检索到文档，跳过")
                continue

            contexts = [doc.page_content for doc in docs]

            prompt = f"""你是一个运维专家，请根据以下参考资料回答用户的问题。
                    如果参考资料中没有相关信息，请说明"根据现有资料无法回答"。

                    参考资料：
                    {chr(10).join([f"[{j + 1}] {ctx}" for j, ctx in enumerate(contexts)])}

                    用户问题：{query}

                    请给出简洁、准确的回答（100字以内）："""
            # chr(10)代表换行
            response = llm.invoke(prompt)
            answer = response.content.strip()

            questions.append(query)
            contexts_list.append(contexts)
            answers.append(answer)
            ground_truths.append("")

            print(f"  ✅ 完成")

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            continue

    print(f"\n📊 成功处理 {len(questions)} 条数据")

    data = {
        "question": questions,
        "ground_truth": ground_truths,
        "contexts": contexts_list,
        "answer": answers
    }

    return Dataset.from_dict(data), len(questions)


if __name__ == "__main__":
    eval_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(eval_dir)
    # from pathlib import Path
    # parent_dir = Path(__file__).parent
    # root_dir = parent_dir.parent
    # test_queries_file = str(parent_dir / "data" / "test_queries.json")
    # pdf_path = str(root_dir / "data" / "文档2.pdf")

    test_queries_file = os.path.join(eval_dir, "test_queries.json")
    pdf_path = os.path.join(root_dir, "data", "文档2.pdf")

    if not os.path.exists(test_queries_file):
        print(f"❌ 找不到测试文件: {test_queries_file}")
        sys.exit(1)

    if not os.path.exists(pdf_path):
        print(f"❌ 找不到PDF文件: {pdf_path}")
        sys.exit(1)

    print("=" * 60)
    print("🚀 RAGAS 端到端评估")
    print("=" * 60)

    dataset, num_samples = build_ragas_dataset(
        test_queries_file=test_queries_file,
        pdf_path=pdf_path,
        max_samples=5
    )

    if num_samples == 0:
        print("❌ 没有成功处理任何数据，退出")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("📈 开始RAGAS评估...")
    print("=" * 60)

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
            embeddings=emb,
            raise_exceptions=False
        )
        # 1️⃣ Faithfulness（忠实度）- 答案是否基于资料？
        # 2️⃣ Context Precision（上下文精确度）- 检索的资料有用吗？
        # 3️⃣ Answer Relevancy（答案相关性）- 答案是否回答问题？
        # 4️⃣ Context Recall（上下文召回率）- 资料是否包含答案信息？
        # faithfulness（忠实度）：答案是否严格基于检索到的上下文，有没有幻觉。
        # answer_relevancy（答案相关性）：答案是否直接回答了用户问题。
        # context_precision（上下文精确度）：检索到的上下文中，有多少是与答案真正相关的（需要ground_truth作为参考）。
        # 计算 faithfulness
        df = result.to_pandas()

        report_path = os.path.join(eval_dir, "ragas_report.csv")
        df.to_csv(report_path, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 60)
        print("✅ 评估完成！")
        print("=" * 60)
        print(f"\n📊 平均得分:")
        print(df[["faithfulness", "answer_relevancy", "context_precision"]].mean())
        print(f"\n💾 报告已保存至: {report_path}")
        print(f"\n📄 详细结果:")
        print(df.to_string())

    except Exception as e:
        print(f"\n❌ RAGAS评估失败: {e}")
        print("💡 提示: 可能是因为ground_truth为空，尝试只评估 faithfulness 和 answer_relevancy")

        try:
            result = evaluate(
                dataset,
                metrics=[faithfulness, context_precision],
                llm=llm,
                embeddings=emb,
                raise_exceptions=False
            )
            df = result.to_pandas()
            report_path = os.path.join(eval_dir, "ragas_report.csv")
            df.to_csv(report_path, index=False, encoding="utf-8-sig")
            print(f"\n✅ 部分评估完成，已保存至: {report_path}")
            print(df[["faithfulness", "answer_relevancy"]].mean())
        except Exception as e2:
            print(f"❌ 再次失败: {e2}")