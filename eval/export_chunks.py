import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent.parent


def export_chunks_from_retriever(
    pdf_path: Optional[str] = None,
    output_file: Optional[str] = None,
    include_preview_length: int = 200,
) -> List[Dict]:
    from src.retriever import OpsRetriever

    if not pdf_path:
        pdf_path = str(_BASE_DIR / "data" / "文档2.pdf")

    if not Path(pdf_path).exists():
        logger.error(f"PDF 文件不存在: {pdf_path}")
        logger.info("支持的数据源: PDF文件路径")
        return []

    retriever = OpsRetriever(pdf_path=pdf_path)

    if not hasattr(retriever, "splits") or not retriever.splits:
        logger.error("检索器切片为空，无法导出")
        return []

    chunks = []
    for i, doc in enumerate(retriever.splits):
        chunk = {
            "doc_id": doc.metadata.get("doc_id", f"chunk_{i}"),
            "source": doc.metadata.get("source", ""),
            "page": doc.metadata.get("page", ""),
            "preview": doc.page_content[:include_preview_length],
            "full_content": doc.page_content,
            "char_count": len(doc.page_content),
        }
        chunks.append(chunk)

    if not output_file:
        output_file = str(Path(__file__).parent / "all_chunks_for_label.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"已导出 {len(chunks)} 个 chunk 至 {output_file}")
    logger.info(f"总字符数: {sum(c['char_count'] for c in chunks)}")
    logger.info(f"平均字符数: {sum(c['char_count'] for c in chunks) / len(chunks):.0f}")

    return chunks


def generate_test_queries_template(
    chunks_file: str,
    output_file: Optional[str] = None,
    num_queries: int = 10,
    use_llm: bool = False,
    llm=None,
) -> None:
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not output_file:
        output_file = str(Path(__file__).parent / "test_queries.json")

    if use_llm and llm:
        logger.info("使用 LLM 自动生成测试查询...")
        test_queries = []
        for chunk in chunks[:num_queries]:
            prompt = f"""根据以下文档片段，生成一个运维相关的查询问题。
要求：问题应该能通过这段文档内容回答。

文档片段：
{chunk['preview']}

生成的问题："""
            try:
                response = llm.invoke(prompt)
                query = response.content.strip()
                test_queries.append({
                    "query": query,
                    "relevant_ids": [chunk["doc_id"]],
                    "ground_truth": "",
                })
            except Exception as e:
                logger.warning(f"LLM 生成失败: {e}")
        if test_queries:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(test_queries, f, ensure_ascii=False, indent=2)
            logger.info(f"已生成 {len(test_queries)} 条测试查询至 {output_file}")
            return

    logger.info("生成空模板，请手动标注...")
    template = []
    for i in range(min(num_queries, len(chunks))):
        template.append({
            "query": f"请输入第{i+1}个测试查询",
            "relevant_ids": [chunks[i]["doc_id"]],
            "ground_truth": "",
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    logger.info(f"已生成 {len(template)} 条模板至 {output_file}")
    logger.info("请手动编辑该文件，填写 query 和 ground_truth 字段")


def print_labeling_guide():
    guide = """
╔══════════════════════════════════════════════════════════════════╗
║                    检索评估标注指南                              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. 运行 export_chunks.py 导出所有文档切片                       ║
║  2. 查看每个 chunk 的 preview 和 doc_id                         ║
║  3. 为每个测试查询标注:                                          ║
║     - query: 运维相关的查询问题                                   ║
║     - relevant_ids: 能回答该问题的文档 doc_id 列表               ║
║     - ground_truth: (可选) 标准答案                              ║
║                                                                  ║
║  test_queries.json 格式:                                         ║
║  [                                                               ║
║    {                                                             ║
║      "query": "Redis 内存溢出如何排查？",                        ║
║      "relevant_ids": ["a1b2c3d4e5f6", "f6e5d4c3b2a1"],          ║
║      "ground_truth": "检查 maxmemory 配置..."                   ║
║    }                                                             ║
║  ]                                                               ║
║                                                                  ║
║  4. 运行 eval_retrievers.py 进行检索评估                         ║
║  5. 运行 run_eval.py 进行端到端 RAGAS 评估                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(guide)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--guide":
        print_labeling_guide()
        sys.exit(0)

    chunks = export_chunks_from_retriever()

    if chunks:
        chunks_file = str(Path(__file__).parent / "all_chunks_for_label.json")
        generate_test_queries_template(chunks_file, num_queries=10)
        print_labeling_guide()
