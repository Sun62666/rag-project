import json, os
from pathlib import Path
from src.retriever import OpsRetriever

# pdf_path = os.path.join(os.path.dirname(__file__), "data", "文档2.pdf")
parent_dir = Path(__file__).parent
gradnparent_dir = parent_dir.parent
pdf_path = gradnparent_dir / "data" / "文档2.pdf"

retriever = OpsRetriever(str(pdf_path))


# 提取 doc_id 与内容片段供人工/LLM 标注
chunks = [
    {"doc_id": d.metadata["doc_id"], "source": d.metadata.get("source",""), "preview": d.page_content[:150]}
    for d in retriever.splits
]

with open("all_chunks_for_label.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
print(f"✅ 已导出 {len(chunks)} 个 chunk，请基于此标注 test_queries.json")
