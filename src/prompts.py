import os
from langchain_core.prompts import ChatPromptTemplate

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


def _load_prompt_file(filename: str) -> str:
    filepath = os.path.join(_PROMPTS_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_system_prompt() -> str:
    return _load_prompt_file("ops_system.md")


def load_agent_prompt() -> str:
    return _load_prompt_file("agent_system.md")


def load_classify_prompt() -> str:
    return _load_prompt_file("classify.md")


def load_rewrite_prompt() -> str:
    return _load_prompt_file("rewrite.md")


def load_tool_call_prompt() -> str:
    return _load_prompt_file("tool_call.md")


def load_kg_extraction_prompt() -> str:
    return _load_prompt_file("kg_extraction.md")


def get_classify_prompt() -> ChatPromptTemplate:
    system_text = load_classify_prompt()
    return ChatPromptTemplate.from_messages([
        ('system', system_text),
        ('human', '用户问题: {query}\n\n最近对话历史:\n{chat_history}')
    ])


def get_rewrite_prompt() -> ChatPromptTemplate:
    system_text = load_rewrite_prompt()
    return ChatPromptTemplate.from_messages([
        ('system', system_text),
        ('human', '{query}')
    ])


def get_generate_prompt() -> ChatPromptTemplate:
    sys_prompt = load_system_prompt()
    return ChatPromptTemplate.from_messages([
        ('system', sys_prompt + "\n\n特别注意：当用户问题是对之前对话的追问或引用时，必须基于历史对话内容回答，不要输出兜底文案。"),
        ('human', '用户问题: {query}\n\n可用上下文:\n{context}\n\n历史对话:\n{chat_history}')
    ])


def get_tool_call_prompt() -> ChatPromptTemplate:
    system_text = load_tool_call_prompt()
    return ChatPromptTemplate.from_messages([
        ('system', system_text),
        ('human', '用户问题: {query}')
    ])
