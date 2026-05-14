from langchain_core.prompts import ChatPromptTemplate
import os


def load_system_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"prompts", "ops_system.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_classify_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ('system', """你是运维问题分类器。根据用户问题判断意图类别，只输出类别名称：

        - fault: 需要查阅知识库的故障排查/配置方法/运维规范问题（如"CPU过高怎么办"、"Redis内存溢出"、"Nginx配置负载均衡"）
        - system: 需要实时检查服务器状态的问题（如"查看CPU状态"、"检查8080端口"、"查看nginx日志"）
        - mixed: 既需要知识库又需要实时检查的混合问题（如"服务器CPU高怎么排查"）
        - followup: 基于历史对话的追问、澄清、引用之前话题的问题（如"上次问了什么"、"刚才那个问题再说一下"、"具体怎么操作"、"还有其他方法吗"、"帮我详细解释一下"）
        - reject: 与运维完全无关的闲聊、娱乐、无关话题（如"讲个笑话"、"今天天气怎么样"）

        只输出一个词：fault / system / mixed / followup / reject"""),
        ('human', '用户问题: {query}\n\n最近对话历史:\n{chat_history}')
    ])


def get_rewrite_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ('system', """你是运维检索专家。将用户问题改写为更适合知识库检索的关键词。

        要求：
        1. 提取核心技术名词和故障关键词
        2. 补充同义词和专业术语（如"内存满"→"OOM out-of-memory"）
        3. 输出2-5个检索关键词，用空格分隔
        4. 只输出改写后的关键词，不要解释

        示例：
        - "Redis内存满了怎么办" → "Redis OOM 内存溢出 maxmemory 淘汰策略"
        - "服务器CPU使用率100%" → "CPU使用率过高 CPU满载 进程占用 top"
        - "MySQL连接超时" → "MySQL连接超时 connection_timeout wait_timeout" """),
        ('human', '{query}')
    ])


def get_generate_prompt() -> ChatPromptTemplate:
    sys_prompt = load_system_prompt()
    return ChatPromptTemplate.from_messages([
        ('system', sys_prompt + "\n\n特别注意：当用户问题是对之前对话的追问或引用时，必须基于历史对话内容回答，不要输出兜底文案。"),
        ('human', '用户问题: {query}\n\n可用上下文:\n{context}\n\n历史对话:\n{chat_history}')
    ])


def get_tool_call_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ('system', '你是运维工具调用助手。根据用户问题调用合适的系统工具获取实时数据，只调用与问题相关的工具。'),
        ('human', '用户问题: {query}')
    ])
