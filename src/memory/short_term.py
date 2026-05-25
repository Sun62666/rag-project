import logging
from collections import defaultdict
from typing import List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """短期记忆管理器：基于内存的会话消息历史，支持多轮对话上下文连贯、会话隔离"""

    def __init__(self, max_history: int = 20):
        self._store: Dict[str, List[BaseMessage]] = defaultdict(list)
        self._max_history = max_history

    def add_message(self, session_id: str, role: str, content: str):
        if role == "user":
            msg = HumanMessage(content=content)
        elif role == "assistant":
            msg = AIMessage(content=content)
        elif role == "system":
            msg = SystemMessage(content=content)
        else:
            msg = HumanMessage(content=content)
        self._store[session_id].append(msg)
        if len(self._store[session_id]) > self._max_history:
            self._store[session_id] = self._store[session_id][-self._max_history:]
        logger.info(f"[短期记忆] session={session_id} 添加 {role} 消息,当前共 {len(self._store[session_id])} 条")


    def add_user_message(self, session_id: str, content: str):
        self.add_message(session_id, "user", content)

    def add_ai_message(self, session_id: str, content: str):
        self.add_message(session_id, "assistant", content)

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        return list(self._store.get(session_id, []))

    def get_history_strs(self, session_id: str, last_n: int = 6) -> List[str]:
        msgs = self._store.get(session_id, [])
        if last_n > 0:
            msgs = msgs[-last_n:]
        result = []
        for m in msgs:
            if isinstance(m, HumanMessage):
                result.append(f"用户: {m.content}")
            elif isinstance(m, AIMessage):
                result.append(f"助手: {m.content}")
            elif isinstance(m, SystemMessage):
                result.append(f"系统: {m.content}")
        return result

    def clear(self, session_id: str):
        if session_id in self._store:
            del self._store[session_id]
            logger.info(f"[短期记忆] session={session_id} 已清空")

    def load_from_records(self, session_id: str, records: List[Dict]):
        if not records:
            return
        self._store[session_id] = []
        for r in records:
            role = r.get("role", "user")
            content = r.get("content", "")
            self.add_message(session_id, role, content)
        logger.info(f"[短期记忆] session={session_id} 从记录加载 {len(records)} 条消息")

    def session_ids(self) -> List[str]:
        return list(self._store.keys())
