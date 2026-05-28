# test_short_term_smoke.py
from src.memory.short_term import ShortTermMemory
from langchain_core.messages import HumanMessage

def test_short_term_memory():
    stm = ShortTermMemory(max_history=5)
    sid = "s1"
    stm.add_user_message(sid, "hello")
    stm.add_ai_message(sid, "hi there")
    print("BaseMessage list:", stm.get_messages(sid))
    print("As dicts:", stm.get_messages_as_dicts(sid))
    print("As strings:", stm.get_history_strs(sid))


def test_short_term_basic_behavior():
    stm = ShortTermMemory(max_history=5)
    sid = "s1"
    stm.add_user_message(sid, "hello")
    stm.add_ai_message(sid, "hi there")

    msgs = stm.get_messages(sid)
    assert len(msgs) == 2
    assert msgs[0].content == "hello"
    assert msgs[1].content == "hi there"

    dicts = stm.get_messages_as_dicts(sid)
    assert dicts[0]["role"] == "user"
    assert dicts[0]["content"] == "hello"

    strs = stm.get_history_strs(sid)
    assert any("用户: hello" in s for s in strs)