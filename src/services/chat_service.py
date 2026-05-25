import json
import logging
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from src.core.config import get_settings
from src.core.redis import get_cache
from src.services.session_service import get_chat_history, save_chat_history

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "当前知识库未覆盖该问题，建议转交人工运维专家。"


async def handle_cached_answer(query: str, session_id: str, username: str, stm, bg_tasks) -> StreamingResponse:
    cached = get_cache().get(f"ops:{query}")
    if not cached:
        return None

    logger.info(f"answer from cache: True")
    bg_tasks.add_task(save_chat_history, session_id, query, cached, username)
    stm.add_user_message(session_id, query)
    stm.add_ai_message(session_id, cached)

    async def cached_stream():
        yield f"data: {json.dumps({'type': 'status', 'message': '从缓存中获取'})}\n\n"
        for char in cached:
            yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'from_cache': True})}\n\n"

    return StreamingResponse(
        cached_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


async def ask_agent(req, username: str, bg_tasks, agent, stm) -> StreamingResponse:
    chat_history = get_chat_history(req.session_id)
    if chat_history and not stm.get_messages(req.session_id):
        stm.load_from_records(req.session_id, chat_history)

    async def agent_stream():
        full_answer = []
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': '智能体分析中...'})}\n\n"

            async for event in agent.astream(req.query, req.session_id, username):
                if event["type"] == "status":
                    yield f"data: {json.dumps({'type': 'status', 'message': event['message']})}\n\n"
                elif event["type"] == "token":
                    full_answer.append(event["content"])
                    yield f"data: {json.dumps({'type': 'token', 'content': event['content']})}\n\n"
                elif event["type"] == "done":
                    answer_text = "".join(full_answer)
                    cfg = get_settings()
                    if answer_text and answer_text != FALLBACK_MESSAGE:
                        bg_tasks.add_task(get_cache().setex, f"ops:{req.query}", cfg.CACHE_TTL_SHORT, answer_text)
                    bg_tasks.add_task(save_chat_history, req.session_id, req.query, answer_text, username)
                    yield f"data: {json.dumps({'type': 'done', 'from_cache': False})}\n\n"

        except Exception as e:
            logger.error(f"agent_stream 异常: {e}")
            answer_text = "".join(full_answer) or f"智能体执行失败: {str(e)}"
            bg_tasks.add_task(save_chat_history, req.session_id, req.query, answer_text, username)
            yield f"data: {json.dumps({'type': 'status', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        agent_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


async def ask_graph(req, username: str, bg_tasks, graph, stm) -> StreamingResponse:
    chat_history = get_chat_history(req.session_id)
    history_strs = [f"{'用户' if h['role']=='user' else '助手'}: {h['content']}" for h in chat_history]

    async def stream_gen():
        try:
            state = {
                "query": req.query,
                "intent": "",
                "rewritten_query": "",
                "retrieved_context": "",
                "tool_results": {},
                "answer": "",
                "chat_history": history_strs
            }
            full_answer = []

            yield f"data: {json.dumps({'type': 'status', 'message': '正在分析问题...'})}\n\n"

            async for msg, metadata in graph.astream(state, stream_mode="messages"):
                node = metadata.get("langgraph_node")
                if node == "classify":
                    yield f"data: {json.dumps({'type': 'status', 'message': '正在分类意图...'})}\n\n"
                elif node == "rewrite_query":
                    yield f"data: {json.dumps({'type': 'status', 'message': '正在优化检索词...'})}\n\n"
                elif node == "retrieve":
                    yield f"data: {json.dumps({'type': 'status', 'message': '正在检索知识库...'})}\n\n"
                elif node == "execute_tools":
                    yield f"data: {json.dumps({'type': 'status', 'message': '正在执行系统工具...'})}\n\n"
                elif node == "generate" and isinstance(msg, AIMessageChunk) and msg.content:
                    token = msg.content
                    full_answer.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            answer_text = "".join(full_answer)

            if not answer_text.strip():
                logger.debug("流式未捕获答案，尝试直接获取...")
                final_state = await graph.ainvoke(state)
                answer_text = final_state.get("answer", "").strip()
                if answer_text:
                    for char in answer_text:
                        yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"

            # logger.info(f"answer_text: {answer_text}")

            cfg = get_settings()
            if answer_text and answer_text != FALLBACK_MESSAGE:
                bg_tasks.add_task(get_cache().setex, f"ops:{req.query}", cfg.CACHE_TTL_SHORT, answer_text)

            bg_tasks.add_task(save_chat_history, req.session_id, req.query, answer_text, username)

            yield f"data: {json.dumps({'type': 'done', 'from_cache': False})}\n\n"
        except Exception as e:
            logger.error(f"stream_gen 异常: {e}")
            yield f"data: {json.dumps({'type': 'status', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )
