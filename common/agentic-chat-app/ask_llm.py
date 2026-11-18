import logging
import time

from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError

import llm_usage  # beside this file in /app; byte-identical copy of shared/llm_usage.py

logger = logging.getLogger(__name__)

# Retryable OpenAI error types
_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError)

_MAX_RETRIES = 3
_BACKOFF_SECS = [2, 4, 8]


def get_llm_answer(client: OpenAI, context: str, question: str, model: str = "gpt-4o"):
    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]

    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        t0 = time.time()
        try:
            logger.info(
                "[LLM] attempt=%d/%d model=%s context_len=%d question_len=%d",
                attempt, _MAX_RETRIES, model, len(context), len(question),
            )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            elapsed = time.time() - t0
            usage = response.usage
            logger.info(
                "[LLM] ok attempt=%d elapsed=%.2fs prompt_tokens=%s "
                "completion_tokens=%s total_tokens=%s model=%s",
                attempt, elapsed,
                usage.prompt_tokens if usage else "?",
                usage.completion_tokens if usage else "?",
                usage.total_tokens if usage else "?",
                response.model,
            )
            llm_usage.record_or_warn(
                logger, "agentic-app", llm_usage.selected_chat_provider(), "chat", model,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
                usage_missing=usage is None,
            )
            return response.choices[0].message.content

        except _RETRYABLE as e:
            elapsed = time.time() - t0
            last_exc = e
            backoff = _BACKOFF_SECS[attempt - 1] if attempt <= len(_BACKOFF_SECS) else _BACKOFF_SECS[-1]
            logger.warning(
                "[LLM] retryable error attempt=%d/%d elapsed=%.2fs "
                "error=%s backoff=%ds",
                attempt, _MAX_RETRIES, elapsed, e, backoff,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(backoff)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("get_llm_answer: retry loop completed without result")
