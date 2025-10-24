# app/llm_client.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, time
from typing import Any, Dict, Optional
from fastapi import HTTPException
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError, BadRequestError
from pathlib import Path
import re

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_RETRIES = 2  # Maximum number of retry attempts per request

# ---------------------------------------- LLM client helpers ---------------------------------------- #
def _get_client() -> OpenAI:
    api_key  = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None  # Recommended to include /v1
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY missing. Set it in .env.")
    # Configure HTTPS_PROXY/HTTP_PROXY in the environment when a proxy is required; httpx will read it automatically.
    return OpenAI(api_key=api_key, base_url=base_url)

def _extract_first_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from arbitrary text by scanning for balanced braces."""
    # First try to parse the body as-is
    try:
        return json.loads(text)
    except Exception:
        pass
    # Otherwise match braces manually to extract the first JSON object
    s = text
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found.")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    frag = s[start:i+1]
                    return json.loads(frag)
    raise ValueError("No complete JSON object found.")

# ---------------------------------------- Logging helpers ---------------------------------------- #
def _maybe_log(debug: Dict[str, Any]) -> None:
    """Persist raw request/response debug data when LOG_LLM is enabled."""
    try:
        if os.getenv("LOG_LLM", "").strip() not in ("1", "true", "yes", "on"):
            return
        Path("logs").mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = Path("logs") / f"llm_{ts}.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)
    except Exception:
        # Logging failures should not impact primary flow
        pass
def _normalize_dataurl(image_data: str, image_mime: str = "image/png") -> str:
    """Return a data URL whether the input is bare base64 or already prefixed."""
    if not image_data:
        return ""
    if image_data.startswith("data:"):
        return image_data  # Already a data URL
    # Otherwise treat it as bare base64
    return f"data:{image_mime};base64,{image_data}"
def _inject_vision_content(messages: list[dict]) -> list[dict]:
    """Convert _image_data/_image_mime placeholders into OpenAI-style multimodal message payloads."""
    out = []
    for m in messages:
        if m.get("role") != "user":
            out.append(m)
            continue
        content = m.get("content")
        if not isinstance(content, str) or "_image_data" not in content:
            out.append(m)
            continue

        # The content is an f-string that resembles a dict; attempt a lightweight parse
        # Ideally prompting would pass a dict directly and avoid string parsing altogether
        # Use a safe parsing strategy instead of eval()
        import json
        try:
            # Translate single quotes to double quotes first
            content_json = json.loads(re.sub(r"'", '"', content))
        except Exception:
            out.append(m); continue

        image_data = content_json.pop("_image_data", None)
        image_mime = content_json.pop("_image_mime", "image/png")
        if not image_data:
            out.append({"role": "user", "content": str(content_json)})
            continue

        dataurl = _normalize_dataurl(image_data, image_mime)
        mm_content = [
            {"type": "text", "text": json.dumps(content_json, ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": dataurl}},
        ]
        out.append({"role": "user", "content": mm_content})
    return out

def call_chat_completions(
    messages: list[dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.4,
    top_p: float = 0.95,
    max_tokens: int = 1024,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Call /v1/chat/completions with JSON-first handling and retries. Returns (parsed_payload, debug_info)."""
    # Convert to multimodal content when _image_data is present
    needs_mm = any(isinstance(m.get("content"), str) and "_image_data" in m.get("content", "") for m in messages)
    if needs_mm:
        messages = _inject_vision_content(messages)

    client = _get_client()  # Build a fresh client per call to avoid import-time issues

    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model or DEFAULT_MODEL,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            parsed = json.loads(content)
            debug = {
                "raw_text": content,
                "response_dump": resp.model_dump(exclude_none=True),
                "mode": "json_object",
                "model": model or DEFAULT_MODEL,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                # Log only the head/tail of messages to keep debug output small
                "messages_head": messages[:3],
                "messages_tail": messages[-3:] if len(messages) > 3 else [],
            }
            _maybe_log(debug)
            return parsed, debug
        except BadRequestError:
            # Downgrade to plain text when response_format is rejected
            try:
                resp2 = client.chat.completions.create(
                    model=model or DEFAULT_MODEL,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                content2 = resp2.choices[0].message.content or "{}"
                parsed2 = _extract_first_json(content2)
                debug = {
                    "raw_text": content2,
                    "response_dump": resp2.model_dump(exclude_none=True),
                    "mode": "fallback_extract",
                    "model": model or DEFAULT_MODEL,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "messages_head": messages[:3],
                    "messages_tail": messages[-3:] if len(messages) > 3 else [],
                }
                _maybe_log(debug)
                return parsed2, debug
            except Exception as e2:
                last_err = e2
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(0.8 * (2 ** attempt))
                continue
        except Exception as e:
            last_err = e
            break

    raise HTTPException(502, f"LLM upstream failed: {last_err}")
