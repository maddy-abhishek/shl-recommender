from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import json
from openai import OpenAI

_client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

_MODEL = os.environ.get(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


def call_structured(
    system_prompt: str,
    messages: list[dict],
    tool_name: str,
    tool_schema: dict,
    max_tokens: int = 1024,
) -> dict:

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Emit {tool_name} output.",
                    "parameters": tool_schema,
                },
            }
        ],
        tool_choice={
            "type": "function",
            "function": {"name": tool_name},
        },
        max_tokens=max_tokens,
    )

    tool_calls = response.choices[0].message.tool_calls

    if not tool_calls:
        raise RuntimeError("Model did not call tool.")

    return json.loads(tool_calls[0].function.arguments)


def call_text(system_prompt, messages, max_tokens=512):

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()
