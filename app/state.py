"""
Shared graph state. Every node reads/writes a subset of this.
Nothing here is persisted between API calls — the graph is
instantiated fresh per /chat request (stateless by design).
"""
from __future__ import annotations
from typing import Literal, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # ---- input ----
    messages: list[dict]           # raw [{"role": "...", "content": "..."}]

    # ---- supervisor output ----
    in_scope: bool
    refusal_category: Optional[Literal["off_topic", "legal_advice", "hiring_advice", "prompt_injection"]]
    mode: Optional[Literal["clarify", "recommend", "compare", "refine"]]
    stance: Optional[Literal["soft_challenge", "hard_directive"]]  # only set when mode == "refine"
    constraints: dict              # re-derived fresh from full history every turn — never patched
    compare_entities: list[str]    # raw names user used, pre-alias-resolution

    # ---- retrieval output ----
    candidates: list[dict]         # catalog records only — nothing outside this list may reach the reply

    # ---- final output ----
    reply: str
    recommendations: list[dict]
    end_of_conversation: bool
