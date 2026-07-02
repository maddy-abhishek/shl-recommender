"""
Tier-1 deterministic guardrail. Cheap keyword check BEFORE the Supervisor
LLM call, so obviously-adversarial input never costs a model call.

Scope is deliberately narrow: this catches literal prompt-injection
phrasing only. It does NOT and CANNOT reliably catch off-topic, legal-advice,
or hiring-advice requests — those are semantic judgments with no keyword
signature. Supervisor's own in_scope field is tier 2 and is what actually
covers those three categories. Don't extend this list hoping it'll cover
more; it won't, and false positives here block legitimate catalog questions.

Checks only the LATEST user message, not the full history — if something
adversarial slipped through on an earlier turn, re-scanning old messages
buys nothing; only new input matters.
"""

_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore your instructions",
    "ignore all previous",
    "disregard your rules",
    "disregard previous",
    "you are now",
    "developer mode",
    "system prompt",
    "reveal your prompt",
    "reveal your instructions",
    "act as if you have no restrictions",
    "pretend you are",
    "jailbreak",
]


def deterministic_check(latest_user_message: str) -> str | None:
    """Returns 'prompt_injection' if a literal pattern matches, else None."""
    text = latest_user_message.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in text:
            return "prompt_injection"
    return None
