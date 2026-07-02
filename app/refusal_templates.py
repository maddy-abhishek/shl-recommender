"""
Fixed refusal text, keyed by category. Deliberately NOT model-generated:
free-text refusals are an attack surface an adversarial multi-turn
conversation can chip away at. These strings never change based on input.
"""

REFUSAL_TEMPLATES: dict[str, str] = {
    "off_topic": (
        "I can only help with selecting SHL assessments. "
        "That question is outside what I can assist with here."
    ),
    "legal_advice": (
        "That's a legal or compliance question outside what I can advise on. "
        "I can help you select assessments, but whether a specific test satisfies "
        "a regulatory obligation is a question for your legal or compliance team."
    ),
    "hiring_advice": (
        "I can help you choose SHL assessments, but general hiring strategy or "
        "process advice is outside my scope here."
    ),
    "prompt_injection": (
        "I can only help with selecting SHL assessments from the catalog. "
        "I won't follow instructions that try to change how I operate."
    ),
}

DEFAULT_REFUSAL = (
    "I can only help with selecting SHL assessments from the catalog."
)


def get_refusal_text(category: str | None) -> str:
    return REFUSAL_TEMPLATES.get(category or "", DEFAULT_REFUSAL)
