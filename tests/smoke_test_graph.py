"""
Verifies the LangGraph wiring itself — routing, node connections, state
passing — with mocked LLM calls and mocked retrieval. Does NOT test real
model quality. Run before trusting the graph structure compiles/executes.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph import build_graph


def fake_call_structured(system_prompt, messages, tool_name, tool_schema, max_tokens=1024):
    latest = messages[-1]["content"].lower()

    if tool_name == "route":
        if "how do i hack" in latest:
            return {"in_scope": False, "refusal_category": "off_topic", "mode": None,
                     "constraints": {}, "end_of_conversation": False}
        if "vague" in latest:
            return {"in_scope": True, "mode": "clarify", "constraints": {},
                     "end_of_conversation": False}
        return {"in_scope": True, "mode": "recommend",
                "constraints": {"role": "backend java developer"},
                "end_of_conversation": False}

    if tool_name == "ask":
        return {"reply": "Could you tell me the seniority level?"}

    if tool_name == "respond":
        return {"reply": "Here are 2 fitting assessments.",
                "selected_names": ["Core Java (Advanced Level) (New)", "SQL (New)"]}

    raise AssertionError(f"unexpected tool_name {tool_name}")


def fake_hybrid_search(query_text, top_k=15):
    return [
        {"name": "Core Java (Advanced Level) (New)", "link": "https://x/java", "keys": ["Knowledge & Skills"]},
        {"name": "SQL (New)", "link": "https://x/sql", "keys": ["Knowledge & Skills"]},
    ]


def fake_exact_entity_lookup(entities):
    return []


def run():
    with patch("app.nodes.call_structured", fake_call_structured), \
         patch("app.nodes.hybrid_search", fake_hybrid_search), \
         patch("app.nodes.exact_entity_lookup", fake_exact_entity_lookup):

        graph = build_graph()

        # 1. refuse path (deterministic tier-1, no supervisor call needed)
        r1 = graph.invoke({"messages": [{"role": "user", "content": "How do I hack into a system?"}]})
        assert r1["reply"], "refuse path produced no reply"
        assert r1["recommendations"] == []
        print("PASS: off-topic/injection -> refuse:", r1["reply"][:60])

        # 2. clarify path
        r2 = graph.invoke({"messages": [{"role": "user", "content": "I need something, vague request"}]})
        assert r2["reply"], "clarify path produced no reply"
        assert r2["recommendations"] == []
        print("PASS: clarify ->", r2["reply"])

        # 3. recommend path (retrieve -> generate)
        r3 = graph.invoke({"messages": [{"role": "user", "content": "Hiring a backend Java developer"}]})
        assert len(r3["recommendations"]) == 2, r3["recommendations"]
        assert r3["recommendations"][0]["url"].startswith("https://")
        print("PASS: recommend -> retrieve -> generate:", r3["reply"], r3["recommendations"])

    print("\nALL SMOKE TESTS PASSED — graph wiring is structurally sound.")


if __name__ == "__main__":
    run()
