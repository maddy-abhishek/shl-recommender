"""
Graph shape:

    supervisor
       |
       ├─ not in_scope ──────────────────► refuse ──► END
       ├─ mode == clarify ───────────────► generate ──► END   (no retrieval)
       └─ mode in {recommend,compare,refine} ─► retrieve ──► generate ──► END

No checkpointer: every API call builds a fresh graph invocation from the
full message history in the request. There is nothing to persist between
calls — persisting anything here would contradict the stateless spec.
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END

from app.state import AgentState
from app.nodes import supervisor_node, refuse_node, retrieve_node, generate_node


def _route_after_supervisor(state: AgentState) -> str:
    if not state.get("in_scope", True):
        return "refuse"
    if state.get("mode") == "clarify":
        return "generate"
    return "retrieve"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {"refuse": "refuse", "generate": "generate", "retrieve": "retrieve"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("refuse", END)
    graph.add_edge("generate", END)

    return graph.compile()  # no checkpointer — intentionally stateless


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
