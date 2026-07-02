from __future__ import annotations
import asyncio
import logging

from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse, HealthResponse, Recommendation
from app.graph import get_graph
from app.retrieval import warmup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-recommender")

app = FastAPI(title="SHL Assessment Recommender")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting retrieval warmup...")
    warmup()
    logger.info("Retrieval warmup completed.")

# Soft internal budget, below the evaluator's 30s hard timeout, so we can
# return a schema-valid fallback before the external connection is killed.
SOFT_TIMEOUT_SECONDS = 22

# Design target: 16 total messages (8 user + 8 assistant) based on trace
# evidence — see README for the ambiguity in the spec text this resolves.
# We do NOT hard-fail past this; we just force end_of_conversation and stop
# expanding the shortlist, since the spec's literal wording could mean 8
# total and failing outright on the stricter reading is worse than a soft cap.
SOFT_TURN_CAP = 16


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


def _fallback_response(reason: str) -> ChatResponse:
    return ChatResponse(
        reply="I'm having trouble processing that right now — could you try again?",
        recommendations=[],
        end_of_conversation=False,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    graph = get_graph()
    initial_state = {"messages": messages}

    try:
        final_state = await asyncio.wait_for(
            asyncio.to_thread(graph.invoke, initial_state),
            timeout=SOFT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("chat handler exceeded soft timeout, returning fallback")
        return _fallback_response("timeout")
    except Exception:
        logger.exception("chat handler failed")
        return _fallback_response("error")

    end_of_conversation = final_state.get("end_of_conversation", False)
    if len(messages) >= SOFT_TURN_CAP:
        end_of_conversation = True

    recommendations = [
        Recommendation(**r) for r in final_state.get("recommendations", [])
    ]

    return ChatResponse(
        reply=final_state.get("reply", ""),
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )
