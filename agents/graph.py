"""LangGraph: agent (LLM picks tools) <-> tools (code) -> finalize (code builds the JSON)."""

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agents.tools import RunContext, build_tools

log = logging.getLogger(__name__)

MAX_STEPS = 20  # LangGraph recursion limit: caps the agent <-> tools loop

SYSTEM_PROMPT = """You answer questions about data from APIs by calling tools.
Steps: discover_server -> discover_tools -> get_tool_schema -> invoke_tool.
Rules:
- Put every filter from the request into invoke_tool's `where`, in one call.
- "top N", "most expensive", "cheapest" = sort_by + order + limit.
- "average", "sum of", "how many", "highest/lowest total" = aggregate (count needs no field).
- "no X", "missing X", "without X" = where X eq "" (an empty value means the field is missing).
- "over", "more than", "above" = gt. "at least" = gte. "under", "less than", "below" = lt. "at most" = lte.
- Use the field names and value formats that get_tool_schema describes.
- Only use fields and params that get_tool_schema lists. If a tool returns an error, fix the call and retry.
- Never write data yourself. After invoke_tool succeeds, reply only: done
- If the tools can't answer the request (other data, actions like delete or update,
  invalid input), don't call invoke_tool. Reply exactly: UNSUPPORTED: <one short reason>"""

DEFAULT_UNSUPPORTED = "I can't answer that with the available tools."


class State(TypedDict):
    messages: Annotated[list, add_messages]
    output: dict


def build_graph(ctx: RunContext):
    tools = build_tools(ctx)
    llm_with_tools = ctx.llm.bind_tools(tools)

    def agent(state: State):
        reply = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        for call in getattr(reply, "tool_calls", []) or []:
            log.info("agent -> %s(%s)", call["name"], call["args"])
        return {"messages": [reply]}

    def route(state: State):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "finalize"

    def finalize(state: State):
        if ctx.result is not None:
            return {"output": ctx.result}
        # No data fetched: the request was unsupported. Only the LLM's short
        # reason is passed on, never anything else it wrote.
        text = state["messages"][-1].content
        text = text.strip() if isinstance(text, str) else ""
        reason = text.split(":", 1)[1].strip()[:200] if text.startswith("UNSUPPORTED:") else ""
        log.warning("unsupported request: %s", reason or "no reason given")
        return {"output": {"error": reason or DEFAULT_UNSUPPORTED}}

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=(ValueError, KeyError)))
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", "finalize"])
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agent(query: str, llm, **ctx_kwargs) -> dict:
    ctx = RunContext(llm, **ctx_kwargs)
    graph = build_graph(ctx)
    state = graph.invoke({"messages": [HumanMessage(query)]}, {"recursion_limit": MAX_STEPS})
    if ctx.rejected:
        log.warning("%d record(s) rejected and left out of the output", len(ctx.rejected))
    return state["output"]
