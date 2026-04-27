"""
LangGraph Orchestrator
Defines the Supervisor-Worker state machine graph.
"""

from typing import TypedDict, Annotated, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from schemas.models import GraphState, AgentState
from agents.supervisor import triage_supervisor, should_interrupt
from agents.search_agent import search_agent
from agents.analyst import analyst_agent


def _graph_state_to_dict(state: GraphState) -> dict:
    return state.model_dump()


def _dict_to_graph_state(d: dict) -> GraphState:
    return GraphState(**d)


# ---- Node functions (LangGraph expects dict state) ----

def triage_node(state: dict) -> dict:
    gs = GraphState(**state)
    gs = triage_supervisor(gs)
    return gs.model_dump()


def search_node(state: dict) -> dict:
    gs = GraphState(**state)
    gs = search_agent(gs)
    return gs.model_dump()


def analyst_node(state: dict) -> dict:
    gs = GraphState(**state)
    gs = analyst_agent(gs)
    return gs.model_dump()


def interrupt_node(state: dict) -> dict:
    """Interrupt node — just returns current state (user input needed)."""
    return state


def error_node(state: dict) -> dict:
    gs = GraphState(**state)
    gs.current_agent = AgentState.ERROR
    return gs.model_dump()


# ---- Conditional edges ----

def route_after_triage(state: dict) -> str:
    gs = GraphState(**state)
    return should_interrupt(gs)


def route_after_search(state: dict) -> str:
    gs = GraphState(**state)
    if gs.current_agent == AgentState.ERROR:
        return "error"
    if gs.current_agent == AgentState.ANALYZING:
        return "analyze"
    return "error"


def route_after_analysis(state: dict) -> str:
    gs = GraphState(**state)
    if gs.current_agent in (AgentState.COMPLETE, AgentState.VERIFYING):
        return "end"
    return "error"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph state machine."""

    builder = StateGraph(dict)

    # Add nodes
    builder.add_node("triage", triage_node)
    builder.add_node("interrupt", interrupt_node)
    builder.add_node("search", search_node)
    builder.add_node("analyze", analyst_node)
    builder.add_node("error", error_node)

    # Entry point
    builder.set_entry_point("triage")

    # Conditional routing after triage
    builder.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "interrupt": "interrupt",
            "search": "search",
            "error": "error",
        }
    )

    # After search
    builder.add_conditional_edges(
        "search",
        route_after_search,
        {
            "analyze": "analyze",
            "error": "error",
        }
    )

    # After analysis
    builder.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            "end": END,
            "error": "error",
        }
    )

    # Interrupt and error both end the graph turn
    builder.add_edge("interrupt", END)
    builder.add_edge("error", END)

    return builder.compile()


# Singleton graph instance
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
