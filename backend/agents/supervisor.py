"""
Triage Supervisor Agent
Pure LLM-based extraction using a simpler, more reliable prompt.
"""

import re
import json
import os
from groq import Groq
from schemas.models import AgentState, GraphState

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def _llm_extract(user_input: str, conversation_history: list) -> dict:
    client = Groq(api_key=GROQ_API_KEY)

    # Last 4 messages for context
    history_text = ""
    for msg in (conversation_history or [])[-4:]:
        r = msg.get("role", "")
        c = msg.get("content", "")
        if r and c:
            history_text += f"{r.upper()}: {c}\n"

    system_prompt = """Extract business review search intent from the user message. Return ONLY JSON, nothing else.

JSON format:
{"intent":"...","business_name":"...","location":"...","category":"...","vs_a_name":"...","vs_a_loc":"...","vs_b_name":"...","vs_b_loc":"..."}

intent values:
- single_business = user wants one specific business analysed (e.g. "Analyse HydePark Gym in Austin")
- top5 = user wants a list of best businesses (e.g. "Top 5 restaurants in Austin")  
- vs_comparison = user wants two businesses compared (e.g. "Starbucks Austin vs Starbucks NYC")
- needs_location = user gave business name but no city (e.g. "Starbucks", "Analyse McDonald's")
- needs_info = completely unclear

Rules:
- "analyse X in Y" or "look into X in Y" or "review X in Y" = single_business
- "X vs Y" or "X versus Y" = vs_comparison
- "top N" or "best" or "good" + category = top5
- business name with no city = needs_location
- Put null (not the string "null") for missing fields
- vs_a_name/vs_a_loc = first business in comparison, vs_b_name/vs_b_loc = second

Examples:
"Analyse HydePark Gym in Austin" -> {"intent":"single_business","business_name":"HydePark Gym","location":"Austin","category":"gym","vs_a_name":null,"vs_a_loc":null,"vs_b_name":null,"vs_b_loc":null}
"top 5 restaurants in Austin" -> {"intent":"top5","business_name":null,"location":"Austin","category":"restaurant","vs_a_name":null,"vs_a_loc":null,"vs_b_name":null,"vs_b_loc":null}
"Starbucks Austin vs Starbucks New York" -> {"intent":"vs_comparison","business_name":null,"location":null,"category":"coffee","vs_a_name":"Starbucks","vs_a_loc":"Austin","vs_b_name":"Starbucks","vs_b_loc":"New York"}
"McDonald's" -> {"intent":"needs_location","business_name":"McDonald's","location":null,"category":"fast food","vs_a_name":null,"vs_a_loc":null,"vs_b_name":null,"vs_b_loc":null}"""

    content = f"Conversation:\n{history_text}\nMessage: {user_input}" if history_text else f"Message: {user_input}"

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": content},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        print(f"[Supervisor] Raw LLM output: {raw}")

        # Strip markdown fences
        raw = re.sub(r"```json|```", "", raw).strip()
        # Extract just the JSON object
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)

        parsed = json.loads(raw)
        print(f"[Supervisor] Parsed: {parsed}")
        return parsed

    except Exception as e:
        print(f"[Supervisor] LLM error: {e}")
        # Last-ditch regex fallback so the app never gets stuck
        return _regex_fallback(user_input)


def _regex_fallback(text: str) -> dict:
    """Simple regex fallback when LLM fails — catches the most common patterns."""
    t = text.lower()
    result = {
        "intent": "needs_info",
        "business_name": None, "location": None, "category": None,
        "vs_a_name": None, "vs_a_loc": None,
        "vs_b_name": None, "vs_b_loc": None,
    }

    # VS comparison: "X in LOC vs Y in LOC" or "X LOC vs Y LOC"
    vs_match = re.search(r"(.+?)\s+(?:in\s+)?([a-z\s]+?)\s+vs\.?\s+(.+?)\s+(?:in\s+)?([a-z\s]+)$", t)
    if vs_match:
        result["intent"]    = "vs_comparison"
        result["vs_a_name"] = vs_match.group(1).strip().title()
        result["vs_a_loc"]  = vs_match.group(2).strip().title()
        result["vs_b_name"] = vs_match.group(3).strip().title()
        result["vs_b_loc"]  = vs_match.group(4).strip().title()
        return result

    # Top 5 / best
    if re.search(r"\btop\s*\d+\b|\bbest\b|\bgood\b", t):
        # Try "top 5 CATEGORY in LOCATION"
        m = re.search(r"(?:top\s*\d+\s+|best\s+)(.+?)\s+in\s+(.+)", t)
        if m:
            result["intent"]   = "top5"
            result["category"] = m.group(1).strip()
            result["location"] = m.group(2).strip().title()
            return result

    # "analyse/analyze/look into/review X in Y"
    m = re.search(r"(?:analyse|analyze|look into|review|check)\s+(.+?)\s+in\s+(.+)", t)
    if m:
        result["intent"]        = "single_business"
        result["business_name"] = m.group(1).strip().title()
        result["location"]      = m.group(2).strip().title()
        return result

    # "X in Y" generic
    m = re.search(r"(.+?)\s+in\s+(.+)", t)
    if m:
        result["intent"]        = "single_business"
        result["business_name"] = m.group(1).strip().title()
        result["location"]      = m.group(2).strip().title()
        return result

    return result


def triage_supervisor(state: GraphState) -> GraphState:
    user_input = state.user_input.strip()
    history    = state.conversation_history or []

    print(f"[Supervisor] Input='{user_input}'")

    ext    = _llm_extract(user_input, history)
    intent = ext.get("intent", "needs_info")

    # ── vs_comparison ────────────────────────────────────────────────────────
    if intent == "vs_comparison":
        a_name = ext.get("vs_a_name")
        a_loc  = ext.get("vs_a_loc")
        b_name = ext.get("vs_b_name")
        b_loc  = ext.get("vs_b_loc")

        if a_name and a_loc and b_name and b_loc:
            state.search_intent     = "vs_comparison"
            state.vs_a              = {"business_name": a_name, "location": a_loc}
            state.vs_b              = {"business_name": b_name, "location": b_loc}
            state.category          = ext.get("category")
            state.current_agent     = AgentState.SEARCHING
            state.missing_field     = None
            state.interrupt_message = None
        else:
            state.current_agent     = AgentState.WAITING_FOR_LOCATION
            state.missing_field     = "vs_details"
            state.interrupt_message = (
                "⚖️ For a comparison I need both businesses with their locations.\n"
                "Try: **Starbucks Austin vs Starbucks New York**"
            )
        return state

    # ── single_business ──────────────────────────────────────────────────────
    if intent == "single_business":
        biz = ext.get("business_name") or state.business_name
        loc = ext.get("location")      or state.location

        if biz and loc:
            state.search_intent     = "single_business"
            state.business_name     = biz
            state.location          = loc
            state.category          = ext.get("category") or state.category
            state.current_agent     = AgentState.SEARCHING
            state.missing_field     = None
            state.interrupt_message = None
        else:
            state.business_name     = biz or state.business_name
            state.current_agent     = AgentState.WAITING_FOR_LOCATION
            state.missing_field     = "location"
            state.interrupt_message = (
                f"📍 Which city or area should I analyse "
                f"**{state.business_name}** in?\n"
                f"(e.g., 'Austin TX', 'downtown Chicago', 'Manhattan')"
            )
        return state

    # ── top5 ─────────────────────────────────────────────────────────────────
    if intent == "top5":
        cat = ext.get("category") or state.category
        loc = ext.get("location")  or state.location

        if cat and loc:
            state.search_intent     = "top5"
            state.category          = cat
            state.location          = loc
            state.business_name     = None
            state.current_agent     = AgentState.SEARCHING
            state.missing_field     = None
            state.interrupt_message = None
        elif cat:
            state.category          = cat
            state.current_agent     = AgentState.WAITING_FOR_LOCATION
            state.missing_field     = "location"
            state.interrupt_message = f"📍 Which city should I find the best **{cat}s** in?"
        else:
            state.current_agent     = AgentState.WAITING_FOR_CATEGORY
            state.missing_field     = "category"
            state.interrupt_message = (
                "🏪 What type of business are you looking for?\n"
                "(e.g., Restaurants, Gyms, Hotels, Coffee Shops)"
            )
        return state

    # ── needs_location ───────────────────────────────────────────────────────
    if intent == "needs_location":
        state.business_name     = ext.get("business_name") or state.business_name
        state.category          = ext.get("category") or state.category
        state.current_agent     = AgentState.WAITING_FOR_LOCATION
        state.missing_field     = "location"
        state.interrupt_message = (
            f"📍 Where should I look for **{state.business_name}**?\n"
            "Please provide a city or area (e.g., 'Austin TX', 'Chicago')."
        )
        return state

    # ── fallback ─────────────────────────────────────────────────────────────
    state.current_agent     = AgentState.WAITING_FOR_LOCATION
    state.missing_field     = "location"
    state.interrupt_message = (
        "🤔 I didn't quite catch that. Here are some examples:\n"
        "• **Analyse Starbucks in Seattle**\n"
        "• **Top 5 gyms in Austin**\n"
        "• **McDonald's Chicago vs McDonald's New York**"
    )
    return state


def should_interrupt(state: GraphState) -> str:
    if state.current_agent in (AgentState.WAITING_FOR_LOCATION, AgentState.WAITING_FOR_CATEGORY):
        return "interrupt"
    elif state.current_agent == AgentState.SEARCHING:
        return "search"
    elif state.current_agent == AgentState.ERROR:
        return "error"
    return "interrupt"