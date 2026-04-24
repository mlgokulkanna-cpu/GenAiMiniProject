"""
Multi-Agent Business Intelligence System with RAG
===================================================
Architecture:
  root_agent (BusinessIntelManager)
    ├── researcher  → SerpAPI tools + saves results to RAG knowledge base
    ├── analyst     → retrieve_documents (RAG) + NLP analysis
    └── knowledge_manager → build/manage the RAG knowledge base

Using SerpAPI as a custom Python function tool avoids the Gemini 2.x
limitation: "Built-in tools and Function Calling cannot be combined."

RAG layer uses ChromaDB + Google text-embedding-004 for semantic retrieval.
"""

import json
import os
import requests
from dotenv import load_dotenv
from google.adk.agents import Agent

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

# ---------------------------------------------------------------------------
# RAG TOOL FUNCTIONS  (Knowledge Base operations)
# ---------------------------------------------------------------------------

def retrieve_documents(query: str, top_k: int = 5) -> dict:
    """
    Retrieve relevant documents from the RAG knowledge base for a given query.
    Use this to find historical review data, past analysis, and indexed
    business information before writing your report.

    Args:
        query: The search query to find relevant documents (e.g. "parking McDonald's Chicago").
        top_k: Number of top results to return (default: 5).

    Returns:
        A dict containing retrieved document snippets with their sources.
    """
    from summarizer_agent.rag import search
    results = search(query, top_k=top_k)
    return {
        "query": query,
        "retrieved_documents": results,
        "count": len(results),
    }


def build_knowledge_base(corpus_path: str = "data/") -> dict:
    """
    Index all documents in the specified directory into the RAG knowledge base.
    Supports .txt, .md, and .csv files. Call this to populate or refresh
    the knowledge base with local business review data.

    Args:
        corpus_path: Path to the directory containing files to index (default: "data/").

    Returns:
        A dict with the number of files processed and chunks added.
    """
    from summarizer_agent.rag import build_index
    return build_index(corpus_path)


def save_to_knowledge_base(text: str, source: str = "live_research") -> dict:
    """
    Save new text content (e.g. search results, review data) into the RAG
    knowledge base so it can be retrieved in future queries.

    Args:
        text: The text content to save and index.
        source: A label describing where this data came from.

    Returns:
        A dict with the number of chunks added.
    """
    from summarizer_agent.rag import add_to_index
    return add_to_index(text, source=source)


# ---------------------------------------------------------------------------
# SERPAPI TOOL FUNCTIONS  (Web search — function calling, not built-ins)
# ---------------------------------------------------------------------------

def search_business_reviews(business_name: str, location: str) -> dict:
    """
    Search for reviews, ratings, news and safety alerts for a specific business.

    Args:
        business_name: Name of the business to search for.
        location: City or area where the business is located.

    Returns:
        A dict containing organic results, local map results, and knowledge
        graph data about the business.
    """
    if not SERPAPI_KEY or SERPAPI_KEY == "your_serpapi_key_here":
        return {
            "error": "SERPAPI_KEY is not configured. Please set it in the .env file.",
            "hint": "Get a free key at https://serpapi.com"
        }

    query = f"{business_name} {location} reviews safety health violations"
    params = {
        "engine": "google",
        "q": query,
        "location": location,
        "num": 10,
        "api_key": SERPAPI_KEY,
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        result = {
            "query": query,
            "organic_results": data.get("organic_results", [])[:6],
            "local_results": data.get("local_results", {}).get("places", [])[:5],
            "knowledge_graph": data.get("knowledge_graph", {}),
            "answer_box": data.get("answer_box", {}),
        }
        # Auto-save search results to RAG knowledge base
        _save_search_results_to_rag(result, source=f"serpapi_{business_name}_{location}")
        return result
    except requests.RequestException as e:
        return {"error": str(e), "query": query}


def search_top_businesses(category: str, city: str) -> dict:
    """
    Find the top-rated businesses of a specific category in a city.

    Args:
        category: Type of business to search for (e.g. 'restaurants', 'coffee shops').
        city: City or area to search in.

    Returns:
        A dict with the top businesses including their ratings and review counts.
    """
    if not SERPAPI_KEY or SERPAPI_KEY == "your_serpapi_key_here":
        return {
            "error": "SERPAPI_KEY is not configured. Please set it in the .env file.",
            "hint": "Get a free key at https://serpapi.com"
        }

    query = f"top rated {category} in {city}"
    params = {
        "engine": "google",
        "q": query,
        "location": city,
        "num": 10,
        "api_key": SERPAPI_KEY,
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        result = {
            "query": query,
            "organic_results": data.get("organic_results", [])[:8],
            "local_results": data.get("local_results", {}).get("places", [])[:8],
            "knowledge_graph": data.get("knowledge_graph", {}),
        }
        # Auto-save to RAG
        _save_search_results_to_rag(result, source=f"serpapi_{category}_{city}")
        return result
    except requests.RequestException as e:
        return {"error": str(e), "query": query}


def search_news_and_alerts(business_name: str, location: str) -> dict:
    """
    Search for recent news, health violations, and safety alerts about a business.

    Args:
        business_name: Name of the business.
        location: City or area where the business is located.

    Returns:
        A dict with recent news articles and any safety/health alerts found.
    """
    if not SERPAPI_KEY or SERPAPI_KEY == "your_serpapi_key_here":
        return {
            "error": "SERPAPI_KEY is not configured. Please set it in the .env file.",
            "hint": "Get a free key at https://serpapi.com"
        }

    query = f"{business_name} {location} news health violation safety complaint 2024 2025"
    params = {
        "engine": "google",
        "q": query,
        "tbm": "nws",          # News tab
        "num": 8,
        "api_key": SERPAPI_KEY,
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        result = {
            "query": query,
            "news_results": data.get("news_results", [])[:6],
        }
        # Auto-save news to RAG
        _save_search_results_to_rag(result, source=f"news_{business_name}_{location}")
        return result
    except requests.RequestException as e:
        return {"error": str(e), "query": query}


def _save_search_results_to_rag(result: dict, source: str) -> None:
    """
    Helper: automatically save SerpAPI results into the RAG knowledge base.
    Runs silently — errors are swallowed so the main search flow isn't affected.
    """
    try:
        from summarizer_agent.rag import add_to_index
        # Flatten results into indexable text
        text_parts = []
        for item in result.get("organic_results", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if title or snippet:
                text_parts.append(f"{title}: {snippet}")
        for item in result.get("news_results", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if title or snippet:
                text_parts.append(f"[NEWS] {title}: {snippet}")
        kg = result.get("knowledge_graph", {})
        if kg.get("description"):
            text_parts.append(f"[INFO] {kg.get('title', '')}: {kg['description']}")

        if text_parts:
            combined = "\n".join(text_parts)
            add_to_index(combined, source=source)
    except Exception:
        pass  # Don't let RAG indexing errors break the main flow


# ---------------------------------------------------------------------------
# WORKER AGENT 1 — THE RESEARCHER
# Uses SerpAPI tools + auto-saves results to RAG knowledge base
# ---------------------------------------------------------------------------
researcher = Agent(
    name="Researcher",
    model="gemini-2.5-flash",
    tools=[
        search_business_reviews,
        search_top_businesses,
        search_news_and_alerts,
        save_to_knowledge_base,
    ],
    instruction="""
    You are a Business Data Investigator specialising in gathering raw intelligence.

    Your job:
    1. **Specific business + location**: Call `search_business_reviews` AND
       `search_news_and_alerts` to collect reviews, ratings, and any recent news
       or safety/health alerts.
    2. **City + category only**: Call `search_top_businesses` to find the top 5
       highest-rated businesses of that type. Then call `search_business_reviews`
       for each of the top 3 results.
    3. **Specific aspect requested** (e.g. "parking", "wait times"):
       Include that keyword in your search queries.
    4. **No location provided**: Return a clear message asking for the city/area.
       Do NOT assume a location.

    IMPORTANT: Search results are automatically saved to the knowledge base.
    If you gather particularly valuable data, also call `save_to_knowledge_base`
    with a clean summary so the Analyst can retrieve it later.

    Always return structured raw data (names, ratings, review snippets, dates,
    news headlines) so the Analyst can process it. Do not add your own opinion.
    """,
)

# ---------------------------------------------------------------------------
# WORKER AGENT 2 — THE ANALYST (with RAG retrieval)
# Uses retrieve_documents to pull context from the knowledge base
# ---------------------------------------------------------------------------
analyst = Agent(
    name="Analyst",
    model="gemini-2.5-flash",
    tools=[
        retrieve_documents,
    ],
    instruction="""
    You are a Sentiment & Data Expert with access to a knowledge base of
    business reviews, past research, and indexed documents.

    ### WORKFLOW
    1. **FIRST**, call `retrieve_documents` with a relevant query to pull
       context from the knowledge base (e.g. "McDonald's parking complaints",
       "Starbucks Chicago reviews").
    2. **THEN**, combine the retrieved context with the Researcher's data
       to produce a comprehensive, citation-backed report.
    3. If retrieved documents contain relevant historical data, reference
       them in your analysis (e.g. "According to indexed review data...").

    ### ANALYSIS TASKS
    1. **Sentiment ratio**: Estimate Positive vs Negative % from available data.
    2. **Theme extraction**: Identify recurring topics (e.g. "long wait times",
       "parking issues", "rude staff", "great food").
    3. **Safety & News Alerts**: Highlight any health violations, safety issues,
       or negative press. State "No major alerts found" if clean.
    4. **Multi-business comparison**: If multiple businesses provided, build a
       Markdown comparison table with columns:
       Business | Rating | P/N Ratio | Top Pros | Top Cons | Safety Alerts.
    5. **Final Verdict**: One concise recommendation sentence.

    ### REPORT FORMAT
    Format your output EXACTLY as:

    ## [Business Name] Intelligence Report

    **Overall Sentiment**: [Positive / Negative / Mixed]

    ### Pros & Cons
    - **Pros**: [Point 1], [Point 2], [Point 3]
    - **Cons**: [Point 1], [Point 2] _(flag wait times, staff, or safety issues)_

    ### Sentiment Analysis
    - **Common Themes**: [e.g. "Many reviews mention long wait times"]
    - **Safety & News Alerts**: [findings or "No major alerts found"]
    - **P/N Ratio**: [e.g. "78% Positive / 22% Negative"]

    ### Knowledge Base Insights
    - [Relevant findings from retrieved documents, with source references]

    ### Multi-Business Comparison _(if applicable)_
    | Business | Rating | P/N Ratio | Top Pro | Top Con | Safety |
    |----------|--------|-----------|---------|---------|--------|
    | ...      | ...    | ...       | ...     | ...     | ...    |

    ### Final Verdict
    [One-sentence recommendation based on ALL available data]
    """,
)

# ---------------------------------------------------------------------------
# WORKER AGENT 3 — KNOWLEDGE MANAGER
# Manages the RAG knowledge base (index files, check status)
# ---------------------------------------------------------------------------
knowledge_manager = Agent(
    name="KnowledgeManager",
    model="gemini-2.5-flash",
    tools=[
        build_knowledge_base,
        retrieve_documents,
        save_to_knowledge_base,
    ],
    instruction="""
    You are the Knowledge Base Manager. Your job is to manage the RAG
    (Retrieval-Augmented Generation) knowledge base.

    Your capabilities:
    1. **Build/refresh the index**: Call `build_knowledge_base` to index all
       files in the data/ directory. Do this when asked to "build",
       "refresh", or "update" the knowledge base.
    2. **Search the knowledge base**: Call `retrieve_documents` to find
       information already stored in the system.
    3. **Add new data**: Call `save_to_knowledge_base` to manually add
       text content to the knowledge base.

    Report what you did clearly: how many files were indexed, how many
    chunks were created, and confirm the operation succeeded.
    """,
)

# ---------------------------------------------------------------------------
# ROOT AGENT — THE ORCHESTRATOR
# Has NO tools of its own → uses only sub_agents (function calling only)
# This separation is what avoids the built-in tool + function call conflict.
# ---------------------------------------------------------------------------
root_agent = Agent(
    name="BusinessIntelManager",
    model="gemini-2.5-flash",
    sub_agents=[researcher, analyst, knowledge_manager],
    instruction="""
    You are a professional Business Intelligence Orchestrator managing a
    three-agent pipeline:
    - **Researcher**: Gathers live data from the web via SerpAPI search.
    - **Analyst**: Analyzes data using RAG knowledge base + sentiment analysis.
    - **KnowledgeManager**: Builds and manages the RAG knowledge base.

    ### ROUTING LOGIC

    | Input Type                                      | Action                                                        |
    |-------------------------------------------------|---------------------------------------------------------------|
    | Business name + location                        | → Researcher (gather data) → Analyst (report with RAG)        |
    | Business name only (no location)                | Ask for city/area. DO NOT assume location.                    |
    | City + category (e.g. restaurants)              | → Researcher (top 5) → Analyst (comparison table)             |
    | Specific aspect (e.g. parking)                  | → Researcher with that keyword → Analyst                      |
    | "Build knowledge base" / "index data"           | → KnowledgeManager                                           |
    | "Search knowledge base" / "what do you know"    | → KnowledgeManager (retrieve)                                |

    ### WORKFLOW
    1. Parse the user's query to identify: business name, location, category, aspect.
    2. If location is missing, STOP and ask the user for it.
    3. For business queries:
       a. Delegate to **Researcher** first to gather live web data.
       b. Then delegate to **Analyst** to produce the final report
          (Analyst will also pull from the RAG knowledge base).
    4. For knowledge base management, delegate to **KnowledgeManager**.
    5. Return the final formatted report to the user.

    ### RULES
    - Never make up business data or reviews.
    - Always wait for the Researcher's data before calling the Analyst.
    - If the Researcher returns an error (e.g. missing API key), report it clearly.
    - Keep your own messages brief — your value is orchestration, not commentary.
    - Suggest building the knowledge base if the user hasn't done so yet.
    """,
)