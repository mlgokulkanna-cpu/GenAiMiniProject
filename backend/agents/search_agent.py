"""
Search Agent — SerpAPI Integration
Handles: single_business, top5, vs_comparison search intents.

KEY FIXES:
- Tries data_id first, then place_id, then Google Search fallback for reviews
- Uses google_search with "reviews" query as guaranteed fallback
- Logs every step so issues are visible in terminal
"""

import os
import time
from typing import List, Dict, Any
from serpapi import GoogleSearch
from schemas.models import GraphState, AgentState

SERP_API_KEY = os.getenv("SERP_API_KEY", "")


def _parse_reviews(raw: list, limit: int = 20) -> List[dict]:
    out = []
    for r in raw[:limit]:
        text = (r.get("snippet") or r.get("text") or r.get("description") or "").strip()
        if not text or len(text) < 20:
            continue
        out.append({
            "text":   text,
            "rating": r.get("rating"),
            "date":   r.get("date", ""),
            "author": (r.get("user") or {}).get("name") or r.get("author", "Reviewer"),
        })
    return out


def _fetch_reviews(place: dict, business_name: str, location: str) -> List[dict]:
    """
    Three-method waterfall to get reviews:
    1. google_maps_reviews with data_id
    2. google_maps_reviews with place_id
    3. google search for review snippets (always returns something)
    """

    # ── Method 1: data_id ────────────────────────────────────────────────────
    data_id = place.get("data_id", "")
    if data_id:
        try:
            res = GoogleSearch({
                "engine":  "google_maps_reviews",
                "data_id": data_id,
                "api_key": SERP_API_KEY,
                "hl":      "en",
                "sort_by": "qualityScore",
            }).get_dict()
            reviews = _parse_reviews(res.get("reviews", []))
            print(f"[Search] data_id -> {len(reviews)} reviews for '{business_name}'")
            if reviews:
                return reviews
        except Exception as e:
            print(f"[Search] data_id error: {e}")

    # ── Method 2: place_id ───────────────────────────────────────────────────
    place_id = place.get("place_id", "")
    if place_id:
        try:
            res = GoogleSearch({
                "engine":   "google_maps_reviews",
                "place_id": place_id,
                "api_key":  SERP_API_KEY,
                "hl":       "en",
                "sort_by":  "qualityScore",
            }).get_dict()
            reviews = _parse_reviews(res.get("reviews", []))
            print(f"[Search] place_id -> {len(reviews)} reviews for '{business_name}'")
            if reviews:
                return reviews
        except Exception as e:
            print(f"[Search] place_id error: {e}")

    # ── Method 3: Google Search (guaranteed fallback) ────────────────────────
    try:
        res = GoogleSearch({
            "engine":  "google",
            "q":       f'"{business_name}" {location} reviews customers experience',
            "api_key": SERP_API_KEY,
            "num":     10,
        }).get_dict()
        snippets = []
        for r in res.get("organic_results", []):
            snippet = r.get("snippet", "").strip()
            if snippet and len(snippet) > 30:
                snippets.append({
                    "text": snippet, "rating": None,
                    "date": "", "author": "Google Search",
                })
        print(f"[Search] Google Search fallback -> {len(snippets)} snippets for '{business_name}'")
        return snippets
    except Exception as e:
        print(f"[Search] Google Search fallback error: {e}")

    return []


def _maps_search(query: str) -> List[dict]:
    try:
        res = GoogleSearch({
            "engine":  "google_maps",
            "q":       query,
            "api_key": SERP_API_KEY,
            "type":    "search",
            "hl":      "en",
        }).get_dict()
        results = res.get("local_results", [])
        print(f"[Search] Maps '{query}' -> {len(results)} results")
        return results
    except Exception as e:
        print(f"[Search] Maps error: {e}")
        return []


def _build_biz_info(place: dict, fallback_name: str = "") -> dict:
    return {
        "name":          place.get("title", fallback_name),
        "address":       place.get("address", ""),
        "phone":         place.get("phone", ""),
        "website":       place.get("website", ""),
        "rating":        place.get("rating"),
        "reviews_count": place.get("reviews"),
        "hours":         str(place.get("hours", "")),
        "place_id":      place.get("place_id", ""),
        "data_id":       place.get("data_id", ""),
    }


def _search_single(business_name: str, location: str) -> Dict[str, Any]:
    result = {"business_info": {}, "reviews": [], "error": None}
    places = _maps_search(f"{business_name} {location}")
    if not places:
        result["error"] = f"No results found for '{business_name}' in {location}"
        return result
    place = places[0]
    result["business_info"] = _build_biz_info(place, business_name)
    result["reviews"]       = _fetch_reviews(place, business_name, location)
    print(f"[Search] Single result: {result['business_info'].get('name')} | {len(result['reviews'])} reviews")
    return result


def _search_top5(category: str, location: str) -> List[Dict[str, Any]]:
    places = _maps_search(f"best {category} in {location}")
    if not places:
        return []
    businesses = []
    for place in places[:5]:
        name = place.get("title", category)
        biz  = {
            "business_info": _build_biz_info(place, name),
            "reviews":       [],
        }
        try:
            time.sleep(0.3)
            biz["reviews"] = _fetch_reviews(place, name, location)
        except Exception as e:
            print(f"[Search] Reviews failed for '{name}': {e}")
        businesses.append(biz)
    return businesses


def _search_vs(vs_a: dict, vs_b: dict) -> List[Dict[str, Any]]:
    results = []
    for side in [vs_a, vs_b]:
        name = side.get("business_name", "")
        loc  = side.get("location", "")
        res  = _search_single(name, loc)
        res["business_info"]["search_location"] = loc
        results.append(res)
    return results


def search_agent(state: GraphState) -> GraphState:
    state.current_agent = AgentState.SEARCHING
    intent = getattr(state, "search_intent", "single_business") or "single_business"
    print(f"[Search] Intent={intent}")

    try:
        if intent == "vs_comparison":
            vs_a = getattr(state, "vs_a", {}) or {}
            vs_b = getattr(state, "vs_b", {}) or {}
            if not (vs_a.get("business_name") and vs_b.get("business_name")):
                state.current_agent = AgentState.ERROR
                state.error_message = "Missing business names for comparison"
                return state
            results = _search_vs(vs_a, vs_b)
            if not results:
                state.current_agent = AgentState.ERROR
                state.error_message = "Could not find either business"
                return state
            state.search_results = results

        elif intent == "top5":
            if not state.category or not state.location:
                state.current_agent = AgentState.ERROR
                state.error_message = "Missing category or location"
                return state
            results = _search_top5(state.category, state.location)
            if not results:
                state.current_agent = AgentState.ERROR
                state.error_message = f"No results for {state.category} in {state.location}"
                return state
            state.search_results = results

        else:  # single_business
            if not state.business_name or not state.location:
                state.current_agent = AgentState.ERROR
                state.error_message = "Missing business name or location"
                return state
            result = _search_single(state.business_name, state.location)
            if result.get("error") and not result.get("business_info", {}).get("name"):
                state.current_agent = AgentState.ERROR
                state.error_message = result["error"]
                return state
            state.search_results = [result]

        state.current_agent = AgentState.ANALYZING

    except Exception as e:
        state.current_agent = AgentState.ERROR
        state.error_message = f"Search failed: {str(e)}"

    return state