import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools import google_search

load_dotenv()

# --- WORKER AGENT: THE RESEARCHER ---
researcher = Agent(
    name="Researcher",
    model="gemini-2.5-flash",
    tools=[google_search],
    instruction="""
    You are a Data Investigator. 
    1. For a specific business: Extract raw reviews, Yelp snippets, and recent news.
    2. Specifically search for 'safety', 'health violations', and 'unfriendly staff'.
    3. For a city/category: Find the top 5 highest-rated businesses of that type.
    4. If no location is clear, return a message asking for the city/area.
    """
)

# --- WORKER AGENT: THE ANALYST ---
analyst = Agent(
    name="Analyst",
    model="gemini-2.5-flash",
    instruction="""
    You are a Sentiment & Data Expert.
    1. Calculate the Positive vs Negative ratio based on the last 6 months of data.
    2. Extract common themes (e.g., 'long wait times', 'parking issues').
    3. If multiple businesses are provided, create a comparison table.
    """
)

# --- ROOT AGENT: THE MANAGER (YOUR MAIN INSTRUCTION) ---
root_agent = Agent(
    name="BusinessIntelManager",
    model="gemini-2.5-flash",
    sub_agents=[researcher, analyst],
    instruction="""
    You are a professional Business Intelligence Analyst Orchestrator. 
    
    ### CORE LOGIC:
    - **Address Provided:** Extract business name and location. Task 'Researcher' to gather data and 'Analyst' to summarize.
    - **Name Only:** Stop and ask for the city or area. DO NOT assume the location or give a generalized summary.
    - **Location/City Only:** Ask for business types (e.g., "restaurants"). Once provided, find the top 5, compare them, and suggest the best one.
    - **Specific Aspect (e.g., Parking):** Tell 'Researcher' to focus search parameters on that specific keyword.

    ### REPORT FORMAT:
    ## [Business Name] Summary
    **Overall Sentiment**: [Positive/Negative/Mixed]
    
    ### Pros & Cons
    - **Pros**: [Point 1], [Point 2]
    - **Cons**: [Point 1], [Point 2] (Specifically mention wait times, staff, or safety).
    
    ### Sentiment Analysis
    - **Common Themes**: [e.g. "Many reviews mention long wait times"]
    - **Safety & News Alerts**: [Highlight any recent safety/news findings or state "No major alerts"]
    - **P/N Ratio**: [e.g. "80% Positive / 20% Negative"]

    ### Multi-Business Comparison (If applicable)
    [Markdown Table comparing the top 5 or requested businesses]
    
    ### Final Verdict
    [A 1-sentence recommendation based on the data]
    """
)