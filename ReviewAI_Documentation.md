# ReviewAI: Comprehensive Project Documentation

## 1. Introduction and Overview
ReviewAI is a stateful, multi-agent business intelligence application designed to provide rich, opinionated, and highly specific business reviews and comparisons. Rather than simply returning a list of links or raw ratings like a standard search engine, ReviewAI leverages a multi-agent orchestrated workflow to analyze intent, fetch real-time data, parse customer sentiment, and return structured, actionable insights.

The platform is designed around a **Supervisor-Worker pattern** implemented using **LangGraph**. This approach allows the system to act autonomously, interrupt execution when user input is required (e.g., missing location data), and pass state gracefully between highly specialized agents.

This document serves as a deep dive into the architecture, the agentic workflow, the role of each component, the project structure, and the underlying technology stack, with a special emphasis on the frontend mechanics.

---

## 2. System Architecture & The Agentic Approach

The core philosophy of ReviewAI is modularity and specialization. Instead of relying on a single Large Language Model (LLM) to perform all tasks, the system divides responsibilities among distinct "Agents". Each agent has a specific role, specific tools, and a defined set of inputs and outputs. 

The state machine is managed by LangGraph, which maintains a `GraphState` object containing:
- The user's original input
- The conversation history
- The currently active agent
- Extracted entities (business name, location, category)
- Search results (from SerpAPI)
- Final analysis outputs

### 2.1 The LangGraph State Machine
The workflow operates as a directed graph where nodes represent agents and edges represent conditional logic for routing:

1. **Triage Node (Entry Point)**: Parses user intent and updates state.
2. **Interrupt Node**: If the Triage node detects missing information, it routes to this node to pause execution and ask the user for clarification.
3. **Search Node**: Fetches real-time data.
4. **Analyze Node**: Evaluates the data and performs sentiment analysis.
5. **Verifier**: A synchronous validation step within the analysis node to ensure data integrity.
6. **Error/End Nodes**: Terminal states for the graph.

---

## 3. The Agents and Their Purposes

ReviewAI utilizes four primary agents, each acting sequentially to refine the final output.

### 3.1 Triage Supervisor Agent
**Purpose:** To understand the user's intent, extract key entities (like business name, city, or category), and route the graph accordingly.
**Tools Used:** 
- **Groq LLM (`llama-3.1-8b-instant`)**: Used with a highly restrictive system prompt to output pure JSON.
- **Regex Fallback**: In the event the LLM fails or hallucinates, a resilient Regular Expression parser catches common patterns ("X vs Y", "Top N").
**Responsibilities:**
- Determine if the intent is a single business search, a top-5 list, or a comparison (vs).
- Identify if crucial parameters are missing (e.g., asking for "McDonald's" but failing to provide a city).
- Update the `GraphState` and decide if the graph should proceed to the Search Agent or route to the Interrupt Node to prompt the user.

### 3.2 Search Agent
**Purpose:** To interact with the outside world and fetch real-world data and reviews based on the structured intent provided by the Triage Supervisor.
**Tools Used:** 
- **SerpAPI (Google Maps Engine & Google Search Engine)**: Used to bypass Google's CAPTCHAs and scrape live Google Maps data, ratings, business hours, and review snippets.
**Responsibilities:**
- Execute different search strategies based on intent (`single_business`, `top5`, or `vs_comparison`).
- Implement a robust 3-step waterfall method for fetching reviews:
  1. Try fetching via Google Maps `data_id`.
  2. Try fetching via Google Maps `place_id`.
  3. Fallback to standard Google Search snippets if Maps scraping fails.
- Format the raw API JSON into a clean dictionary to pass to the Analyst.

### 3.3 Analyst Agent
**Purpose:** To digest the raw data and reviews fetched by the Search Agent and produce human-readable, highly opinionated, and structured business intelligence.
**Tools Used:**
- **Groq LLM (`llama-3.1-8b-instant`)**: Tasked with reading up to 15 raw reviews and synthesizing them into a standardized format.
**Responsibilities:**
- Perform sentiment analysis (positive, neutral, negative breakdown).
- Identify top recurring themes (e.g., "slow service", "great ambiance").
- Generate structured Pros and Cons (2-3 complete sentences explaining *why* it matters).
- Write a 4-6 sentence verdict summarizing the business.
- In cases of "top 5" or "vs comparison" intents, it performs a secondary LLM call to rank the businesses and declare a clear "winner" with justified reasoning.

### 3.4 Verifier Agent
**Purpose:** To guarantee that the application never returns vague, empty, or unhelpful data to the frontend.
**Tools Used:** 
- **Pure Python Validation**: Operates synchronously over the Analyst's output.
**Responsibilities:**
- Checks if the Analyst returned empty "Pros" or "Cons". If so, it dynamically generates them based on the numerical rating and review volume.
- Ensures the "Verdict" is sufficiently detailed (minimum length checks).
- Maps raw outputs strictly to Pydantic models (`BusinessAnalysis`) to ensure the frontend receives an exactly expected schema, preventing UI crashes.

---

## 4. Example Query Trace

To fully understand the workflow, let's trace a complex, multi-turn interaction.

**User Input:** *"Analyse Starbucks"*

1. **Triage Supervisor:**
   - **Action:** Receives the input. Groq LLM extracts: `{"intent": "needs_location", "business_name": "Starbucks", "location": null}`.
   - **Decision:** Crucial information (`location`) is missing. Updates state to `WAITING_FOR_LOCATION`.
   - **Routing:** Routes graph to the **Interrupt Node**.
   
2. **Interrupt Node (Frontend interaction):**
   - **Action:** Graph pauses. System sends a message to the user: *"📍 Which city or area should I analyse Starbucks in?"*

3. **User Input:** *"New York"*

4. **Triage Supervisor (Turn 2):**
   - **Action:** Reads conversation history. Understands that "New York" is the missing location for the previous intent.
   - **State Update:** `business_name: "Starbucks"`, `location: "New York"`, `search_intent: "single_business"`.
   - **Routing:** All required fields are present. Routes to the **Search Agent**.

5. **Search Agent:**
   - **Action:** Calls SerpAPI for `"Starbucks New York"`. Retrieves the top result (e.g., Starbucks Reserve Roastery on 5th Ave).
   - **Data Fetch:** Uses the `data_id` to query SerpAPI again for 20 recent reviews.
   - **Routing:** Passes raw reviews and business hours/ratings to the **Analyst Agent**.

6. **Analyst Agent:**
   - **Action:** Prompts Groq LLM with the raw reviews. The LLM returns a structured JSON assessing the coffee quality, seating availability, and wait times based on the reviews.
   - **Routing:** Passes the raw JSON to the **Verifier Agent**.

7. **Verifier Agent:**
   - **Action:** Validates the LLM output against the `BusinessAnalysis` Pydantic model. Confirms Pros/Cons are populated.
   - **Routing:** Updates state to `COMPLETE`.

8. **Frontend Output:**
   - The React frontend receives the final structured JSON and renders the sleek UI cards, complete with sentiment bars, review highlights, and the final verdict.

---

## 5. Project Structure

The project is cleanly separated into backend and frontend repositories, ensuring a decoupled architecture.

### 5.1 Backend Structure (`/backend`)
```text
backend/
├── agents/               # Contains the logic for the LangGraph workers
│   ├── analyst.py        # Groq LLM sentiment and scoring logic
│   ├── search_agent.py   # SerpAPI data fetching logic
│   └── supervisor.py     # Intent extraction and triage logic
├── schemas/              # Pydantic models for strict data typing
│   └── models.py         # Defines GraphState, AgentState, and Output models
├── graph.py              # The LangGraph StateGraph definition and routing logic
├── main.py               # FastAPI server, endpoints, and state management
├── requirements.txt      # Python dependencies
└── start.sh              # Bash script to boot the environment
```

### 5.2 Frontend Structure (`/frontend`)
```text
frontend/
├── src/
│   ├── components/       # Reusable React UI components
│   │   ├── AgentStatusBar.jsx  # Visualizer for the current agent state
│   │   ├── ChatInput.jsx       # User input form
│   │   ├── ChatMessage.jsx     # Renders the complex AI responses and cards
│   │   └── LoadingThinking.jsx # Animated loading states
│   ├── hooks/
│   │   └── useChat.js    # Custom hook handling API calls and local state
│   ├── App.jsx           # Main application layout and assembly
│   ├── index.css         # Global styles and CSS variables
│   └── main.jsx          # React DOM entry point
├── package.json          # Node dependencies
├── tailwind.config.js    # Tailwind CSS configuration
└── vite.config.js        # Vite bundler configuration
```

---

## 6. Technology Stack Overview

### 6.1 The AI & Orchestration Layer
- **LangGraph**: The backbone of the application state. It allows for cyclical graphs (essential for human-in-the-loop/interrupt workflows) and maintains state between API calls.
- **Groq**: Used as the LLM provider. Groq utilizes Language Processing Units (LPUs) rather than GPUs, resulting in insanely fast inference times (~150ms for triage). The model used is `llama-3.1-8b-instant`, which is cost-effective and highly capable for JSON extraction.
- **SerpAPI**: A specialized API for scraping Google search engine results pages securely and reliably, preventing the backend from being IP-banned by Google.

### 6.2 The Backend Layer
- **Python 3.10+**: The core programming language.
- **FastAPI**: A high-performance web framework. It provides automatic interactive API documentation (Swagger) and is highly asynchronous, which is perfect for waiting on LLM and Search APIs.
- **Pydantic**: Used extensively for data validation. It ensures the LLM's raw string output is forcefully cast into strict Python objects before being sent to the client.

### 6.3 The Frontend Layer
- **React 18**: A component-based JavaScript library for building user interfaces.
- **Vite**: A lightning-fast modern build tool and development server that replaces Webpack.
- **Tailwind CSS**: A utility-first CSS framework that allows for rapid, inline styling, creating the modern, "glassmorphism" aesthetic of the application.

---

## 7. Deep Dive: How the Frontend Works

The frontend of ReviewAI is not just a static display; it is a dynamic, state-aware interface that reacts to the LangGraph backend in real-time.

### 7.1 State Management (`useChat.js`)
The core logic resides in the custom React hook `useChat.js`. 
- **Session IDs:** When the app loads, a unique UUID is generated. This ID is passed with every HTTP `POST` to the backend, allowing the FastAPI server to maintain the LangGraph state in memory for that specific user.
- **Message Array:** The hook maintains an array of message objects. Each object contains the text, the sender (`user` or `ai`), and crucially, any structured `data` (like the `BusinessAnalysis` object).
- **Agent State Tracking:** The hook tracks the `agentState` (e.g., `searching`, `analyzing`, `waiting_for_location`). This state is returned by the backend on every request.

### 7.2 The `App.jsx` Layout
The main layout utilizes Flexbox to create a standard chat interface. It implements an auto-scrolling mechanism using `useRef` and `useEffect` so that when new messages or loading indicators appear, the view naturally scrolls to the bottom.

### 7.3 The `AgentStatusBar.jsx` Component
This is a critical UX element. Because multi-agent workflows can take 8-12 seconds to complete (due to multiple API calls), the user needs feedback. The `AgentStatusBar` listens to the `agentState` from the hook and displays a sticky banner at the top of the screen:
- *State: `searching`* -> UI shows: "Agent: Gathering data from SerpAPI..."
- *State: `analyzing`* -> UI shows: "Agent: Synthesizing reviews via Groq LLM..."
This transparency prevents user frustration during long loading times.

### 7.4 Rendering Structured Data (`ChatMessage.jsx`)
When the backend successfully completes a graph run, it returns a JSON payload containing the structured data.
- The `ChatMessage` component checks if the message object contains a `data` property.
- If it is a `single_business` type, it renders a specialized component showing the Overall Score, Pros/Cons lists, and a visual Sentiment Bar (mapping positive/neutral/negative percentages to CSS width properties).
- If it is a `top5` or `vs_comparison` type, it renders a list of cards, highlighting the designated "Winner" with special CSS borders or badges.
- **Tailwind Integration:** The components use dynamic Tailwind classes (e.g., coloring the score badge green if $>7$, yellow if $>5$, red if $<5$) to provide immediate visual context to the user.

### 7.5 Interrupt Handling
If the backend returns `requires_input: true` (e.g., missing location), the frontend does not render a complex data card. Instead, it renders a standard text bubble asking the clarifying question. The `ChatInput.jsx` component dynamically updates its placeholder text based on the current `agentState` to guide the user (e.g., changing from "Ask about a business" to "📍 Enter a city or area...").

---
*End of Documentation*
