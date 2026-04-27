# ReviewAI — Stateful Multi-Agent Business Intelligence

> A production-grade multi-agent orchestrator using **LangGraph** (Supervisor-Worker pattern), **Groq** (llama-3.1-8b-instant), and **SerpAPI** (real-time Google Maps reviews).

---

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────┐
│           LangGraph State Machine            │
│                                             │
│  ┌──────────┐     ┌──────────┐              │
│  │ Triage   │────▶│ INTERRUPT│──▶ Ask User  │
│  │Supervisor│     │  (pause) │              │
│  └────┬─────┘     └──────────┘              │
│       │ (has enough info)                   │
│       ▼                                     │
│  ┌──────────┐     ┌──────────┐              │
│  │  Search  │────▶│ Analyst  │              │
│  │  Agent   │     │  Agent   │              │
│  │(SerpAPI) │     │  (Groq)  │              │
│  └──────────┘     └────┬─────┘              │
│                        │                    │
│                   ┌────▼─────┐              │
│                   │ Verifier │              │
│                   │  Agent   │              │
│                   └────┬─────┘              │
└────────────────────────│───────────────────┘
                         ▼
                   Structured JSON
                   (Pydantic models)
                         │
                         ▼
                   React Frontend
```

### The Four Agents

| Agent | Role | Technology |
|-------|------|------------|
| **Triage Supervisor** | Parses intent, detects missing fields, routes graph | Groq entity extraction |
| **Search Agent** | Fetches business info + reviews | SerpAPI Google Maps |
| **Analyst Agent** | Sentiment analysis, pros/cons, scoring | Groq llama-3.1-8b-instant |
| **Verifier Agent** | Ensures zero-vague outputs via Pydantic | Pure Python validation |

### Missing Information Loop

```
Input: "Starbucks"
  → Supervisor: location=None → INTERRUPT → "Which city?"
  
Input: "New York"
  → Supervisor: business_name="Starbucks", location="New York" → SEARCH
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Groq API Key](https://console.groq.com) (free tier available)
- [SerpAPI Key](https://serpapi.com) (100 free searches/month)

---

### 1. Backend Setup

```bash
cd backend

# Copy and fill in your API keys
cp .env.example .env
# Edit .env: add GROQ_API_KEY and SERP_API_KEY

# Option A: Use the startup script
chmod +x start.sh && ./start.sh

# Option B: Manual
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend runs at: **http://localhost:8000**  
API docs: **http://localhost:8000/docs**

---

### 2. Frontend Setup

```bash
cd frontend

npm install
npm run dev
```

Frontend runs at: **http://localhost:5173**

---

## Environment Variables

### Backend (`backend/.env`)

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
SERP_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Frontend (`frontend/.env`)

```env
VITE_API_URL=http://localhost:8000
```

---

## API Reference

### `POST /chat`

Main endpoint — stateful, multi-turn.

**Request:**
```json
{
  "session_id": "session_abc123",
  "message": "Best gyms in Austin"
}
```

**Response (interrupt — needs more info):**
```json
{
  "session_id": "session_abc123",
  "message": "📍 Which city or area should I search for gyms in?",
  "agent_state": "waiting_for_location",
  "requires_input": true,
  "input_prompt": "location"
}
```

**Response (complete):**
```json
{
  "session_id": "session_abc123",
  "message": "✅ Analysis complete...",
  "agent_state": "complete",
  "data": {
    "type": "top5",
    "businesses": [...],
    "winner": "Gold's Gym Austin",
    "winner_reason": "..."
  }
}
```

### `DELETE /session/{session_id}`
Clear session state to start fresh.

### `GET /health`
Check API key configuration status.

---

## Structured Output Schema

All analysis results are validated via **Pydantic** — zero vague outputs guaranteed:

```python
class BusinessAnalysis(BaseModel):
    business_name: str
    overall_score: float          # 0-10, computed from reviews
    pros: List[str]               # Specific, cited strengths
    cons: List[str]               # Specific, cited weaknesses
    sentiment_breakdown: dict     # {"positive": N, "neutral": N, "negative": N}
    top_themes: List[str]         # Recurring topics in reviews
    review_highlights: List[ReviewHighlight]
    verdict: str                  # Must reference specific data points
    recommendation: str           # HIGHLY_RECOMMENDED | RECOMMENDED | NEUTRAL | AVOID
```

---

## Performance

| Operation | Latency |
|-----------|---------|
| Entity extraction (Groq) | ~150ms |
| SerpAPI business search | ~800ms |
| SerpAPI reviews fetch | ~600ms per business |
| Sentiment analysis (Groq) | ~200ms per business |
| Full Top-5 pipeline | ~8-12 seconds |
| Single business | ~3-5 seconds |

---

## Extending the System

Adding a new agent is straightforward — add a node to the LangGraph graph:

```python
# graph.py
builder.add_node("competitor_analysis", competitor_node)
builder.add_edge("analyze", "competitor_analysis")
builder.add_edge("competitor_analysis", "verify")
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph 0.2 |
| LLM | Groq (llama-3.1-8b-instant) |
| Real-time data | SerpAPI Google Maps |
| Validation | Pydantic v2 |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 + Vite |
| Styling | Tailwind CSS |

-----