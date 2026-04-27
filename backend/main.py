"""
FastAPI Entry Point — Review AI Multi-Agent System
"""

import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas.models import ChatRequest, ChatResponse, AgentState, GraphState
from graph import get_graph

app = FastAPI(
    title="Review AI — Multi-Agent Orchestrator",
    description="Stateful multi-agent system using LangGraph + Groq + SerpAPI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (replace with Redis for production)
sessions: dict = {}


@app.get("/")
async def root():
    return {"status": "online", "service": "Review AI Multi-Agent Orchestrator"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "serp_configured": bool(os.getenv("SERP_API_KEY")),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id
        user_message = request.message.strip()

        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # ---------------------------------------------------------------
        # Build state for this turn.
        # KEY FIX: always carry over business_name / location / category
        # from the previous turn so the agent doesn't lose context.
        # ---------------------------------------------------------------
        if session_id in sessions:
            prev = sessions[session_id]
            history = list(prev.get("conversation_history", []))
            history.append({"role": "user", "content": user_message})

            state = GraphState(
                session_id=session_id,
                user_input=user_message,
                # Preserve accumulated entities across turns
                business_name=prev.get("business_name"),
                location=prev.get("location"),
                category=prev.get("category"),
                # Always re-triage so the graph runs fresh
                current_agent=AgentState.TRIAGING,
                missing_field=None,
                interrupt_message=None,
                conversation_history=history,
                # Clear stale results
                search_results=None,
                analysis_results=None,
                final_output=None,
                error_message=None,
            ).model_dump()
        else:
            state = GraphState(
                session_id=session_id,
                user_input=user_message,
                current_agent=AgentState.TRIAGING,
                conversation_history=[{"role": "user", "content": user_message}]
            ).model_dump()

        # Run the LangGraph
        graph = get_graph()
        result = graph.invoke(state)

        gs = GraphState(**result)

        # Persist updated state (includes newly extracted entities)
        sessions[session_id] = result

        # Build response
        agent_state = gs.current_agent.value
        response_data = None
        message = ""
        requires_input = False
        input_prompt = None

        if gs.current_agent in (AgentState.WAITING_FOR_LOCATION, AgentState.WAITING_FOR_CATEGORY):
            message = gs.interrupt_message or "Could you provide more details?"
            requires_input = True
            input_prompt = gs.missing_field
            sessions[session_id]["conversation_history"].append({
                "role": "assistant", "content": message
            })

        elif gs.current_agent == AgentState.COMPLETE and gs.final_output:
            output = gs.final_output
            if output.get("type") == "single_business":
                biz = output["data"]
                message = (
                    f"✅ Analysis complete for **{biz['business_name']}**!\n\n"
                    f"Overall Score: **{biz['overall_score']}/10** | "
                    f"Recommendation: **{biz['recommendation'].replace('_', ' ')}**\n\n"
                    f"{biz['verdict']}"
                )
            else:
                winner = output.get("winner", "")
                count = len(output.get("businesses", []))
                message = (
                    f"🏆 Top {count} **{output.get('category', '').title()}s** "
                    f"in **{output.get('location', '')}** analyzed!\n\n"
                    f"**Winner: {winner}** — {output.get('winner_reason', '')}"
                )
            response_data = output
            sessions[session_id]["conversation_history"].append({
                "role": "assistant", "content": message
            })

        elif gs.current_agent == AgentState.ERROR:
            message = f"❌ {gs.error_message or 'An error occurred. Please try again.'}"
            sessions[session_id]["conversation_history"].append({
                "role": "assistant", "content": message
            })

        else:
            message = "Processing your request..."

        return ChatResponse(
            session_id=session_id,
            message=message,
            agent_state=agent_state,
            data=response_data,
            requires_input=requires_input,
            input_prompt=input_prompt,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Chat] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared", "session_id": session_id}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)