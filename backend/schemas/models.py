from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class AgentState(str, Enum):
    TRIAGING = "triaging"
    WAITING_FOR_LOCATION = "waiting_for_location"
    WAITING_FOR_CATEGORY = "waiting_for_category"
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    ERROR = "error"


class ReviewSentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ReviewHighlight(BaseModel):
    text: str = Field(description="The exact review snippet")
    sentiment: ReviewSentiment
    theme: str = Field(description="Category: service, quality, price, ambiance, etc.")


class BusinessAnalysis(BaseModel):
    business_name: str
    location: str
    overall_score: float = Field(ge=0, le=10, description="Score out of 10")
    rating: Optional[float] = Field(None, description="Google rating")
    total_reviews: Optional[int] = None
    pros: List[str] = Field(default_factory=list, min_length=0)
    cons: List[str] = Field(default_factory=list, min_length=0)
    sentiment_breakdown: dict = Field(
        default_factory=lambda: {"positive": 0, "neutral": 0, "negative": 0}
    )
    top_themes: List[str] = Field(default_factory=list)
    review_highlights: List[ReviewHighlight] = Field(default_factory=list)
    verdict: str = Field(description="One-paragraph verdict backed by specific data")
    recommendation: str = Field(description="HIGHLY_RECOMMENDED | RECOMMENDED | NEUTRAL | AVOID")
    address: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    hours: Optional[str] = None


class Top5Analysis(BaseModel):
    category: str
    location: str
    businesses: List[BusinessAnalysis]
    winner: str = Field(description="Name of the top-ranked business")
    winner_reason: str = Field(description="Why this business won, citing specific data")


class GraphState(BaseModel):
    """LangGraph state shared across all agents"""
    session_id: str
    user_input: str
    business_name: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    # Intent routing: single_business | top5 | vs_comparison
    search_intent: Optional[str] = None
    # VS comparison sides
    vs_a: Optional[dict] = None
    vs_b: Optional[dict] = None
    current_agent: AgentState = AgentState.TRIAGING
    missing_field: Optional[str] = None
    interrupt_message: Optional[str] = None
    search_results: Optional[List[dict]] = None
    analysis_results: Optional[List[BusinessAnalysis]] = None
    final_output: Optional[dict] = None
    error_message: Optional[str] = None
    conversation_history: List[dict] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str
    agent_state: Optional[str] = None
    data: Optional[dict] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    conversation_history: List[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    message: str
    agent_state: str
    data: Optional[dict] = None
    requires_input: bool = False
    input_prompt: Optional[str] = None