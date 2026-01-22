from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)

class AskResponse(BaseModel):
    answer: str
    tool_used: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
