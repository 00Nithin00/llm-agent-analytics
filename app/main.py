from fastapi import FastAPI, HTTPException
from app.models import AskRequest, AskResponse
from app.eval import now
from app.agent import answer

app = FastAPI(title="LLM Agent Analytics")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = now()
    try:
        final, tool_used, tool_result = answer(req.question)
        latency_ms = (now() - t0) * 1000
        return AskResponse(
            answer=final,
            tool_used=tool_used,
            tool_result=tool_result,
            latency_ms=latency_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
