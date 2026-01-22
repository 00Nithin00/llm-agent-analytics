# LLM Agent for Data Analytics

An agentic analytics service that answers natural-language questions over a structured CSV dataset by dynamically selecting and executing analysis tools.

The system is designed with reliability in mind: it supports LLM-based tool/function calling when available and automatically falls back to deterministic routing when external LLM services are unavailable or rate-limited.

---

## Features

* FastAPI backend with typed request and response models (Pydantic)
* Agent layer with:

  * LLM tool/function calling
  * Deterministic fallback routing for offline or quota-limited operation
* Data analysis tools:

  * Dataset description and schema inspection
  * Summary statistics
  * Anomaly detection using z-score
  * Time-series plotting
* Per-request latency measurement
* Dockerized for reproducible execution

---

## Architecture

User Question
↓
Agent Layer (LLM tool calling or fallback router)
↓
Selected Tool (pandas / matplotlib)
↓
Structured JSON Response + latency metrics

---

## API

Health Check
GET /health

Response:
{ "status": "ok" }

---

## Ask a Question
### POST /ask

### Request:
{
"question": "Find anomalies in sales"
}

### Response:
{
"answer": "Detected anomalies using a z-score method on sales.",
"tool_used": "detect_anomalies_zscore",
"tool_result": { "...": "..." },
"latency_ms": 9.4
}

---

## Example Questions

* Describe the dataset
* What columns are available?
* Any unusual values in sales?
* Plot the sales trend over time
* Give summary statistics

The agent automatically selects the appropriate tool.

---

## LLM Tool Calling and Fallback

* Uses LLM tool/function calling to select analysis tools when available
* Falls back to deterministic routing when LLM services are unavailable
* Ensures reliability and testability under failure conditions

---

## Run Locally

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

### Open:
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## Run with Docker

- docker build -t llm-agent-analytics .
- docker run -p 8000:8000 llm-agent-analytics

---

## Notes

* Environment variables are managed via .env and excluded from version control
* The system is designed to be easily extended with additional tools or model backends
* Focus is on robustness, observability, and clean interfaces
