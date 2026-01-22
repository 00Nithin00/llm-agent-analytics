from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from app import tools

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tool specs for function calling
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "describe_data",
            "description": "Get dataset schema, dtypes, and a small preview of rows.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summary_stats",
            "description": "Compute summary statistics for the dataset or selected columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of columns to summarize",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies_zscore",
            "description": "Detect anomalies in a numeric column using a z-score threshold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "z_thresh": {"type": "number", "default": 2.5},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_timeseries",
            "description": "Generate a time series plot from a date column and a numeric value column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_col": {"type": "string"},
                    "value_col": {"type": "string"},
                },
                "required": ["date_col", "value_col"],
            },
        },
    },
]

SYSTEM = """You are an AI assistant that answers questions about a CSV dataset.
Choose the best tool to use when needed. Prefer tools over guessing.
Return concise, factual answers. If the user request is unclear, ask a short clarifying question.
"""

def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "describe_data":
        return tools.describe_data()
    if name == "summary_stats":
        return tools.summary_stats(columns=args.get("columns"))
    if name == "detect_anomalies_zscore":
        return tools.detect_anomalies_zscore(
            column=args["column"],
            z_thresh=float(args.get("z_thresh", 2.5)),
        )
    if name == "plot_timeseries":
        return tools.plot_timeseries(date_col=args["date_col"], value_col=args["value_col"])
    raise ValueError(f"Unknown tool: {name}")

def answer(question: str) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """Returns: (final_answer, tool_used, tool_result)"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        tools=TOOL_SPECS,
        tool_choice="auto",
        temperature=0.2,
    )

    msg = resp.choices[0].message

    # If the model decided to call a tool
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments or "{}")
        tool_result = run_tool(tool_name, tool_args)

        # Ask model to summarize tool output
        resp2 = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"Tool used: {tool_name}\nTool result: {json.dumps(tool_result)[:6000]}"},
                {"role": "user", "content": "Provide a concise answer based on the tool result."},
            ],
            temperature=0.2,
        )
        final_answer = resp2.choices[0].message.content or "Done."
        return final_answer, tool_name, tool_result

    # Otherwise answer directly
    return msg.content or "Done.", None, None
