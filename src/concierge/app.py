"""Gradio chat UI for the restaurant concierge.

    python src/concierge/app.py
"""
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from agent import ask  # noqa: E402  (load .env before the agent reads settings)

EXAMPLES = [
    "Find me a moody place for dinner in Los Angeles",
    "Where can I get affordable food in San Francisco?",
    "What do people say about the best-rated zen restaurant?",
    "Recommend a romantic spot in wine country",
]


def format_tool_calls(tool_calls: list[dict]) -> str:
    lines = []
    for call in tool_calls:
        args = ", ".join(f"{key}={value!r}" for key, value in call["args"].items())
        lines.append(f"- `{call['name']}({args})`")
    return "\n".join(lines)


async def respond(message: str, history: list[dict]):
    reply = await ask(message, history)
    if not reply.tool_calls:
        return reply.answer
    return [
        gr.ChatMessage(
            role="assistant",
            content=format_tool_calls(reply.tool_calls),
            metadata={"title": f"🔧 Used {len(reply.tool_calls)} MCP tool call(s)"},
        ),
        gr.ChatMessage(role="assistant", content=reply.answer),
    ]


demo = gr.ChatInterface(
    fn=respond,
    title="🍽️ California Restaurant Concierge",
    description="Ask for restaurants by cuisine, city, vibe or budget. "
    "Answers come from an MCP server, powered by a local LLM via Ollama.",
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE", "false").lower() == "true")
