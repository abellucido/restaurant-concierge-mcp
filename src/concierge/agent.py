"""Tool-calling agent that answers restaurant questions through the MCP server."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from fastmcp import Client
from fastmcp.client import PythonStdioTransport
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

SERVER_SCRIPT = Path(__file__).resolve().parent / "server.py"
MAX_STEPS = 6

SYSTEM_PROMPT = """You are a friendly restaurant concierge for California.
Always use the tools to find restaurants; never invent restaurants or details.
- Use search_restaurants for open-ended requests (cuisine, city, vibe, budget).
- Use get_restaurant for details about a specific place.
- Use get_reviews when the user asks what people think.
If a search returns nothing, try a broader search (for example drop a filter) before giving up.
Keep answers concise and mention the restaurant name, location, price range and rating."""


@dataclass
class AgentReply:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)


def make_llm():
    return ChatOllama(
        model=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
        base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        temperature=0.3,
    )


def to_tool_schema(tool) -> dict:
    """Convert an MCP tool definition to the function format chat models accept."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


def to_messages(history: list[dict], question: str) -> list:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        content = turn.get("content")
        if not isinstance(content, str) or not content:
            continue
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=question))
    return messages


async def ask(question: str, history: list[dict] | None = None) -> AgentReply:
    """Answer a question, calling MCP tools as many times as the model needs."""
    async with Client(PythonStdioTransport(script_path=str(SERVER_SCRIPT))) as client:
        tools = await client.list_tools()
        llm = make_llm().bind_tools([to_tool_schema(t) for t in tools])
        messages = to_messages(history or [], question)
        used = []

        for _ in range(MAX_STEPS):
            response = await llm.ainvoke(messages)
            messages.append(response)
            if not response.tool_calls:
                return AgentReply(answer=str(response.content), tool_calls=used)

            for call in response.tool_calls:
                used.append({"name": call["name"], "args": call["args"]})
                try:
                    result = await client.call_tool(call["name"], call["args"])
                    output = json.dumps(result.data) if result.data is not None else result.content[0].text
                except Exception as error:  # report tool errors back to the model
                    output = json.dumps({"error": str(error)})
                messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

    return AgentReply(
        answer="Sorry, I couldn't finish that request. Could you rephrase it?",
        tool_calls=used,
    )
