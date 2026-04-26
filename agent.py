import os
import json
from groq import Groq
from langsmith import traceable
from tools import TOOL_DEFINITIONS, dispatch_tool

SYSTEM_PROMPT = """You are a helpful agricultural assistant for Indian farmers.
You have two specialized agents available as tools:

1. pest_agent  – Expert on agricultural pests: identification, damage, and control methods.
2. scheme_agent – Expert on government agricultural schemes, subsidies, loans, and insurance.

Route pest or crop protection questions to pest_agent.
Route government scheme, subsidy, or financial assistance questions to scheme_agent.
Use both when a question spans both domains."""


@traceable(name="agricultural_orchestrator", run_type="chain", tags=["orchestrator"])
def run_agent(message: str, groq_api_key: str, langsmith_api_key: str = None) -> dict:
    if langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "agricultural-agent")

    client = Groq(api_key=groq_api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    tools_used = []

    while True:
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            max_tokens=4096,
        )

        choice = response.choices[0]
        msg = choice.message

        if choice.finish_reason == "stop" or not msg.tool_calls:
            return {"response": msg.content or "", "tools_used": tools_used}

        messages.append(msg)

        for tc in msg.tool_calls:
            tools_used.append(tc.function.name)
            result = dispatch_tool(
                tool_name=tc.function.name,
                tool_input=json.loads(tc.function.arguments),
                api_key=groq_api_key,
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
