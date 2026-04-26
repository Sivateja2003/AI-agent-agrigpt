from groq import Groq
from langsmith import traceable

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "pest_agent",
            "description": (
                "A specialized AI agent for agricultural pest management. "
                "Use for any question about pests: identification, lifecycle, "
                "damage symptoms, affected crops, organic control, chemical control, "
                "and prevention strategies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Pest name, crop name, or pest-related question"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scheme_agent",
            "description": (
                "A specialized AI agent for government agricultural schemes. "
                "Use for questions about subsidies, loans, crop insurance, "
                "financial assistance, and farmer support programs (central & state)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Scheme name or type of assistance needed"
                    },
                    "state": {
                        "type": "string",
                        "description": "Indian state name for state-specific schemes (optional)"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


@traceable(name="pest_agent", run_type="llm", tags=["tool", "pest"])
def run_pest_agent(api_key: str, query: str) -> str:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert agricultural entomologist and pest management specialist. "
                    "Provide detailed, practical information structured as: "
                    "1) Pest Overview  2) Identification  3) Damage & Symptoms  "
                    "4) Affected Crops  5) Prevention  6) Organic Control  7) Chemical Control. "
                    "Include safety precautions for any chemical recommendations."
                )
            },
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content


@traceable(name="scheme_agent", run_type="llm", tags=["tool", "scheme"])
def run_scheme_agent(api_key: str, query: str, state: str = None) -> str:
    client = Groq(api_key=api_key)
    full_query = query + (f" (state: {state})" if state else "")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert on Indian government agricultural schemes and farmer support programs. "
                    "Structure your response as: "
                    "1) Scheme Overview  2) Eligibility  3) Benefits/Amount  "
                    "4) How to Apply  5) Required Documents  6) Deadlines  7) Helpline/Contact. "
                    "Cover both central and state government schemes when relevant."
                )
            },
            {"role": "user", "content": full_query}
        ]
    )
    return response.choices[0].message.content


def dispatch_tool(tool_name: str, tool_input: dict, api_key: str) -> str:
    if tool_name == "pest_agent":
        return run_pest_agent(api_key=api_key, query=tool_input["query"])
    elif tool_name == "scheme_agent":
        return run_scheme_agent(
            api_key=api_key,
            query=tool_input["query"],
            state=tool_input.get("state")
        )
    return f"Unknown tool: {tool_name}"
