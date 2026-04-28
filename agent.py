from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tools import run_pest_tool, run_scheme_tool

SYSTEM_PROMPT = """You are a helpful agricultural assistant for Indian farmers.
You have two specialized tools available:

1. simulate_pests  — Expert on agricultural pests, crop diseases, identification, damage, and control methods.
2. government_schemes — Expert on Indian government agricultural schemes, subsidies, loans, and insurance.

Route pest or crop disease/protection questions to simulate_pests.
Route government scheme, subsidy, or financial assistance questions to government_schemes.
If the question is NOT related to agriculture, do NOT call any tool. Politely explain that you only handle agriculture-related topics."""


def run_agent(message: str, api_key: str) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=api_key)

    @tool
    def simulate_pests(query: str) -> str:
        """A specialized tool for agricultural pest and disease management. Use for any question
        about pests: identification, lifecycle, damage symptoms, affected crops, organic control,
        chemical control, and prevention strategies."""
        return run_pest_tool(api_key=api_key, query=query)

    @tool
    def government_schemes(query: str) -> str:
        """A specialized tool for Indian government agricultural schemes. Use for questions about
        subsidies, loans, crop insurance, financial assistance, and farmer support programs
        (central and state)."""
        return run_scheme_tool(api_key=api_key, query=query)

    tools = [simulate_pests, government_schemes]
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=message),
    ]
    sources = []

    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            sources.append(tc["name"])
            result = tools_by_name[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return {"response": response.content, "sources": sources}
