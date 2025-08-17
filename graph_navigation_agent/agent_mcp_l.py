# agent_mcp_graph_explicit.py
from __future__ import annotations
import os, asyncio, uuid

# LLMs
USE_GEMINI = bool(os.getenv("GOOGLE_API_KEY"))
if USE_GEMINI:
    from langchain_google_genai import ChatGoogleGenerativeAI as ChatModel
    LLM_KW = dict(model="models/gemini-2.5-flash", temperature=0,
                  model_kwargs={"convert_tool_to_function_call": True})
    llm = ChatModel(**LLM_KW)
else:
    from langchain_openai import AzureChatOpenAI
    llm = AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
            model_name=os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
            azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
            cache=None,
            openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_1"),
            api_version="2024-08-01-preview",
            model_version=os.getenv("OPENAI_DEPLOYMENT_VERSION_GPT4_1"),
            temperature=0.1,
            max_tokens=12000,
            openai_api_type="azure",
        )

# LangGraph ReAct agent
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# MCP adapter (agent side)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# ---------- REQUIRED: point to YOUR server ----------
# Streamable HTTP (recommended)
MCP_URL = os.environ.get("MCP_URL")  # e.g., "http://127.0.0.1:8765/mcp"
if not MCP_URL:
    raise SystemExit("Set MCP_URL to your MCP server, e.g. MCP_URL=http://127.0.0.1:8765/mcp")

SERVERS = {
        "KnowledgeEngine":{
            "transport": "streamable_http",
            "url": MCP_URL,
        # Optional: custom headers if your gateway needs them
        # "headers": {"MCP-Protocol-Version": "2025-03-26"},
        }
    }
    


# If you prefer stdio, replace SERVERS with:
# SERVERS = [{
#   "name": "KnowledgeEngine",
#   "transport": "stdio",
#   "command": "python",
#   "args": ["server_mcp.py"],   # your stdio entrypoint
#   "env": {"PYTHONUNBUFFERED": "1"},
# }]

# Tools you expect your server to expose
EXPECTED_TOOLS = {
    "kg_find_edges",
    "kg_neighbors",
    "kg_k_hop",
    "kg_shortest_path",
    "kg_semantic_seed_then_expand_text",  # or "doc_query" if you exposed that instead
    "doc_query",
}


async def build_agent():
    client = MultiServerMCPClient(SERVERS)

    # Open sessions to all configured servers
    ctxs  = [client.session(s) for s in SERVERS]
    # actually enter them and keep the returned sessions
    sessions = await asyncio.gather(*(s.__aenter__() for s in ctxs ))
    # Load tools from all sessions
    tools_all = []
    for s in sessions:
        tools_all.extend(await load_mcp_tools(s))

    # Verify expected tools (or relax to any you find)
    server_tool_names = {t.name for t in tools_all}
    wanted = EXPECTED_TOOLS & server_tool_names
    if not wanted:
        raise RuntimeError(
            f"No expected tools found on {MCP_URL}.\n"
            f"Found: {sorted(server_tool_names)}\n"
            f"Expected any of: {sorted(EXPECTED_TOOLS)}"
        )
    tools = [t for t in tools_all if t.name in wanted]

    # Build the agent
    
    agent = create_react_agent(llm, tools, checkpointer=InMemorySaver())
    from langchain_core.runnables.graph import MermaidDrawMethod
    graph = agent.get_graph(xray=True)

    graph.draw_mermaid_png(
        output_file_path="mcp_query_mermaid.png",
        # draw_method=MermaidDrawMethod.API,           # or PYPPETEER if you installed it
    )

    from langchain_core.runnables.graph_png import PngDrawer
    PngDrawer().draw(graph, "mcp_query_graphviz.png")
    async def _cleanup():
        await asyncio.gather(*(ctx.__aexit__(None, None, None) for ctx in ctxs))

    return agent, _cleanup


async def run_once(user_question: str):
    agent, cleanup = await build_agent()
    try:
        config = {
            "configurable": {
                "run_name": "graph_qa",
                "run_id": str(uuid.uuid4()),
                "thread_id": "default-thread",
            }
        }
        # result = await agent.ainvoke({"messages": [{"role": "user", "content": user_question}]}, config=config)
        events = []
        cnt = 0
        async for event in agent.astream_events({"messages": {"role": "user", "content": user_question # "Use Chroma to  LangChain and scrape langchain.com"
                                                               }}, config = config):
            cnt +=1
            print(event)
            events.append(event)
        final_results = event['data']['output']['messages'][-1].content
        print(final_results)
        print("\n=== FINAL ANSWER ===\n", final_results)
    except Exception as e:
        pass
    finally:
        await cleanup()


if __name__ == "__main__":
    # Example that nudges the agent to use your graph tools
    q = os.getenv(
        "AGENT_QUERY",
        "In document D1, list edges with relation 'causes', then expand one hop around the source node and explain."
    )
    asyncio.run(run_once(q))
    
# uvicorn server_mcp:mcp.streamable_http_app --factory --port 8765