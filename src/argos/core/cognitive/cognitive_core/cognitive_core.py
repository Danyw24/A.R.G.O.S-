from smolagents import ToolCallingAgent, LiteLLMModel, GradioUI, CodeAgent, tool, DuckDuckGoSearchTool,   VisitWebpageTool
import litellm
import gradio as gr

# 1. Configurar el modelo
model = LiteLLMModel(
    model_id="ollama_chat/deepseek-r1",
    api_base="http://127.0.0.1:5000",
    num_ctx=4096,
    temperature=0.7
)

search_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
)

manager_agent = ToolCallingAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
)
manager_agent.run(
    "f"
)
