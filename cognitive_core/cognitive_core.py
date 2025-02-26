# !pip install smolagents[litellm]
from smolagents import ToolCallingAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="ollama_chat/deepseek-r1:1.5b", # This model is a bit weak for agentic behaviours though
    api_base="http://127.0.0.1:5000", # replace with 127.0.0.1:11434 or remote open-ai compatible server if necessary
    num_ctx=4096, # ollama default is 2048 which will fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

agent = ToolCallingAgent(tools=[], model=model)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)