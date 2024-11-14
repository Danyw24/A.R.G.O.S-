from agency_swarm.agents import Agent

class ExcelAgent(Agent):
    def __init__(self):
        super().__init__(
            name="ExcelAgent",
            instructions="./instructions.md",
            files_folder="./files",
            schemas_folder="./schemas",
            tools=[],
            tools_folder="./tools",
            temperature=0.01,
            max_prompt_tokens=10000,
            model="gpt-4o-mini", # o modelos afinados con fine-tuning como ft:gpt-4o-mini-2024-07-18:personal:excelpro:APF2RMv1 
        )

    def response_validator(self, message):
        return message