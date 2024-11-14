from agency_swarm.agents import Agent

class CEO(Agent):
    def __init__(self):
        super().__init__(
            name="CEO",
            instructions="./instructions.md",
            files_folder="./files",
            schemas_folder="./schemas",
            tools=[],
            tools_folder="./tools",
            temperature=0.01,
            max_prompt_tokens=10000,
            model="gpt-4o-mini",
        )

    def response_validator(self, message):
        return message