from agency_swarm.tools import BaseTool
from pydantic import Field, BaseModel
import requests
import os

class DeepSeekParameters(BaseModel):
    prompt: str = Field(..., description="Prompt para DeepSeek-R1")
    max_tokens: int = Field(512, description="Tokens máximos de respuesta")
    temperature: float = Field(0.7, description="Creatividad del modelo")

class DeepSeekTool(BaseTool):
    """Herramienta para interactuar con NVIDIA DeepSeek-R1 API"""
    
    api_key: str = Field(None, description="Clave API de NVIDIA", env="DEEPSEEK_API_KEY")
    base_url: str = "https://api.nvidia.com/deepseek/v1/r1"
    
    def run(self, params: DeepSeekParameters) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": params.prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except Exception as e:
            return f"Error en DeepSeek-R1: {str(e)}"