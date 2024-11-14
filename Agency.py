from agency_swarm import set_openai_key
from dotenv import load_dotenv
from agency_swarm.tools import BaseTool
from pydantic import Field
from pathlib import Path
from agency_swarm import Agency
import os
from CEO import CEO
from ExcelAgent import ExcelAgent
from BuissnesAdvisor import BuisnessAgent
# 
#from agency_swarm import Agency

def setApiKey():
    try:
        load_dotenv(Path(__file__).parent / ".env")
        set_openai_key(os.getenv("OPENAI_KEY"))
    except:
        print("[-]Error al obtener: OPENAI_KEY")


if __name__ == "__main__":
    setApiKey()
    
    ceo = CEO()
    ExcelAgent = ExcelAgent()
    BuisnessAdvisor = BuisnessAgent()

    agency = Agency([
        ceo,  
        [ceo, ExcelAgent],
        [ceo, BuisnessAdvisor]
    
    ], shared_instructions='./agency_manifest.md') # instrucciones compartidas 

    agency.demo_gradio(height=900)