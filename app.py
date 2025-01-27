# A.R.G.O.S project beta
from flask import Flask, jsonify, request, render_template
import requests
from flask_cors import CORS
from dotenv import load_dotenv
from agency_swarm import Agency
from gradio_client import Client
from agency_swarm import set_openai_key
from dotenv import load_dotenv
from agency_swarm.tools import BaseTool
from pydantic import Field
from pathlib import Path
import os
from CEO import CEO
from ExcelAgent import ExcelAgent
from BuissnesAdvisor import BuisnessAgent
from ssh_agent_manager import ssh_agent

load_dotenv()

app = Flask(__name__)
API_KEY=os.getenv("OPENAI_KEY")


# Instrucciones y condiciones de gtp4 realtime API
with open("realtime_manifest.md", "r", encoding="utf-8") as rt_model_instructions:
    realtime_instructions = rt_model_instructions.read()  


# Establecer API KEY 
def setApiKey():
    try:
        load_dotenv(Path(__file__).parent / ".env")
        set_openai_key(os.getenv("OPENAI_KEY"))

    except:
        print("[-]Error al obtener: OPENAI_KEY")
   


@app.route('/')
def index():
    return render_template('main_interface.html')


# sesión RTC para API
@app.route('/session', methods=['GET'])
def get_session():
    try:
        url = "https://api.openai.com/v1/realtime/sessions"
        
        payload = {
            "model": "gpt-4o-realtime-preview",
            "modalities": ["audio", "text"],
            "instructions": realtime_instructions,
            "voice" : "ash"
        }     
        
        headers = {
            'Authorization': 'Bearer ' + API_KEY,
            'Content-Type': 'application/json'
        }

        response = requests.post(url, json=payload, headers=headers)
        return response.json()

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Obtener datos de Agency Swarm
@app.route("/ask", methods=['POST'])
def ask():
    try: 
        data = request.json
        user_input = data.get("message", "")
        agency_completion = agency.get_completion(user_input)
        print(agency_completion)
        return jsonify({"response": agency_completion})
    except Exception as err:
        return jsonify({"error" : "Ha ocurrido un error :" + str(err)})
    
    
# Entry point
if __name__ == "__main__":
    # OpenAI API key
    setApiKey()
    
    #Inicializar Agentes IA especializados 
    ceo = CEO()
    ExcelAgent = ExcelAgent()
    BuisnessAdvisor = BuisnessAgent()
    ssh_agent = ssh_agent()
    global agency
    agency = Agency([
        ceo,  
        [ceo, ExcelAgent],
        [ceo, BuisnessAdvisor],
        [ceo, ssh_agent]
    
        ],
        shared_instructions='./agency_manifest.md',
        temperature=0.5, # default temperature for all agents
        max_prompt_tokens=75    
    ) # instrucciones compartidas 

    # usar para gradio y depurar agentes agency.demo_gradio(height=900)
    app.run(debug=True)

