# ARGOS_project_beta.py
import os
import logging
from pathlib import Path
from typing import Dict, Any
import requests

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from agency_swarm import Agency, set_openai_key
from pydantic import ValidationError

# Importaciones de los agentes d:
from CEO import CEO
from ExcelAgent import ExcelAgent
from ssh_agent_manager import ssh_agent

# Configuración inicial
load_dotenv(Path(__file__).parent / ".env")
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*")}})


# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ARGOS")

# Constantes
MANIFEST_PATH = Path(__file__).parent / "agency_manifest.md"
REALTIME_MANIFEST_PATH = Path(__file__).parent / "realtime_manifest.md"



class AppConfig:
    """Configuración de la aplicación"""
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_KEY") #OPEN_AI API KEY 
        self.validate_config()
        self.realtime_manifest = ""
        
    def validate_config(self):
        """Valida las variables de entorno requeridas"""
        if not self.openai_key:
            raise EnvironmentError("OPENAI_KEY no está configurada en el .env o no es valida")
        
        required_files = [MANIFEST_PATH, REALTIME_MANIFEST_PATH]
        for file in required_files:
            if not file.exists():
                raise FileNotFoundError(f"Archivo requerido no encontrado: {file.name}")

    def _initialize_realtime_manifest(self):
        with open (REALTIME_MANIFEST_PATH, 'r') as realtime_manifest_file:
            return realtime_manifest_file.read()


def initialize_agency() -> Agency:
    """Inicializa y configura la agencia de agentes"""

    try:
        ceo = CEO()
        excel_agent = ExcelAgent()
        ssh = ssh_agent()
        
        return Agency(
            agency_chart=[
                ceo,
                [ceo, excel_agent],
                [ceo, ssh]
            ],
            shared_instructions=str(MANIFEST_PATH),
            temperature=0.5,
            max_prompt_tokens=2000 
        )
    except Exception as e:
        logger.error("Error inicializando agencia: %s", str(e))
        raise



@app.route('/')
def index():
    """Endpoint principal para la interfaz web de ARGOS"""
    return render_template('main_interface.html')



# sesión RTC para API
@app.route('/session', methods=['GET'])
def get_session():
    try:
        url = "https://api.openai.com/v1/realtime/sessions"
        
        payload = {
            "model": "gpt-4o-mini-realtime-preview",
            "modalities": ["audio", "text"],
            "instructions": config._initialize_realtime_manifest(),
            "voice" : "shimmer"
        }     
        # alloy ta bien
        # coral, meh masomenos
        headers = {
            'Authorization': 'Bearer ' + config.openai_key,
            'Content-Type': 'application/json'
        }

        response = requests.post(url, json=payload, headers=headers)
        return response.json()

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route("/ask", methods=['POST'])
def process_query():
    """Procesa consultas a través de la agencia"""
    try:
        data: Dict[str, Any] = request.get_json()
        # Verifica si existe el campo 'message'
        if not data or 'message' not in data:
            return jsonify({"error": "Formato de solicitud inválido"}), 400 
            
        response = agency.get_completion(data['message']) 
        logger.info("Respuesta generada por la agencia: %s", response)
        return jsonify({"response": response})
        
    except ValidationError as e:
        logger.warning("Error de validación: %s", str(e))
        return jsonify({"error": "Datos de entrada inválidos"}), 400
    except Exception as e:
        logger.error("Error procesando consulta: %s", str(e))
        return jsonify({"error": "Error procesando la solicitud"}), 500


# Integracion y mejora de argos realtime

#def process_query():
    # Paso 1: Obtener contexto histórico desde Redis
    #context = get_conversation_context(user_id)
    """
    # Paso 2: Análisis simbólico inicial
    symbolic_analysis = SymbolicProcessor.analyze(
        query=data['message'],
        context=context,
        neo4j_graph=graph
    )
    
    # Paso 3: Construcción de prompt aumentado
    enriched_prompt = build_enriched_prompt(
        user_query=data['message'],
        symbolic_data=symbolic_analysis,
        manifest=config.realtime_manifest
    )
    
    # Paso 4: Generación con GPT-4
    raw_response = agency.get_completion(enriched_prompt)
    
    # Paso 5: Validación y grounding
    validated_response = Validator.validate(
        response=raw_response,
        user_context=context,
        neo4j_graph=graph
    )
    
    # Paso 6: Actualización de memoria
    update_knowledge_base(
        user_query=data['message'],
        gpt_response=raw_response,
        final_response=validated_response,
        neo4j_graph=graph,
        redis_conn=redis
    )
    
    return jsonify(validated_response)
"""


if __name__ == "__main__":

    try:
        # Configuración inicial
        config = AppConfig()
        set_openai_key(config.openai_key)
        
        # Inicializar agencia
        agency = initialize_agency()
        
        # Configuración del servidor
        app.run(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", 5000)),
            debug=os.getenv("DEBUG_MODE", "false").lower() == "true"
        )
        
    except Exception as e:
        logger.critical("Error crítico durante la inicialización: %s", str(e))
        exit(1)