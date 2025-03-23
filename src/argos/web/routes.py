# ARGOS_project_beta.py
"""
ARGOS:  
El sistema propuesto es una arquitectura de agencia IA multicapa que integra capacidades cognitivas
avanzadas con ejecución autónoma de tareas complejas. En la capa superior de interfaz de usuario,
se procesan entradas multimodales mediante GPT-4 mejorado con un intérprete neuro-simbólico que 
combina modelos de lenguaje con reglas lógicas programables, seguido de un clasificador de intenciones
que detecta objetivos primarios, contexto implícito y riesgos potenciales. Las decisiones estratégicas son 
gestionadas por un núcleo cognitivo tipo CEO que coordina un consejo de directores autónomos especializados
en análisis técnico (CTO), operaciones (COO), gestión de recursos (CFO) y seguridad (CSO), permitiendo 
evaluaciones multidisciplinares para solicitudes críticas. La ejecución se delega en agentes especializados 
(SSH, Excel, desarrollo de código, operaciones web, RAG y seguridad) que pueden auto-generarse para tareas 
específicas, operando sobre una infraestructura distribuida con memoria multinivel (Redis para contexto 
inmediato, bases vectoriales para patrones semánticos y grafos de conocimiento para relaciones complejas).
El ciclo de desarrollo autónomo integra generación de código con análisis estático/dinámico mejorado, 
validación de seguridad, implementación controlada y monitoreo en tiempo real, alimentando un sistema de 
auto-depuración que corrige errores y actualiza la base de conocimiento mediante triple bucle de aprendizaje 
(rápido, medio y estratégico). La seguridad se implementa como capa transversal con verificación en múltiples 
etapas, simulaciones predictivas y arquitectura Zero-Trust, mientras que la personalización se logra mediante 
perfiles dinámicos que almacenan preferencias cognitivas y patrones de interacción física con dispositivos.
Todo el flujo mantiene trazabilidad completa mediante registros en el grafo de conocimiento, permitiendo evolución
continua del sistema sin intervención humana directa, combinando eficiencia operativa con capacidad de 
razonamiento estratégico profundo.
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, Any
import requests
from openai import OpenAI
from datetime import datetime
import traceback
from jsonschema import validate
from json_repair import repair_json


from flask import Flask, jsonify, request, render_template, stream_with_context, Response
from flask_cors import CORS
from dotenv import load_dotenv
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, retry_if_exception_type
import requests
import itertools
import socket

# Configuración inicial
load_dotenv(Path(__file__).parents[3] / ".env")
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*")}})

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)



# Constantes
MANIFEST_PATH = Path(__file__).parent / "agency_manifest.md"
REALTIME_MANIFEST_PATH = Path(__file__).parent / "realtime_manifest.md"

# Configuración de logging
class Log:
  def status(self, msg):
    print("\033[0;96m[~]\033[0m %s" % msg)
  def success(self, msg):
    print("\033[0;92m[+]\033[0m %s" % msg)
  def error(self, msg):
    print("\033[0;91m[!]\033[0m %s" % msg)
  def debug(self, msg):
    print("\033[0;37m[.]\033[0m %s" % msg)
  def notice(self, msg):
    print("\033[0;93m[?]\033[0m %s" % msg)
  def info(self, msg):
    print("\033[0;94m[*]\033[0m %s" % msg)
  def enum(self, index, msg):
    print("\033[0;94m<\033[0m%s\033[0;94m>\033[0m %s" % (index, msg))
  def warning(self, msg):
    print("\033[0;93m[!]\033[0m %s" % msg)

log = Log()


class AppConfig:
    """Configuración de la aplicación"""
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")                                                                                                                                
        self.validate_config()
        self.realtime_manifest = ""
        
    def validate_config(self):
        """Valida las variables de entorno requeridas"""
        if not self.openai_key:
            raise EnvironmentError("OPENAI_API_KEY no está configurada en el .env o no es valida")
        
        required_files = [MANIFEST_PATH, REALTIME_MANIFEST_PATH]
        for file in required_files:
            if not file.exists():
                raise FileNotFoundError(f"Archivo requerido no encontrado: {file.name}")

    def _initialize_realtime_manifest(self):
        with open (REALTIME_MANIFEST_PATH, 'r') as realtime_manifest_file:
            return realtime_manifest_file.read()


def request_deepseek(message: str) -> str:
    full_response = []
    max_retries = 3 
    
    @retry(stop=stop_after_attempt(max_retries), 
    retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError)))
    def stream_completion():
        try:
            completion = client.chat.completions.create(
                model="deepseek-ai/deepseek-r1",
                messages=[{"role": "user", "content": message}],
                response_format={"type": "json_object"},
                temperature=0.6,
                top_p=0.7,
                max_tokens=4096,
                stream=True,
                timeout=15  # Aumentar timeout inicial
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response.append(content)
                    print(content, end="", flush=True)
        
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"\nError de conexión: {str(e)}. Reintentando...")
            raise  # Relanza la excepción para el decorador @retry
            
        return "".join(full_response).strip()

    try:
        for attempt in itertools.count(1):
            try:
                return stream_completion()
                
            except (requests.exceptions.Timeout, 
                    requests.exceptions.ConnectionError) as e:
                if attempt > max_retries:
                    print("\nError: Máximos reintentos alcanzados")
                    return f"Error: No se pudo completar la solicitud después de {max_retries} intentos"
                    
                print(f"\nReintento #{attempt}...")
                continue
                
    except Exception as e:
        print(f"\nError: {str(e)}")
        return "Error procesando la solicitud"


@app.route('/')
def index():
    """Endpoint principal para la interfaz web de ARGOS"""
    return render_template('main.html')


@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    try:
        # Procesar solicitud
        post_data = request.get_json()

        response = request_deepseek(post_data['messages'][-1]['content']) 
        response_data = {
            "model": "deepseek-r1",
            "created_at": int(datetime.now().timestamp()),
            "message": {
                "role": "assistant",
                "content": json.dumps({ 
                    "name": "final_answer",
                    "arguments": {"answer": response}
                })
            },
            "done": True
        }

        return jsonify(response_data)

    except Exception as e:
        log.error(e)
        return jsonify({
            "error": "Error interno",
            "details": str(e)
        }), 500


@app.route('/session', methods=['GET'])
def get_session():
    try:
        url = "https://api.openai.com/v1/realtime/sessions"
        
        payload = {
            "model": "gpt-4o-mini-realtime-preview",
            "modalities": ["audio", "text"],
            "instructions": config._initialize_realtime_manifest(),
            "voice" : "echo"
        }     
        headers = {
            'Authorization': 'Bearer ' + config.openai_key,
            'Content-Type': 'application/json'
        }

        response = requests.post(url, json=payload, headers=headers)
        return response.json()

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    
    try:
        # Configuración inicial
        config = AppConfig()

        #Configuración del servidor 
        app.run(
           host=os.getenv("HOST", "0.0.0.0"),
           port=int(os.getenv("PORT", 5000)),
           debug=os.getenv("DEBUG_MODE", "false").lower() == "true"
        )
        #agency.demo_gradio(height=900)
        
    except Exception as e:
        log.error("Error crítico durante la inicialización: %s", str(e))
        exit(1)