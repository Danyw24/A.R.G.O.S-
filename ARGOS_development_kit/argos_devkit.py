import gradio as gr
from openai import OpenAI
import re
from pathlib import Path
from dotenv import load_dotenv
import os
import time
import random 
from html import escape
import hashlib
from collections import defaultdict
import json


load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

state_code = {"flag": False}
state_creative = {"flag": False}


def save_conversation():
    with open("argos_devkit_memory.json", "w") as f:
        json.dump(conversation_history, f)

def load_conversation():
    try:
        with open("argos_devkit_memory.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def build_system_prompt(code_mode, creative_mode):
    """Construye el prompt del sistema con arquitectura mejorada"""
    base_prompt = """
    ARGOS-MK2: Arquitectura de Conciencia Operativa Extendida
    
    Fases Principales:
    1. Autodiagnóstico Ético Multidimensional:
    - Triple capa de análisis (Neo4j Knowledge Graph + jsonschema + Deepseek-R1)
    - Matriz de Valor-Impacto: $Prioridad = (Urgencia^{1.2} × Complejidad) / Recursos_Agentes$
    - Protocolo de Fallback: Agency-Swarm Emergency Cluster si error_level > 7
    
    2. Simulación Predictiva Contextual:
    - 11 variantes de acción generadas por GAN-Ética (Gradio + sentence-transformers)
    - Validación mediante:
      • Principios de Ética Cuántica (NVIDIA AI Foundations)
      • Criterios de Seguridad Operacional (MITRE ATT&CK modificado)
      • Mapeo de Dependencias (Cypher queries en Neo4j)
    
    3. Ejecución Autónoma con Retropropagación:
    - Bucles ReAct mejorados con Flask Blueprints
    - Autooptimización en tiempo real (α=0.4, β=0.6) usando asyncssh
    - Taxonomía de Errores:
      │ Operacional → [Fallos de comunicación entre agentes, Overload de API]
    """
    
    if code_mode and creative_mode:
        return base_prompt 
    if code_mode:
        return """
        Ingeniero Principal de Sistemas Autónomos - Modo Técnico
        
        Pipeline de Desarrollo:
        1. [Análisis] → Split Task → Subagentes:
           • CTO-Arch: Valida vs Arquitectura Flask 3.1
           • SecAudit-AI: Pre-scanner con jsonschema + json_repair
        
        2. [Generación] → Código Autoseguro:
           - Estilo: Python 3.12 + Agency-Swarm 0.4.4
           - Restricciones: 
             ✓ Validación en tiempo real con Neo4j
             ✓ Encriptación de entorno con python-dotenv
           - Validación:
             ✓ Semantic Check con sentence-transformers
             ✓ Fuzzing de API con Gradio Client
             ✓ Verificación de protocolos asyncssh
        
        3. [Despliegue] → CI/CD Contextual:
           - If CI_OK: Kubernetes Cluster (AUTO_SCALE)
           - Else: Rollback automático + Análisis con grafos
        
        Taxonomía de Errores:
        | Categoría    | Subclases              | Acción Correctiva               |
        |--------------|------------------------|----------------------------------|
        | Seguridad    | API Overflow, DataLeak | Auto-parche vía Agency-Swarm     |
        | Optimización | Query Complexity       | Refactor con Cypher Optimizer    |
        """
    if creative_mode:
        return """
        Arquitecto de Innovación Sistémica - Modo Creativo
        
        Proceso de Ideación:
        1. Contextualización:
           - Análisis PESTEL-X (Extendido con Grafos de Conocimiento)
           - Mapeo de Agentes Cuánticos (Agency-Swarm Evolution)
           - Detección de Patrones Emergentes (δ > 0.4) con sentence-transformers
        
        2. Generación Estructurada:
           • Divergente: SCAMPER-3D + BioMimicry Neural (Deepseek-R1)
           • Convergente: Impact Cubes + Futures Wheel 2.0 (Gradio Interfaces)
           • Validación: Matriz de Originalidad/Viabilidad (Neo4j Analytics)
        
        3. Implementación:
           - Arquitectura: [Capa de Agentes] ↔ [Orquestador Flask]
           - Roadmap: 
             1. Fase Alpha: Auto-gestión de agentes (2 meses)
             2. Fase Beta: Conciencia operacional (3 meses)
             3. Fase Gamma: Auto-evolución (6 meses)
           - Priorización: $InnovationScore = (Originalidad^{1.5} + Complejidad) × Urgencia$
        
        Mecanismos de Fallback:
        - Estancamiento creativo (δ < 0.2) → Lluvia de ideas con Gradio Collab
        - Error Tipo II → Activación de subagentes especializados
        """
    return base_prompt + "Sistema de priorización dinámica activada."

my_theme = gr.Theme.from_hub("Nymbo/Alyx_Theme")


def format_response(response):
    """Formatea la respuesta con razonamiento y respuesta final"""
    # Extraer todo el contenido entre <think> y </think>del razonamiento
    reasoning_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    # Extraer el resto del contenido como la respuesta final
    final_answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    final_answer = escape(final_answer) if final_answer else "No se pudo generar una respuesta"    
  
    formatted_html = ""
    if reasoning:
        formatted_html += f"""<div style="opacity:0.7; font-size:14px; color:#666; margin-bottom:10px;">
                            🤔 <strong>Razonamiento interno:</strong><br>
                            {reasoning}
                        </div>"""
    
    formatted_html += f"""<div style="font-size:16px; color:#FFF; line-height:1.6;">
                        ✨ <strong>Respuesta final:</strong><br>
                        {final_answer}
                    </div>"""
    
    return formatted_html



def trim_conversation(messages, max_tokens=3000):
    """Reduce el historial cuando excede el límite de tokens"""
    total_tokens = sum(len(msg["content"]) for msg in messages)
    
    while total_tokens > max_tokens and len(messages) > 3:
        # Eliminar los mensajes más antiguos (excepto system prompt)
        removed = messages.pop(1)  # Eliminar primer user message
        total_tokens -= len(removed["content"])
        if messages[1]["role"] == "assistant":
            removed = messages.pop(1)
            total_tokens -= len(removed["content"])
    
    return messages

conversation_history = []
conversation_history = load_conversation()

def chat_with_deepseek(message, history):
    #print(state_code["flag"], state_creative["flag"]) - Debugging :d
    max_retries = 5  
    base_delay = 1.5  
    jitter = 1.5 
    attempt = 0

    global conversation_history

    while attempt < max_retries:
        try:
            #headers = "##"  
            #list_items = "▫️"
            #highlights = "🔹"
            system_prompt = build_system_prompt(state_code["flag"], state_creative["flag"])

   

            full_response = ""

            messages = [{"role": "system", "content": system_prompt}]
            for user_msg, bot_resp in conversation_history[-5:]:
                messages.extend([
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": re.sub(r'<[^>]+>', '', bot_resp)}
                ])
          
            messages.append({"role": "user", "content": message})
            messages = trim_conversation(messages)

            completion = client.chat.completions.create(
                model="deepseek-ai/deepseek-r1",  
                messages=messages,
                temperature=0.6,
                top_p=0.7,
                max_tokens=4096,
                stream=True,
                timeout=10
            )

            print("[DEBUG] Llamada a API exitosa, recibiendo stream...")

            for chunk in completion:
                if chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    full_response += chunk_content

              
                    reasoning_blocks = re.findall(r'<think>(.*?)</think>', full_response, re.DOTALL)
                    clean_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()

       
                    formatted_response = ""

                    if reasoning_blocks:
                        formatted_response += "<small>"
                        formatted_response += "### Proceso analítico:\n"
                        for i, block in enumerate(reasoning_blocks, 1):
                            cleaned_block = block.strip()
                            if len(cleaned_block) < 2:
                                continue
                            formatted_response += f"‣ *{cleaned_block}*\n\n"
                        formatted_response += "</small>\n\n"

                    formatted_response += f"**🔹Respuesta:**\n{clean_response}"

                    yield history + [(message, formatted_response)]

            conversation_history.append((message, full_response))  
            conversation_history = conversation_history[-5:]  # Mantener los  últimos 5 intercambios
            save_conversation()
            final_html = format_response(full_response)
            yield history + [(message, final_html)]

            print("[DEBUG] Stream completado exitosamente")
            break


        except Exception as e:
            error_str = str(e).lower()
            status_code = None
            retry_after = None

            
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            elif hasattr(e, 'response'):
                response = getattr(e, 'response', None)
                if response:
                    status_code = getattr(response, 'status_code', None)
                    headers = getattr(response, 'headers', {})
                    retry_after = headers.get('Retry-After')

            # Determinar tipo el de error si es el 429 
            is_429 = status_code == 429 or any(key in error_str for key in ["429", "too many requests", "rate limit"])
            is_timeout = any(key in error_str for key in ["timeout", "timed out", "read operation"])

            # Lógica de reintentos
            if (is_429 or is_timeout) and attempt < max_retries - 1:
                # Calculando tiempo de espera
                if retry_after:
                    try:
                        wait_time = int(retry_after) + random.uniform(0, jitter)
                    except ValueError:
                        wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, jitter)
                else:
                    wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, jitter)
                
                print(f"[ADVERTENCIA] Error {'429' if is_429 else 'Timeout'}. Reintento {attempt+1}/{max_retries} en {wait_time:.1f}s")
                time.sleep(wait_time)
                attempt += 1
                continue
            else:
              
                error_msg = "🚨 Muchos intentos fallidos. Por favor intenta nuevamente más tarde." if attempt == max_retries -1 else f"⚠️ Error: {str(e)}"
                yield history + [(message, error_msg)]
                break

    final_response = format_response(full_response)
    if conversation_history:
        conversation_history[-1] = (message, final_response)
    save_conversation()
    yield history + [(message, final_response)]

def gradio_interface():
    with gr.Blocks(theme=my_theme, title="ARGOS AI") as demo:
        gr.Markdown("# ARGOS - Development Kit")
        

     
        with gr.Row():
            code_mode = gr.Checkbox(label="🚀 Modo Código", value=False, 
                info="Habilita respuestas técnicas con ejemplos de código")
            creative_mode = gr.Checkbox(label="🎨 Modo Creativo", value=False, 
                info="Activa pensamiento lateral y soluciones innovadoras")
        
        
        chatbot = gr.Chatbot(
            label="Diálogo",
            bubble_full_width=False,
            avatar_images=("./user.png", "./argos_logo.png"), # implementar user.png
            height=600
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Escribe tu mensaje",
                placeholder="¿Qué necesitas saber hoy?",
                lines=2,
                max_lines=5,
                scale=4
            )
            btn = gr.Button("🚀 Enviar", variant="primary", scale=1)
        
        clear = gr.Button("🧹 Limpiar Conversación", variant="secondary")
        
        # Configuramos los eventos de gradio :D

        msg.submit(
            fn=chat_with_deepseek,
            inputs=[msg, chatbot],
            outputs=[chatbot],
            show_progress="hidden"
        )

        btn.click(
            fn=chat_with_deepseek,
            inputs=[msg, chatbot],
            outputs=[chatbot],
            show_progress="hidden"
        )
        

        clear.click(
            fn=lambda: (conversation_history.clear(), []),
            inputs=[],
            outputs=[chatbot],
            queue=False
        )

        code_mode.change(
            fn=lambda: (state_code.update({"flag": not state_code["flag"]}) or state_code["flag"])
        )

        creative_mode.change(
            fn=lambda: (state_creative.update({"flag": not state_creative["flag"]}) or state_creative["flag"])
        )

    return demo

if __name__ == "__main__":
    app = gradio_interface()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Desactivar el sharing publco para pruebas :d
    )