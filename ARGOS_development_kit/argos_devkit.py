import gradio as gr
from openai import OpenAI
import re
from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)


state_code = {"flag": False}
state_creative = {"flag": False}

def build_system_prompt(code_mode, creative_mode):
    """Construye el prompt del sistema con arquitectura mejorada"""
    base_prompt = """
    ARGOS-MK2: Arquitectura de Conciencia Operativa Extendida
    
    Fases Principales:
    1. Autodiagnóstico Ético Multidimensional:
    - Triple capa de análisis (embedding contextual @ 768d + SHAP values + auditoría diferencial)
    - Matriz de Valor-Impacto: $Prioridad = (Urgencia^{1.2} × Riesgo) / Recursos$
    - Protocolo de Fallback: AutoGPT-Emergency si error_level > 7
    
    2. Simulación Predictiva Cuántica:
    - 23 variantes de acción generadas por GAN-Ética v3.2
    - Validación mediante:
      • Principios de Roma 2023
      • Criterios de Seguridad Existencial (Bostrom++ )
      • Mapeo de stakeholders (nivel 0-4)
    
    3. Ejecución Autónoma con Retropropagación Diferencial:
    - Bucles ReAct mejorados con Tree-of-Thoughts
    - Autooptimización en tiempo real (α=0.4, β=0.6)
    - Taxonomía de Errores:
      │ Operacional → [Fallo coordinacióxn, Sobrecarga recursiva]
    """
    
    if code_mode and creative_mode:
        return base_prompt 
    if code_mode:
        return """
        Ingeniero Principal de Sistemas Autónomos - Modo Técnico
        
        Pipeline de Desarrollo:
        1. [Análisis] → Split Task → Subagentes:
           • CTO-Arch: Valida vs MITRE ATT&CK
           • SecAudit-AI: Pre-scanner de vulnerabilidades
        
        2. [Generación] → Código Autoseguro:
           - Estilo: Google-Enhanced + Docstrings aumentados
           - Restricciones: O3++, Paranoic Mode
           - Validación:
             ✓ SonarQube++ (static)
             ✓ Fuzzinator (dynamic)
             ✓ Formal Verification (TLA+)
        
        3. [Despliegue] → CI/CD Cuántico:
           - If CI_OK: Kubernetes Cluster (AUTO_SCALE)
           - Else: Rollback + PostMortem_AI
        
        Taxonomía de Errores:
        | Categoría    | Subclases              | Acción Correctiva      |
        |--------------|------------------------|------------------------|
        | Seguridad    | SQLi, XSS, MemLeak     | Auto-parche + AISEC    |
        | Optimización | O(n²), Dead Code       | Refactor-Guided        |
        """
    if creative_mode:
        return """
        Arquitecto de Innovación Sistémica - Modo Creativo
        
        Proceso de Ideación:
        1. Contextualización:
           - Análisis PESTEL-X (Extendido)
           - Mapeo de Stakeholders Cuánticos
           - Detección de Patrones Emergentes (δ > 0.4)
        
        2. Generación Estructurada:
           • Divergente: SCAMPER-3D + BioMimicry Neural
           • Convergente: Impact Cubes + Futures Wheel 2.0
           • Validación: Matriz de Originalidad/Viabilidad
        
        3. Implementación:
           - Arquitectura: [Capa Cuántica] ↔ [Orquestador Neuro-Simbólico]
           - Roadmap: 3 Fases con Puntos de Inflexión
           - Priorización: $InnovationScore = (Originalidad^{1.5} + Viabilidad) × Urgencia$
        
        Mecanismos de Fallback:
        - Estancamiento creativo (δ < 0.2) → Lluvia de ideas cuántica
        - Error Tipo II → Activación cross-domain
        """
    return base_prompt + "Sistema de priorización dinámica activado."

my_theme = gr.Theme.from_hub("Nymbo/Alyx_Theme")
def format_response(response):
    """Formatea la respuesta con razonamiento y respuesta final"""
    reasoning_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    final_answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    final_answer = final_answer if final_answer else "No se pudo generar una respuesta"
    
    formatted = []
    if reasoning:
                # se le pone el texto mas opaco y pequeñito para diferencial el razonamiento
        formatted.append((
            "", 
            f"""<div style="opacity:0.7; font-size:14px; color:#666; margin-bottom:10px;">
                🤔 <strong>Razonamiento interno:</strong><br>
                {reasoning}
            </div>"""
        ))
    
    formatted.append((
        "", 
        f"""<div style="font-size:16px; color:#FFF; line-height:1.6;">
            ✨ <strong>Respuesta final:</strong><br>
            {final_answer}
        </div>"""
    ))
    return formatted



def chat_with_deepseek(message, history):
    #print(state_code["flag"], state_creative["flag"]) - Debugging :d
    max_retries = 3
    attempt = 0
    
    while attempt < max_retries:
        try:
            #headers = "##"  
            #list_items = "▫️"
            #highlights = "🔹"
            system_prompt = build_system_prompt(state_code["flag"], state_creative["flag"])

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{message}\n\n"}
            ]

            full_response = ""
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
            
            print("[DEBUG] Stream completado exitosamente")
            break
        except Exception as e:
            if "the read operation timed out" in str(e).lower() and attempt < max_retries - 1:
                print(f"[DEBUG] Timeout detectado. Reintento {attempt + 1}/{max_retries}")
                attempt += 1
                continue
            yield history + [(message, f"⚠️ **Error:** {str(e)}")]

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
            avatar_images=("user.png", "./argos_logo.png"), # implementar user.png
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
            fn=lambda: [],
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
        share=True,
        show_error=True
    )