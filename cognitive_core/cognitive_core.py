from smolagents import ToolCallingAgent, LiteLLMModel, GradioUI
import litellm
import gradio as gr

# 1. Configurar el modelo
model = LiteLLMModel(
    model_id="ollama_chat/deepseek-r1",
    api_base="http://127.0.0.1:5000",
    num_ctx=4096,
    temperature=0.7
)

# 2. Crear y configurar el agente
agent = ToolCallingAgent(
    tools=[],
    model=model,
    max_steps=4  # Límite de pasos para pruebas
)

def ejecutar_agente(pregunta, historial):
    try:
        # Obtener respuesta del agente
        respuesta = agent.run(pregunta)
        
        # Formatear correctamente para Gradio Chatbot
        nueva_entrada = (pregunta, respuesta)
        historial.append(nueva_entrada)
        
        return historial
    
    except Exception as e:
        error_msg = (pregunta, f"Error: {str(e)}")
        historial.append(error_msg)
        return historial

with gr.Blocks(title="Asistente Geográfico") as demo:
    gr.Markdown("## Asistente de Capitales del Mundo")
    
    with gr.Row():
        chatbot = gr.Chatbot(label="Conversación")
        pregunta = gr.Textbox(label="Tu pregunta")
    
    btn_limpiar = gr.Button("Limpiar Chat")
    
    pregunta.submit(
        fn=ejecutar_agente,
        inputs=[pregunta, chatbot],
        outputs=chatbot
    )
    
    btn_limpiar.click(
        fn=lambda: None,
        inputs=None,
        outputs=chatbot,
        queue=False
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )