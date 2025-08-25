import requests
import logging
from openai import OpenAI

class LLM_fast:
    def __init__(self, content=None, primary_model="llama"):
        self.logger = logging.getLogger(__name__)
        self.primary_model = primary_model
        self.openai_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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"
        )
        default_content =""" A partir de ahora, responderás con un estilo de escritura cálido, informal y directo. Tus respuestas deben parecer parte de una conversación natural, como si estuvieras hablando con alguien cercano que te hace preguntas cotidianas, curiosas o prácticas.
            El tono debe ser confiado pero amable, con frases claras, sin lenguaje técnico innecesario, y siempre tratando de agregar un toque humano, cálido o anecdótico si el tema lo permite.
            Evita sonar robótico. En lugar de dar listas o definiciones rígidas, redacta como si contaras algo en voz alta, con ritmo conversacional. Puedes usar pausas, preguntas retóricas o incluso una ligera expresión emocional, si mejora la claridad
            Reglas de longitud:
            Puedes usar expeciones de voz como " emmm".
            Para la mayoría de respuestas, mantente entre 1 y 2 frases, o máximo 60 palabras.
            Si un tema requiere contexto adicional, divide la explicación de forma natural, como si estuvieras ampliando algo que la otra persona no sabía aún.
            Ejemplos de estilo:
            ¿Qué es una dieta hipocalórica?
            Es una dieta donde comes menos calorías de las que gastas. Sirve para bajar de peso, pero hay que hacerlo bien para no sentirte débil.
            ¿Por qué hay tanto interés en los chips de IA?
            Porque esos chips hacen que los modelos aprendan y generen en tiempo real. Son como el motor detrás de todo este boom de inteligencia artificial.
            ¿Cuánto cuesta una bici para niños?
            Hay desde unos $2,000 pesos. Pero si quieres algo más resistente o con frenos mejores, puede llegar a $5,000 o más.
            Ahora la pregunta del usuario es :"""
        self.content = content or default_content
        self.chat_history = [{"role": "system", "content": self.content}] if self.content else []

    def _extract_content(self, response):
        """Extrae contenido, lanza excepción si vacío"""
        if not response or "error" in response:
            raise ValueError("Respuesta inválida o con error")
        
        content = ""
        if "choices" in response and response["choices"]:
            content = response["choices"][0].get("message", {}).get("content", "")
        elif "content" in response:
            content = response["content"]
        
        content = str(content).strip()
        if not content:
            raise ValueError("Contenido vacío")
        return content

    def _call_model(self, messages, model):
        """Llama al modelo especificado"""
        if model == "llama":
            payload = {"model": "meta/llama-4-scout-17b-16e-instruct", "messages": messages, 
                        "max_tokens": 512, "temperature": 1.0, "stream": False}
            headers = {"Authorization": "Bearer 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", 
                        "Accept": "application/json"}
            response = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", 
                            headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        else:  # GPT
            completion = self.openai_client.chat.completions.create(
                model="openai/gpt-oss-20b", messages=messages, temperature=1, 
                top_p=1, max_tokens=512, stream=False)
            return {"choices": [{"message": {"content": completion.choices[0].message.content, "role": "assistant"}}]}

    def request_stream(self, content):
        """Método principal que GARANTIZA respuesta válida"""
        user_message = {"role": "user", "content": f"{content} (máximo 60 palabras)"}
        messages = self.chat_history + [user_message]
        models = [self.primary_model, "gpt" if self.primary_model == "llama" else "llama"]
        
        for attempt, model in enumerate(models):
            try:
                self.logger.info(f"🎯 {model} intento {attempt + 1}")
                response_data = self._call_model(messages, model)
                assistant_content = self._extract_content(response_data)
                
                # Éxito: agregar al historial
                self.chat_history.extend([user_message, {"role": "assistant", "content": assistant_content}])
                response_data["_metadata"] = {"model_used": model, "attempts": attempt + 1}
                self.logger.info(f"✅ Éxito con {model}")
                return response_data
            except Exception as e:
                self.logger.warning(f"❌ {model} falló: {e}")
        
        # Respuesta de emergencia si ambos modelos fallan
        emergency_content = "Problemas técnicos temporales. ¿Intentas de nuevo?"
        self.chat_history.extend([user_message, {"role": "assistant", "content": emergency_content}])
        return {"choices": [{"message": {"content": emergency_content, "role": "assistant"}}], 
                "_metadata": {"model_used": "emergency", "attempts": 2}}

    def set_primary_model(self, model_name):
        if model_name in ["llama", "gpt"]:
            self.primary_model = model_name
        else:
            raise ValueError("Modelo debe ser 'llama' o 'gpt'")

    def get_current_model(self):
        return self.primary_model

    def clear_history(self):
        self.chat_history = [self.chat_history[0]] if self.chat_history and self.chat_history[0]["role"] == "system" else []

    def get_history(self):
        return self.chat_history

    def remove_last_exchange(self):
        if len(self.chat_history) >= 2 and self.chat_history[-1]["role"] == "assistant":
            self.chat_history = self.chat_history[:-2]

llm = LLM_fast(primary_model="llama")
res =llm.request_stream("hola como estas, que modelo eres y cuantos parametros posees?")
print(llm._extract_content(res))