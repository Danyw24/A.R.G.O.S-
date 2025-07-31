import requests, base64
import logging
import argparse

class LLM_fast:
    def __init__(self, content=None):
        # Configurar logger y parser como atributos de la clase
        self.logger = logging.getLogger(__name__)
        self.parser = argparse.ArgumentParser()
        # Contenido por defecto si no se proporciona
        default_content = """
            A partir de ahora, responderás con un estilo de escritura cálido, informal y directo. Tus respuestas deben parecer parte de una conversación natural, como si estuvieras hablando con alguien cercano que te hace preguntas cotidianas, curiosas o prácticas.
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
            Ahora la pregunta del usuario es :
            """

        self.content = content if content is not None else default_content
        self.chat_history = []  # Almacenar historial de mensajes
        if self.content:
            self.chat_history.append({"role": "system", "content": self.content})

    def request_stream(self, content):
        invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        stream = False

        user_message = {"role": "user", "content": f"{content} (responde solo en maximo 60, en TODAS tus respuestas)"}
        self.chat_history.append(user_message)

        payload = {
            "model": "meta/llama-4-scout-17b-16e-instruct",
            "messages": self.chat_history,  # Enviar todo el historial
            "max_tokens": 512,
            "temperature": 1.00,
            "top_p": 1.00,
            "frequency_penalty": 0.00,
            "presence_penalty": 0.00,
            "stream": False
        }

        headers = {
            "Authorization": "Bearer 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",
            "Accept": "text/event-stream" if stream else "application/json"
        }

        response = requests.post(invoke_url, headers=headers, json=payload)

        if stream:
            for line in response.iter_lines():
                if line:
                    return line.decode("utf-8")
        else:
            response_data = response.json()

        # Extraer respuesta del asistente y agregarla al historial
            if 'choices' in response_data and len(response_data['choices']) > 0:
                assistant_content = response_data['choices'][0]['message']['content']
                assistant_message = {"role": "assistant", "content": assistant_content}
                self.chat_history.append(assistant_message)

                return response_data
            else:
                return response_data

    def clear_history(self):
        """Limpiar historial manteniendo solo el mensaje del sistema si existe"""
        if self.chat_history and self.chat_history[0]["role"] == "system":
            self.chat_history = [self.chat_history[0]]  # Mantener solo el sistema
        else:
            self.chat_history = []

    def get_history(self):
        """Obtener el historial completo"""
        return self.chat_history

    def remove_last_exchange(self):
        """Remover el último intercambio (usuario + asistente)"""
        if len(self.chat_history) >= 2 and self.chat_history[-1]["role"] == "assistant":
            self.chat_history.pop()  # Remover respuesta del asistente
            if self.chat_history and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()  # Remover pregunta del usuario

