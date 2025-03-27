import os
import json
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Configuración de Flask
app = Flask(__name__)
CORS(app)

# Configuración de la API de Hugging Face
HUGGINGFACE_API_KEY = "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"
HUGGINGFACE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Puedes cambiarlo a otro modelo si lo deseas
API_URL = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Archivo donde se almacenarán los resultados
RESULTADOS_FILE = "resultados.json"

# Función para sintetizar información con Hugging Face
def sintetizar_informacion(texto):
    prompt = f"""
    Eres un experto en historia, ciencia y cultura. 
    - Resume el siguiente texto con precisión, sin omitir información clave. 
    - Enriquece la respuesta con datos adicionales relevantes. 
    - La respuesta debe ser extensa, bien estructurada y en español formal. 
    - Evita repetir el texto original y usa múltiples párrafos. 

    Texto: {texto}
    """

    try:
        payload = {"inputs": prompt, "parameters": {"return_full_text": False, "max_length": 1024}}
        response = requests.post(API_URL, headers=HEADERS, json=payload)

        if response.status_code != 200:
            return f"Error en la API: {response.json()}"

        data = response.json()

        if isinstance(data, list) and "generated_text" in data[0]:
            respuesta = data[0]["generated_text"]
        else:
            respuesta = "Error: No se pudo generar una respuesta válida."

        # Guardar resultado en resultados.json
        guardar_resultado(texto, respuesta)

        return respuesta

    except Exception as e:
        return f"Error: {str(e)}"

# Función para guardar resultados en un archivo JSON
def guardar_resultado(texto, respuesta):
    try:
        if os.path.exists(RESULTADOS_FILE):
            with open(RESULTADOS_FILE, "r", encoding="utf-8") as file:
                datos = json.load(file)
        else:
            datos = []

        # Agregar nueva entrada
        datos.append({"texto_original": texto, "respuesta_generada": respuesta})

        # Guardar en el archivo
        with open(RESULTADOS_FILE, "w", encoding="utf-8") as file:
            json.dump(datos, file, indent=4, ensure_ascii=False)

    except Exception as e:
        print(f"Error al guardar resultado: {e}")

# Ruta principal
@app.route("/")
def index():
    return render_template("index.html")

# Ruta para procesar el texto
@app.route("/procesar", methods=["POST"])
def procesar():
    try:
        datos = request.json
        texto = datos.get("texto", "")

        if not texto:
            return jsonify({"error": "No se recibió texto"}), 400

        respuesta = sintetizar_informacion(texto)

        return jsonify({"respuesta": respuesta})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Iniciar servidor
if __name__ == "__main__":
    app.run(debug=True)
