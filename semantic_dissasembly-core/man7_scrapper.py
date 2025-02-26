import requests
from bs4 import BeautifulSoup
import re
from random import sample, randint, choice
from openai import OpenAI
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

write_lock = threading.Lock()

def generate_training_data(man_data):
    SYSTEM_PROMPT = """Eres un experto en Linux y parsing técnico. Transforma la información de las páginas man en el formato JSON requerido siguiendo estas reglas:

1. Estructura de salida:
{
  "input": "[Consulta técnica realista]",
  "output": {
    "terminos_mapeados": [
      {"termino": "[término técnico]", "confianza": [0.7-0.95]},
      ...
    ],
    "contexto_usuario": {
      "historial": ["[consulta1]", "[consulta2]"],
      "preferencias": "[categoría]"
    },
    "conceptos": [
      {
        "concepto": "[nombre_concepto]",
        "score": [0.6-0.99],
        "comando_relacionado": "[comando válido]",
        "categoria": "[categoría técnica]"
      },
      ...
    ]
  }
}

2. Reglas estrictas:
- Consultas deben ser problemas reales de sysadmins
- Usar solo comandos verificados del man page
- Categorías: mantenimiento, diagnóstico, seguridad, redes, almacenamiento
- Confianza basada en relevancia del término (0.85 para comandos exactos)
- Historial debe ser coherente con la consulta actual
- Preferencias derivadas de la categoría principal
"""

    USER_TEMPLATE = f"""Comando: {man_data['command']}
Título: {man_data['title']}
Opciones válidas: {man_data['options']}
Descripción: {man_data['content'][:500]}

Genera:
1. 3 consultas técnicas realistas con diferentes niveles de experiencia
2. Mapeo preciso de términos técnicos
3. Contexto de usuario coherente
4. Comandos relacionados verificados"""
    try: 
        response = client.chat.completions.create(
            model="deepseek-ai/deepseek-r1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE}
            ],
            temperature=0.3,
            top_p=0.7,
            max_tokens=4096,
            response_format={"type": "json_object"}
        )
        response_content = response.choices[0].message.content


      
        json_match = re.search(r'```json\n([\s\S]*?)\n```', response_content)

        if not json_match:
            print("No se encontró JSON en la respuesta")
            return []

        try:
            data = json.loads(json_match.group(1))
        except json.JSONDecodeError as e:
            print(f"Error decodificando JSON: {str(e)}")
            return []

        # se hace la conversión a JSONL
        jsonl_lines = []
        for entry in data:
            if validate_entry(entry): 
                jsonl_lines.append(
                    json.dumps(convert_to_jsonl_entry(entry), ensure_ascii=False)
                )
        return jsonl_lines
    
    except Exception as e:
        print(f"Error en API: {str(e)}")
        return []



def validate_entry(entry):
    return all(key in entry for key in ["input", "output"])



def convert_to_jsonl_entry(entry):
    return {
        "messages": [
            {
                "role": "system",
                "content": "Eres un modelo especializado en desambiguar y mapear consultas a conceptos técnicos, especialmente en tareas de mantenimiento de sistemas."
            },
            {
                "role": "user",
                "content": entry["input"]
            },
            {
                "role": "assistant",
                "content": json.dumps(entry["output"], ensure_ascii=False)
            }
        ]
    }


OPTION_PATTERN =r'''
(?x)                    # Verbose mode
(?<![\w/])              # Previene match en valores de rutas
(
  --?                   # Soporta - y --
  (?:                   # Estructura principal:
    [a-zA-Z0-9]         # Carácter inicial válido
    [a-zA-Z0-9-]*       # Segmento intermedio
    (?:\=               # Valor opcional:
      (?:               # Tipos de valores:
        [a-zA-Z0-9_@./-]+ |  # Valores simples
        "(?:\\"|[^"])*"      # Valores entre comillas
      )
    )?
  )
)
(?=\s|$|\)|,|>)         # Contexto posterior válido
'''

COMMAND_CONTEXT = {
    'sshfs': {'needs_value': ['-o', '-p'], 'mutually_exclusive': ['-d', '-f']},
    'tset': {'flag_prefixes': ['-e', '-m'], 'value_samples': {'-m': ['vt100', 'xterm']}},
    'pcpintro': {'core_options': ['--host', '--archive'], 'dangerous': []}
}





def process_command(command, index):
    try:
        url = f"https://man7.org/linux/man-pages/man1/{command}.1.html"
        man_data = scrape_man_page_command(url, command)
        if not man_data:
            return

        jsonl_lines = generate_training_data(man_data)
        if not jsonl_lines:
            return

        # Escritura thread-safe :DD
        with write_lock:
            with open("training_dataset.jsonl", "a") as f:
                for line in jsonl_lines:
                    f.write(line + "\n")
        
        # Logging thread-safe d:
        with write_lock:
            print("Index: %s" %  index)
            print(f"Procesado: {command}")
            print(f"Ejemplo: {man_data['example']}")
            print(man_data)
            print("\n")
            
        return True
    except Exception as e:
        with write_lock:
            print(f"Error procesando {command}: {str(e)}")
        return False

#932

def scrape_man_page():
    base_url = "https://man7.org/linux/man-pages/man1/"
    try:
        response = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", href=True)
        commands = list(set(
            link["href"].split(".")[0] 
            for link in links 
            if link["href"].endswith(".1.html")
        ))

        max_workers = min(3, len(commands))
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(process_command, command, commands.index(command)): (command, commands.index(command))
                for command in commands# [925:] #sclicing pq se me acabaron los creditos xd  
            }
            
            for future in as_completed(futures):
                command = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error crítico en {command}: {str(e)}")

    except Exception as e:
        print(f"Error inicial: {str(e)}")



def scrape_man_page_command(url, command):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"[!]Error al obtener la página: {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find("h1").text.strip() if soup.find("h1") else "Sin título"
        
        sections = extract_sections(soup)
        if not sections:
            print("No se encontraron secciones en la página.")
            return None 
            
    
        raw_text = "\n".join([" ".join(content) for content in sections.values()])
        clean_text = re.sub(r"\s{2,}", " ", raw_text).strip()
      
        options = extract_options(clean_text)
        
        example = generate_command_example(command, options[1] if options else [])
        
        return {
            "command": command,
            "title": title,
            "content": clean_text,
            "options": options[0] if options else [],
            "example": example
        }
    except Exception as e:
        print ({"error": f"Error: {str(e)}"})



def extract_options(man_content):
    """Extrae opciones con validación semántica mejorada"""
    matches = re.findall(OPTION_PATTERN, man_content, re.VERBOSE | re.IGNORECASE)
    
 
    valid_options = []
    for opt in set(matches):
       
        if not re.search(r'(^--?[^a-zA-Z0-9]|/{2,}|\\|\.\.)', opt):
            # normalizamos el formato :d
            normalized = re.sub(r'\[.*?\]', '', opt).strip()
            if 2 <= len(normalized) <= 25:
                valid_options.append(normalized)

    sorted_options = sorted(valid_options, key=lambda x: len(x), reverse=True)[:15]
    return sorted_options, list(set([o.split('=')[0] for o in sorted_options]))



def extract_sections(soup):
    sections = {}
    current_section = ""
    for elem in soup.find_all(["h2", "pre", "p", "div"]):
        if elem.name == "h2":
            header_text = elem.get_text(strip=True)
            parts = header_text.split()
            if parts and parts[-1].lower() == "top":
                parts = parts[:-1]
            current_section = "_".join(parts).lower()
            sections[current_section] = []
        elif current_section:
            text = elem.get_text(" ", strip=True)
            if text:
                sections[current_section].append(text)
    return sections


def generate_realistic_value(option):
    """Genera valores contextuales para opciones"""
    value_mapping = {
        '-o': ['allow_other', 'default_permissions', 'reconnect'],
        '-p': ['22', '2222', '443'],
        '--host': ['server01', '192.168.1.100', 'cluster.example.com'],
        '-m': ['xterm', 'vt220', 'ansi'],
        '--archive': ['data.pcp', 'backup_2023.pcp']
    }
    return choice(value_mapping.get(option.split('=')[0], ['VALUE']))


def generate_command_example(command, options):
    """Genera ejemplos realistas con lógica contextual"""
    context_rules = COMMAND_CONTEXT.get(command, {})
    selected = []
    
    for opt in sample(options, min(randint(1,3), len(options))):
        if any(opt.startswith(p) for p in context_rules.get('needs_value', [])):
            selected.append(f"{opt}={generate_realistic_value(opt)}")
        else:
            selected.append(opt)

    example_template = choice([
        f"{command} {{options}}",
        f"{command} {{options}} /path/to/file",
        f"{command} -v {{options}}",
        f"{command} {{options}} --config /etc/default.conf"
    ])
    
    return example_template.format(options=" ".join(selected))


if __name__ == "__main__":
    try: 
        client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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" # <-- NVIDIA API KEY
        )
        scrape_man_page()
    except KeyboardInterrupt:
        print("KeyInterrupt")
