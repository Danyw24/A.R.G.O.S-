# A.R.G.O.S: Sistema Autónomo de Gestión Cognitiva Avanzada  
**Arquitectura Neuro-Simbólica para Operaciones Adaptativas en Entornos Distribuidos**  

---

## 1. ¿Qué es? 
A.R.G.O.S es un asistente de inteligencia artificial autoevolutivo que integra capacidades de razonamiento estratégico avanzado (mediante LLMs mejorados en vez del uso de los convencionales) con ejecución operativa y de razonamiento distribuida. A.R.G.O.S Combina:  
- **Modelos de lenguaje de última generación** (GPT-4 con modificaciones estructurales, Modelos Razonadores Avanzados (DeepSeek-r1))  
- **Grafos de conocimiento dinámicos** (Neo4j para relaciones complejas)  
- **Memoria contextual multinivel** (Redis, PostgreSQL, Pinecone)  
- **Mecanismos de seguridad Zero-Trust adaptativos**  

Diseñado para operar en entornos tecnológicos dinamicos, el sistema implementa un **ciclo continuo de autooptimización** mediante tres bucles de aprendizaje interdependientes.

---

### 2. Arquitectura Central  

## 2.1 Capas Funcionales  

# Interfaz Neuro-Simbólica Avanzada  

La interfaz neuro-simbólica de A.R.G.O.S implementa un **modelo GPT-4 de baja latencia en tiempo real multimodal**, optimizado para procesar entradas de texto, voz y datos estructurados en tiempo real. Este modelo/núcleo cognitivo está equipado con herramientas especializadas que permiten una interacción fluida y adaptativa con el usuario, combinando capacidades de lenguaje natural con razonamiento lógico programable, de esta manera actuando como intermediario entre el usuario y el **Core** del sistema.  

# 1. Componentes Clave  

## Interfaz Neuro-Simbólica y flujo de interpetación   

### Procesamiento de Consultas  
El flujo inicia con la interpretación de consultas en lenguaje natural (texto o voz). En casos especificos hace uso de modelo GPT3.5 modificado para desambiguación técnica la cual identifica términos clave (ej: "SSH", "logs") y los vincula con comandos específicos. Esta información se enriquece consultando un **grafo de conocimiento en Neo4j que almacena relaciones entre - conceptos técnicos, historial de incidentes, y configuraciones de sistemas**.


### 1.1 Modelo GPT-3.5-turbo-0125 Multimodal Mejorado  

- **Baja Latencia**: Optimizado para respuestas en <820ms  
- **Multimodalidad**: Integración nativa de texo y oputput en formato JSON  
- **Herramientas Especializadas**:  
- **Traductor Semántico**: Convierte lenguaje coloquial en conceptos centrales y comandos técnicos ejecutables  
- **Analizador de Contexto**: Detecta tareas, problemas y preferencias implícitas  
  
**Semantic-core-v1-GPT-3.5-turbo-0125: Arquitectura y Funcionamiento**  
(GPT-3.5, desarrollado por OpenAI, es un modelo de lenguaje basado en la arquitectura *Transformer*, entrenado con técnicas de *autoaprendizaje no supervisado* sobre un corpus diverso que incluye libros, artículos científicos, páginas web y código fuente. Utiliza 175 mil millones de parámetros y mecanismos de atención multi-cabeza para capturar relaciones contextuales a largo plazo. Su funcionamiento se basa en predecir el siguiente token en una secuencia, permitiendo generar respuestas coherentes y contextualizadas), **la razón por la que se usó este modelo en especifico es el bajo costo en el consumo de su API (0.024USD por 1K tokens)** 

**Ejemplo de Input/Output  (Formato JSON, Tomado de set de datos de entrenamiento) :** 
 
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Eres un modelo especializado en desambiguar y mapear consultas a conceptos técnicos, especialmente en tareas de mantenimiento de sistemas."
    },
    {
      "role": "user",
      "content": "Al realizar tracing avanzado de syscalls, ¿cómo listar específicamente los eventos del kernel relacionados con operaciones de red usando LTTng?"
    },
    {
      "role": "assistant",
      "content": {
        "terminos_mapeados": [
          {
            "termino": "syscall",
            "confianza": 0.94
          },
          {
            "termino": "kernel events",
            "confianza": 0.9
          },
          {
            "termino": "--syscall",
            "confianza": 0.93
          }
        ],
        "contexto_usuario": {
          "historial": [
            "análisis de rendimiento de red",
            "configuración de tracing kernel-level"
          ],
          "preferencias": "redes"
        },
        "conceptos": [
          {
            "concepto": "Syscalls de red",
            "score": 0.97,
            "comando_relacionado": "lttng list --kernel --syscall",
            "categoria": "redes"
          },
          {
            "concepto": "Eventos del kernel",
            "score": 0.91,
            "comando_relacionado": "lttng list -k --syscall",
            "categoria": "diagnóstico"
          }
        ]
      }
    }
  ]
}
```

## 1.2 Proceso de modificación y creación de modelo desambiguador

![image](https://github.com/user-attachments/assets/1595e17e-60a5-47a9-9a18-eab6defd7ce0)


**¿Que es "Fine-Tuning?:**
El fine-tuning es una técnica avanzada utilizada para adaptar y modificar modelos preentrenados, como GPT-3.5, a tareas o dominios específicos. Mediante un entrenamiento adicional con datos especializados, el modelo ajusta sus pesos para mejorar su precisión y relevancia en un contexto determinado, equilibrando generalización y especialización.

**Generación de datos sintética**
Los datos de entrenamiento fueron extraidos del manual oficial de linux utilizando **Web Scrapping** y posteriormente procesados con **DeepSeek-r1**, el formato del set de datos está en **.jsonl** (JSON LINES) con esta sintaxis: 
```json

{"messages": [{"role": "system", "content": "Eres un modelo especializado en desambiguar y mapear consultas a conceptos técnicos, especialmente en tareas de mantenimiento de sistemas."}, {"role": "user", "content": "¿Cómo monitorear el uso de CPU histórico en servidores Linux para identificar picos de carga? (Nivel Junior)"}, {"role": "assistant", "content": "{\"terminos_mapeados\": [{\"termino\": \"sar\", \"confianza\": 0.95}, {\"termino\": \"uso de CPU\", \"confianza\": 0.9}, {\"termino\": \"-u\", \"confianza\": 0.88}, {\"termino\": \"métricas básicas\", \"confianza\": 0.78}], \"contexto_usuario\": {\"historial\": [\"uso de top\", \"configuración básica de vmstat\"], \"preferencias\": \"mantenimiento\"}, \"conceptos\": [{\"concepto\": \"Monitorización de rendimiento\", \"score\": 0.92, \"comando_relacionado\": \"sar -u 1 5\", \"categoria\": \"mantenimiento\"}, {\"concepto\": \"Estadísticas de CPU\", \"score\": 0.85, \"comando_relacionado\": \"sar -u ALL\", \"categoria\": \"diagnóstico\"}]}"}]}
{"messages": [{"role": "system", "content": "Eres un modelo especializado en desambiguar y mapear consultas a conceptos técnicos, especialmente en tareas de mantenimiento de sistemas."}, {"role": "user", "content": "Necesito analizar patrones de uso de memoria física y swap durante un incidente de fuga de memoria. (Nivel Intermedio)"}, {"role": "assistant", "content": "{\"terminos_mapeados\": [{\"termino\": \"sar -r\", \"confianza\": 0.93}, {\"termino\": \"paginación\", \"confianza\": 0.82}, {\"termino\": \"-A\", \"confianza\": 0.85}, {\"termino\": \"meminfo\", \"confianza\": 0.75}], \"contexto_usuario\": {\"historial\": [\"monitoreo con free -m\", \"análisis con ps aux\"], \"preferencias\": \"diagnóstico\"}, \"conceptos\": [{\"concepto\": \"Gestión de memoria\", \"score\": 0.89, \"comando_relacionado\": \"sar -r ALL 2 10\", \"categoria\": \"diagnóstico\"}, {\"concepto\": \"Estadísticas de swap\", \"score\": 0.87, \"comando_relacionado\": \"sar -W 1 3\", \"categoria\": \"almacenamiento\"}]}"}]}
{"messages": [{"role": "system", "content": "Eres un modelo especializado en desambiguar y mapear consultas a conceptos técnicos, especialmente en tareas de mantenimiento de sistemas."}, {"role": "user", "content": "¿Cómo correlacionar métricas de red con eventos de pérdida de paquetes usando herramientas de sistema? (Nivel Senior)"}, {"role": "assistant", "content": "{\"terminos_mapeados\": [{\"termino\": \"sar -n DEV\", \"confianza\": 0.94}, {\"termino\": \"interface errors\", \"confianza\": 0.88}, {\"termino\": \"-y\", \"confianza\": 0.83}, {\"termino\": \"estadísticas TCP\", \"confianza\": 0.79}], \"contexto_usuario\": {\"historial\": [\"análisis con tcpdump\", \"monitoreo con netstat -s\"], \"preferencias\": \"redes\"}, \"conceptos\": [{\"concepto\": \"Monitorización de red\", \"score\": 0.91, \"comando_relacionado\": \"sar -n EDEV 1 5\", \"categoria\": \"redes\"}, {\"concepto\": \"Diagnóstico de capa 2\", \"score\": 0.88, \"comando_relacionado\": \"sar -n NFSv3\", \"categoria\": \"seguridad\"}]}"}]}

```

**Entrenamiento**

![image](https://github.com/user-attachments/assets/c4b0f755-85a9-4cd0-9065-7aad03932918)

El entrenamiento del modelo GPT-3.5-turbo-0125 para la desambiguación de terminos complejos se basa en un fine-tuning supervisado, ajustando sus pesos para optimizar la identificación de términos clave y la generación de respuestas contextualizadas. Durante este proceso, se aplican técnicas de **Transfer Learning** para preservar el conocimiento general del modelo mientras se especializa en interpretar consultas complejas y vincular conceptos con comandos ejecutables. Además, el modelo integra un sistema de consulta en Neo4j, permitiéndole enriquecer sus respuestas mediante la exploración de grafos de conocimiento, mejorando la precisión en configuraciones y relaciones técnicas. Como resultado, se obtiene un modelo de baja latencia (<820ms) capaz de generar respuestas en formato JSON, combinando análisis semántico con conocimiento estructurado para una asistencia técnica más eficiente y una **comprensión del lenguaje y el usuario mas profunda que los modelos genericos**.



### 1.2 Interfaz Neuro-Simbólica  
- **Aprendizaje Adaptativo**:  
  - Captura patrones lingüísticos del usuario mediante **embeddings dinámicos**  
  - Ajusta el nivel de formalidad y detalle técnico en tiempo real  
  - Aprende de correcciones implícitas (ej: reformulaciones del usuario)  
- **Traducción Técnica**:  
  - Mapea expresiones cotidianas a conceptos técnicos mediante grafos de conocimiento  
  - Genera pseudocódigo intermedio antes de la ejecución  
  - Valida intenciones contra restricciones operativas  

### 1.3 Grafo de Conocimiento (Neo4j)  
- **Estructura Relacional**:  
  - Nodos: Términos técnicos, conceptos, dispositivos y usuarios  
  - Relaciones: Equivalencias semánticas, dependencias operativas, patrones de uso  
- **Funcionalidades Clave**:  
  - Búsqueda contextual en O(1) mediante traversales optimizados  
  - Actualización dinámica basada en interacciones recientes  
  - Detección de patrones emergentes mediante análisis de grafos  


---

## 2. Uso de Embeddings  

### 2.1 Implementación Actual  
- **Modelo de Embeddings**:  
  - Basado en arquitectura **Transformer (1536 dimensiones)**  
  - Entrenado con datos técnicos y coloquiales para capturar matices semánticos  
- **Aplicaciones Clave**:  
  - **Búsqueda Semántica**: Encuentra conceptos relacionados en el grafo de conocimiento  
  - **Clasificación de Intenciones**: Asigna categorías a entradas del usuario  
  - **Detección de Similitudes**: Identifica patrones en interacciones pasadas  

### 2.2 Proceso de Generación  
1. **Extracción de Características**:  
   - Texto → Tokenización → Embedding inicial  
2. **Refinamiento Contextual**:  
   - Ajuste dinámico basado en el historial del usuario  

### 2.3 Ejemplo Práctico  
**Entrada del Usuario**:  
*"El servidor está lento otra vez"*  

**Proceso con Embeddings**:  
1. Genera embedding para "servidor lento"  
2. Busca similitudes en el grafo de conocimiento  
3. Encuentra relación con "alto uso de CPU"  
4. Sugiere solución: *"Parece que el uso de CPU está al 95%. ¿Quieres reiniciar el servicio crítico?"*  

---

## 3. Flujo de Operación  

1. **Entrada Multimodal**:  
   - El usuario interactúa mediante texto, voz o selección de opciones  
   - El sistema detecta el modo preferido y ajusta su respuesta  

2. **Procesamiento Neuro-Simbólico**:  
   - Análisis semántico con GPT-4 mejorado  
   - Validación lógica mediante reglas programables  
   - Mapeo a conceptos técnicos usando el grafo de conocimiento  

3. **Generación de Respuesta**:  
   - Adaptación al nivel técnico del usuario  
   - Inclusión de contexto relevante (historial, preferencias)  
   - Verificación de coherencia antes de la entrega  

---

## 4. Ventajas Clave  

- **Personalización Profunda**:  
  - Aprende del estilo de comunicación del usuario  
  - Ajusta el nivel de detalle técnico automáticamente  
- **Traducción Precisa**:  
  - Convierte intenciones vagas en comandos ejecutables 