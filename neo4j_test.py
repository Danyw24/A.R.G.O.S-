"""
the data model in Neo4j. This includes nodes for users, devices,
commands, and colloquial terms, along with their relationships.
For example, a user interacts with devices, devices have statuses,
and colloquial terms map to technical commands.

"""

"""
Cypher Queries:
These should handle term disambiguation, command validation,
context management, and policy enforcement.
Each query serves a specific purpose in the workflow,
like translating user input into technical commands or checking permissions.

"""


"""

WorkFlow de Trabajo de Neo4j
    Usuario->>API: "Apaga el server que hace ruido"
    API->>Neo4j: 1. Desambiguar términos clave
    Neo4j-->>API: {"server": "srv-web-01", "ruido": "cpu_overload"}
    API->>Neo4j: 2. Validar permisos usuario
    Neo4j-->>API: Permisos válidos: power_off
    API->>Neo4j: 3. Obtener estado actual
    Neo4j-->>API: Estado: active (último check 2m)
    API->>GPT-4: Generar comando técnico
    GPT-4-->>API: "ssh admin@srv-web-01 shutdown -h now"
    API->>Neo4j: 4. Registrar comando ejecutado
    API-->>Usuario: "Servidor srv-web-01 apagado"

"""


"""
(:Término {nombre: "servidor"})-[:PUEDE_REFERIRSE_A {confianza: 0.3}]->(:Concepto {nombre: "smtp-proxy-02"})
"""


"""
Aprendizaje De Nuevas Expresiones:

    A[Usuario: "El cluster está temblando"] --> B(Detectar término nuevo)
    B --> C{Existente en Neo4j?}
    C -->|No| D[Registrar en términos pendientes]
    C -->|Sí| E[Reforzar relación existente]
    D --> F[Umbral de 3 repeticiones]
    F -->|Alcanzado| G[Notificar a administrador]
    G --> H[Asignar equivalencia técnica]
    H --> I[Actualizar grafo]

"""


"""
// Consulta optimizada con pesos contextuales
MATCH (u:Usuario {id: $user_id})-[:HISTORIAL]->(c:Consulta)
WITH u, c ORDER BY c.timestamp DESC LIMIT 5
MATCH (c)-[:CONTENIDO]->(t:Término)
MATCH (t)-[e:EQUIVALENTE_A]->(ct:ConceptoTécnico)
WHERE ct.dispositivo = $current_device
RETURN ct.nombre as concepto, 
       avg(e.confianza) * log(u.experiencia) as score
ORDER BY score DESC
"""



"""
// Establecer Relaciones ** 
MATCH (juan:Usuario {id: "usr-001"}), (servidor1:Dispositivo {id: "srv-001"})
CREATE (juan)-[:ACCEDE_A {
    desde: datetime(),
    hasta: datetime() + duration('P30D'),
    frecuencia: 15
}]->(servidor1)

"""




"""
MANUAL DE MÉTODOS Y USO

Este manual describe detalladamente el propósito y la funcionalidad de cada clase y método definido en el código,
así como los parámetros que aceptan. La solución está diseñada para gestionar nodos y relaciones en una base de datos
Neo4j, integrando procesamiento semántico y aprendizaje automático para mapear términos coloquiales a conceptos
técnicos, gestionar dispositivos y usuarios, y actualizar la confianza en las relaciones basadas en el uso.

=====================================================================
CLASES PRINCIPALES
=====================================================================

1. Neo4jConnection
------------------
Propósito:
    - Manejar la conexión a la base de datos Neo4j, ejecutar consultas y gestionar la sesión.

Métodos:
    __init__(uri, user, password):
        - Inicializa la conexión con Neo4j usando el URI, el nombre de usuario y la contraseña.
        - Parámetros:
            * uri (str): Dirección de conexión (por ejemplo, "bolt://localhost:7687").
            * user (str): Nombre de usuario para la autenticación.
            * password (str): Contraseña para la autenticación.
    
    close():
        - Cierra la conexión a Neo4j.
        - No recibe parámetros.
    
    execute_query(query, parameters=None):
        - Ejecuta una consulta Cypher en la base de datos.
        - Parámetros:
            * query (str): Consulta Cypher a ejecutar.
            * parameters (dict, opcional): Diccionario con parámetros para la consulta.
        - Retorna:
            * List[dict]: Lista de diccionarios con los resultados de la consulta.


=====================================================================
2. Neo4jCRUD
------------------
Propósito:
    - Proveer métodos CRUD (crear, leer, actualizar) para interactuar con el grafo en Neo4j.
    - Gestiona la creación de nodos (usuarios, dispositivos, términos coloquiales, conceptos técnicos) y relaciones entre ellos.
    - Integra un modelo de embeddings para realizar búsquedas semánticas.

Constructor:
    __init__(connection, embedding_model):
        - Parámetros:
            * connection (Neo4jConnection): Objeto que maneja la conexión a Neo4j.
            * embedding_model: Modelo (por ejemplo, SentenceTransformer) que genera embeddings para textos.

Métodos:
    crear_termino_coloquial(termino):
        - Crea un nodo de tipo "TerminoColoquial" en el grafo.
        - Parámetros:
            * termino (TerminoColoquialModel): Objeto que contiene datos como id, nombre, uso_count, contexto y embedding.
              Si el embedding no existe, se genera a partir del nombre.
        - Retorna: Lista de diccionarios con el nodo creado.

    buscar_equivalencias_semanticas(query_text, top_n=3):
        - Busca términos coloquiales semánticamente similares al texto proporcionado.
        - Parámetros:
            * query_text (str): Texto de consulta para generar su embedding.
            * top_n (int, opcional): Número máximo de resultados a retornar (por defecto, 3).
        - Internamente, se genera el embedding del texto y se calcula la similitud (cosine) con los embeddings de los nodos.
        - Retorna: Lista de diccionarios con los nombres de los términos y su similitud, ordenados de mayor a menor.

    crear_usuario(usuario):
        - Crea un nodo de tipo "Usuario".
        - Parámetros:
            * usuario (UsuarioModel): Objeto con los datos del usuario (id, nombre, fecha_registro, ultimo_acceso).
        - Retorna: Lista de diccionarios con el nodo creado.

    crear_dispositivo(dispositivo):
        - Crea un nodo de tipo "Dispositivo".
        - Parámetros:
            * dispositivo (DispositivoModel): Objeto con los datos del dispositivo (id, nombre, tipo, ip, estado, ultimo_check).
        - Retorna: Lista de diccionarios con el nodo creado.

    crear_relacion_acceso_dispositivo(usuario_id, dispositivo_id):
        - Establece una relación "ACCEDE_A" entre un nodo Usuario y un nodo Dispositivo.
        - Parámetros:
            * usuario_id (str): Identificador del nodo Usuario.
            * dispositivo_id (str): Identificador del nodo Dispositivo.
        - La relación se crea con propiedades como "desde" (timestamp actual) y "frecuencia" (inicializada a 0).
        - Retorna: Lista de diccionarios con la relación creada.

    crear_concepto_tecnico(concepto):
        - Crea un nodo de tipo "ConceptoTecnico" en el grafo.
        - Parámetros:
            * concepto (ConceptoTecnicoModel): Objeto que contiene los datos del concepto (id, nombre, descripción,
              comando_relacionado, categoría, confianza_base, contexto, ultima_actualizacion).
        - Retorna: Lista de diccionarios con el nodo creado.

    obtener_usuario_por_id(usuario_id):
        - Recupera un nodo Usuario dado su identificador.
        - Parámetros:
            * usuario_id (str): Identificador del nodo Usuario.
        - Retorna: Lista de diccionarios con el nodo Usuario.

    obtener_todos_usuarios():
        - Recupera todos los nodos de tipo Usuario.
        - No recibe parámetros.
        - Retorna: Lista de diccionarios con todos los usuarios.

    obtener_dispositivo_por_id(dispositivo_id):
        - Recupera un nodo Dispositivo por su identificador.
        - Parámetros:
            * dispositivo_id (str): Identificador del nodo Dispositivo.
        - Retorna: Lista de diccionarios con el nodo Dispositivo asociado al su ID.

    obtener_dispositivos_por_estado(estado):
        - Recupera dispositivos que se encuentren en un estado específico.
        - Parámetros:
            * estado (str): Estado por el cual filtrar los dispositivos.
        - Retorna: Lista de diccionarios con los dispositivos que cumplen la condición.

    obtener_conceptos_relacionados(termino, umbral_confianza=0.6):
        - Recupera conceptos técnicos relacionados con un término coloquial, filtrando por un umbral de confianza.
        - Parámetros:
            * termino (str): Nombre del término coloquial.
            * umbral_confianza (float, opcional): Valor mínimo de confianza para considerar la relación (por defecto 0.6).
        - Retorna: Lista de diccionarios con los nombres de los conceptos y la confianza de la relación.

    obtener_historial_usuario(usuario_id, limite=5):
        - Recupera el historial de consultas (nodos Consulta) de un usuario, ordenado por timestamp descendente.
        - Parámetros:
            * usuario_id (str): Identificador del usuario.
            * limite (int, opcional): Número máximo de registros a retornar (por defecto 5).
        - Retorna: Lista de diccionarios con las consultas del historial.

    crear_relacion_equivalencia(termino_id, concepto_id, confianza):
        - Establece o actualiza una relación "EQUIVALENTE_A" entre un nodo TerminoColoquial y un nodo ConceptoTecnico.
        - Parámetros:
            * termino_id (str): Identificador del nodo TerminoColoquial.
            * concepto_id (str): Identificador del nodo ConceptoTecnico.
            * confianza (float): Valor de confianza que se asignará a la relación.
        - Utiliza MERGE para crear la relación si no existe; en cualquier caso, actualiza la propiedad "confianza"
          y "ultima_actualizacion" con la fecha y hora actual.
        - Retorna: Lista de diccionarios con la relación creada o actualizada.

    buscar_conceptos_por_termino(termino_id, umbral=0.6):
        - Busca conceptos técnicos asociados a un término (usando su id), filtrando por un umbral de confianza.
        - Parámetros:
            * termino_id (str): Identificador del nodo TerminoColoquial.
            * umbral (float, opcional): Valor mínimo de confianza para filtrar la relación (por defecto 0.6).
        - Retorna: Lista de diccionarios con el nombre de cada concepto y la puntuación (confianza).


=====================================================================
3. SemanticProcessor
---------------------
Propósito:
    - Procesar la consulta o comando del usuario, realizando desambiguación semántica y validación del contexto.
    - Se apoya en métodos de Neo4jCRUD para buscar términos y obtener el historial del usuario.

Constructor:
    __init__(crud):
        - Parámetros:
            * crud (Neo4jCRUD): Objeto que permite interactuar con la base de datos.

Métodos:
    procesar_consulta(usuario_id, consulta):
        - Procesa la consulta enviada por el usuario.
        - Paso 1: Utiliza 'buscar_equivalencias_semanticas' para desambiguar y mapear la consulta a términos.
        - Paso 2: Obtiene el contexto del usuario mediante su historial (método obtener_contexto_usuario).
        - Parámetros:
            * usuario_id (str): Identificador del usuario.
            * consulta (str): Texto de la consulta o comando.
        - Retorna: Diccionario con claves 'terminos_mapeados', 'dispositivos' (placeholder o resultado de búsqueda adicional)
          y 'contexto_usuario'.

    obtener_contexto_usuario(usuario_id):
        - Recupera el contexto del usuario basado en su historial de consultas.
        - Parámetros:
            * usuario_id (str): Identificador del usuario.
        - Retorna: Lista de diccionarios con información de términos y confianza de relaciones en el historial.

    mapear_consulta_a_conceptos(consulta, contexto_usuario):
        - (En desarrollo o parcialmente implementado) Mapea la consulta a conceptos técnicos relacionados.
        - Paso 1: Busca términos relevantes.
        - Paso 2: Para cada término, obtiene los conceptos técnicos relacionados usando 'buscar_conceptos_por_termino'.
        - Paso 3: (Opcional) Filtra los conceptos basándose en el contexto del usuario.
        - Parámetros:
            * consulta (str): Texto de la consulta.
            * contexto_usuario (dict): Información contextual del usuario.
        - Retorna: Lista ordenada de conceptos técnicos (ordenados por puntaje).

    ordenar_conceptos(conceptos):
        - Ordena la lista de conceptos técnicos por su score (puntuación de confianza) en orden descendente.
        - Parámetros:
            * conceptos (list): Lista de diccionarios con información de cada concepto (al menos la clave 'score').
        - Retorna: Lista ordenada de conceptos.


=====================================================================
4. LearningEngine
---------------------
Propósito:
    - Gestionar el aprendizaje automático para actualizar y reforzar las relaciones semánticas en el grafo.
    - Permite detectar términos similares y ajustar la confianza en las relaciones a medida que se usan.

Constructor:
    __init__(crud):
        - Parámetros:
            * crud (Neo4jCRUD): Objeto para interactuar con la base de datos.

Métodos:
    actualizar_relaciones_semanticas():
        - Detecta términos similares (usando la similitud del embedding con cosine) y crea relaciones
          "SINONIMO" entre ellos si la similitud es mayor a 0.85 y no existe ya tal relación.
        - Retorna: Lista de diccionarios con las relaciones creadas/actualizadas.

    reforzar_aprendizaje(termino, concepto_correcto):
        - Refuerza o crea la relación "EQUIVALENTE_A" entre un término coloquial y un concepto técnico.
        - Si la relación ya existe, incrementa la confianza en 0.1 hasta un máximo de 1.0; si no, la crea
          con una confianza inicial de 0.9 y marca la fuente como 'manual'.
        - Parámetros:
            * termino (str): Nombre del término coloquial.
            * concepto_correcto (str): Nombre del concepto técnico asociado.
        - Retorna: Lista de diccionarios con la relación reforzada.
    
    actualizar_relaciones_automaticamente():
        - Actualiza las relaciones "EQUIVALENTE_A" incrementando la confianza en 0.05 para aquellos términos
          que han sido usados más de 3 veces y cuya confianza es inferior a 0.9.
        - Retorna: Lista de diccionarios con el número de relaciones actualizadas.


=====================================================================
5. Función handle_user_request
---------------------
Propósito:
    - Orquestar el flujo de trabajo al recibir una consulta del usuario.
    - Combina el procesamiento semántico y el aprendizaje automático.
    
Parámetros:
    * user_id (str): Identificador del usuario que envía la consulta.
    * query (str): Texto de la consulta o comando.
    
Funcionamiento:
    - Llama a 'procesar_consulta' del SemanticProcessor para obtener términos mapeados y contexto.
    - Invoca 'actualizar_relaciones_semanticas' del LearningEngine para mejorar la red semántica.
    
Retorna:
    - dict: Resultado del procesamiento, que incluye los términos mapeados, el contexto del usuario y
      (potencialmente) los dispositivos relevantes.

=====================================================================
6. Flujo de Ejecución en el Bloque Principal (__main__)
---------------------
Descripción:
    - Configura la conexión a Neo4j y carga el modelo de embeddings (por ejemplo, 'all-MiniLM-L6-v2').
    - Inicializa las instancias de Neo4jCRUD, SemanticProcessor y LearningEngine.
    - Crea ejemplos de nodos: Conceptos técnicos, Términos coloquiales y relaciones entre ellos.
    - Simula una consulta del usuario y ejecuta el procesamiento semántico.
    - Actualiza las relaciones semánticas automáticamente basándose en el uso y similitud.
    - Finalmente, cierra la conexión a la base de datos.

=====================================================================
NOTAS GENERALES
---------------------
- Cada método que interactúa con Neo4j utiliza consultas Cypher para gestionar nodos y relaciones.
- El modelo de embeddings se utiliza para generar representaciones vectoriales de los textos, permitiendo la
  comparación semántica entre términos.
- La estructura modular permite ampliar y ajustar el sistema conforme se necesiten nuevos métodos o flujos
  de trabajo adicionales.
- Esta documentación sirve tanto para el mantenimiento como para la extensión del sistema, facilitando la
  integración de nuevos módulos o la modificación de la lógica existente.

"""


# En AppConfig
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

load_dotenv(Path(__file__).parent / ".env")


# Modelos Pydantic para validación
class UsuarioModel(BaseModel):
    id: str
    nombre: str
    fecha_registro: datetime = Field(default_factory=datetime.now)
    ultimo_acceso: datetime = Field(default_factory=datetime.now)

class DispositivoModel(BaseModel):
    id: str
    nombre: str
    tipo: str
    ip: str
    estado: str
    ultimo_check: datetime = Field(default_factory=datetime.now)

class ComandoModel(BaseModel):
    id: str
    nombre_tecnico: str
    descripcion: str
    peligrosidad: int

class TerminoColoquialModel(BaseModel):
    id: str
    nombre: str
    uso_count: int = 0
    embedding: List[float] = Field(default_factory=list)
    contexto: List[str] = Field(default_factory=list)  # Para asociar múltiples contextos
    version_embedding: datetime = Field(default_factory=datetime.now)

class ConceptoTecnicoModel(BaseModel):
    id: str
    nombre: str
    descripcion: str
    comando_relacionado: str  # ID del comando técnico
    categoria: str
    confianza_base: float = 0.7  # Valor inicial de confianza
    contexto: List[str] = Field(default_factory=list)
    ultima_actualizacion: datetime = Field(default_factory=datetime.now)



# Configuración de conexión a Neo4j
class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]



# Clases de operaciones CRUD
class Neo4jCRUD:
    def __init__(self, connection: Neo4jConnection, embedding_model):
        self.conn = connection
        self.embedding_model = embedding_model
    
    def crear_termino_coloquial(self, termino: TerminoColoquialModel):
        # Generar embedding solo si no existe 
        if not termino.embedding:
            termino.embedding = self.embedding_model.encode(termino.nombre)
        
        query = """
        CREATE (t:TerminoColoquial {
            id: $id,
            nombre: $nombre,
            uso_count: $uso_count,
            embedding: $embedding,
            contexto: $contexto,
            version_embedding: datetime()
        })
        RETURN t
        """
        return self.conn.execute_query(query, termino.model_dump())

    def crear_usuario(self, usuario: UsuarioModel):
        query = """
        CREATE (u:Usuario {
            id: $id,
            nombre: $nombre,
            fecha_registro: $fecha_registro,
            ultimo_acceso: $ultimo_acceso
        })
        RETURN u
        """
        return self.conn.execute_query(query, usuario.model_dump())
        
    def crear_dispositivo(self, dispositivo: DispositivoModel):
        query = """
        CREATE (d:Dispositivo {
            id: $id,
            nombre: $nombre,
            tipo: $tipo,
            direccion_ip: $ip,
            estado: $estado,
            ultimo_check: $ultimo_check
        })
        RETURN d
        """
        return self.conn.execute_query(query, dispositivo.model_dump())

    def crear_relacion_acceso_dispositivo(self, usuario_id: str, dispositivo_id: str):
        query = """
        MATCH (u:Usuario {id: $usuario_id}), (d:Dispositivo {id: $dispositivo_id})
        CREATE (u)-[r:ACCEDE_A {
            desde: datetime(),
            frecuencia: 0
        }]->(d)
        RETURN r
        """
        return self.conn.execute_query(query, {
            "usuario_id": usuario_id,
            "dispositivo_id": dispositivo_id
        })

    def crear_concepto_tecnico(self, concepto: ConceptoTecnicoModel):
        query = """
        CREATE (ct:ConceptoTecnico {
            id: $id,
            nombre: $nombre,
            descripcion: $descripcion,
            comando_relacionado: $comando_relacionado,
            categoria: $categoria,
                _base: $confianza_base,
            contexto: $contexto,
            ultima_actualizacion: $ultima_actualizacion
        })
        RETURN ct
        """
        return self.conn.execute_query(query, concepto.model_dump())

    def crear_comando_tecnico(self, comando_data: dict):
        """Registra un comando técnico ejecutable"""
        query = """
        CREATE (c:Comando {
            id: $id,
            instruccion: $instruccion,
            dispositivo_target: $dispositivo_target,
            parametros: $parametros,
            seguridad_level: $seguridad_level
        })
        RETURN c
        """
        return self.conn.execute_query(query, comando_data)

    def crear_relacion_equivalencia(self, termino_id: str, concepto_id: str, confianza: float):
        query = """
        MATCH (t:TerminoColoquial {id: $termino_id})
        MATCH (ct:ConceptoTecnico {id: $concepto_id})
        MERGE (t)-[r:EQUIVALENTE_A]->(ct)
        SET r.confianza = $confianza,
            r.ultima_actualizacion = datetime()
        RETURN r
        """
        return self.conn.execute_query(query, {
            "termino_id": termino_id,
            "concepto_id": concepto_id,
            "confianza": confianza
        })

    def obtener_usuario_por_id(self, usuario_id: str):
        query = """
        MATCH (u:Usuario {id: $usuario_id})
        RETURN u
        """
        return self.conn.execute_query(query, {"usuario_id": usuario_id})
    
    def obtener_todos_usuarios(self):
        query = "MATCH (u:Usuario) RETURN u"
        return self.conn.execute_query(query)
    
    def obtener_dispositivo_por_id(self, dispositivo_id: str):
        query = """
        MATCH (d:Dispositivo {id: $dispositivo_id})
        RETURN d
        """
        return self.conn.execute_query(query, {"dispositivo_id": dispositivo_id})
    
    def obtener_dispositivos_por_estado(self, estado: str):
        query = """
        MATCH (d:Dispositivo {estado: $estado})
        RETURN d
        """
        return self.conn.execute_query(query, {"estado": estado})
    
    # Consultas de relaciones semánticas
    def obtener_conceptos_relacionados(self, termino: str, umbral_confianza=0.6):
        query = """
        MATCH (t:TerminoColoquial {nombre: $termino})-[r:EQUIVALENTE_A]->(ct:ConceptoTecnico)
        WHERE r.confianza >= $umbral_confianza
        RETURN ct.nombre AS concepto, r.confianza AS score
        ORDER BY score DESC
        """
        return self.conn.execute_query(query, {
            "termino": termino,
            "umbral_confianza": umbral_confianza
        })
    
    def buscar_conceptos_por_termino(self, termino_id: str, umbral=0.6):
        query = """
        MATCH (t:TerminoColoquial {id: $termino_id})-[r:EQUIVALENTE_A]->(ct:ConceptoTecnico)
        WHERE r.confianza >= $umbral
        RETURN ct.nombre AS concepto, r.confianza AS score
        ORDER BY score DESC
        """
        return self.conn.execute_query(query, {"termino_id": termino_id, "umbral": umbral})

    def buscar_equivalencias_semanticas(self, query_text: str, top_n=3):
        query_embedding = self.embedding_model.encode(query_text)
        
        cypher = """
        MATCH (t:TerminoColoquial)
        WITH t, gds.similarity.cosine(t.embedding, $embedding) AS similarity
        WHERE similarity > 0.75
        RETURN t.nombre AS termino, similarity
        ORDER BY similarity DESC
        LIMIT $top_n
        """
        return self.conn.execute_query(cypher, {
            "embedding": query_embedding,
            "top_n": top_n
        })

        # Actualización dinámica de relaciones
    def actualizar_confianza_relacion(self, termino_id: str, concepto_id: str, delta: float):
        """Ajusta dinámicamente la confianza en las relaciones"""
        query = """
        MATCH (t:TerminoColoquial {id: $termino_id})-[r:EQUIVALENTE_A]->(ct:ConceptoTecnico {id: $concepto_id})
        SET r.confianza = GREATEST(0, LEAST(1, r.confianza + $delta)),
            r.ultima_actualizacion = datetime()
        RETURN r
        """
        return self.conn.execute_query(query, {
            "termino_id": termino_id,
            "concepto_id": concepto_id,
            "delta": delta
        })

    def actualizar_uso_termino(self, termino_id: str):
        """Incrementa el contador de uso de un término"""
        query = """
        MATCH (t:TerminoColoquial {id: $id})
        SET t.uso_count = coalesce(t.uso_count, 0) + 1,
            t.ultima_actualizacion = datetime()
        RETURN t
        """
        return self.conn.execute_query(query, {"id": termino_id})

    # Consultas del grafo de conocimiento
    def obtener_ecosistema_conceptual(self, concepto_id: str, profundidad=2):
        """Obtiene el contexto relacional de un concepto"""
        query = """
        MATCH (ct:ConceptoTecnico {id: $concepto_id})-[*1..%d]-(related)
        RETURN related, labels(related) AS tipo, COUNT(*) AS fuerza_relacional
        ORDER BY fuerza_relacional DESC
        """ % profundidad
        return self.conn.execute_query(query, {"concepto_id": concepto_id})


        # Actualización dinámica de relaciones
    def actualizar_confianza_relacion(self, termino_id: str, concepto_id: str, delta: float):
        """Ajusta dinámicamente la confianza en las relaciones"""
        query = """
        MATCH (t:TerminoColoquial {id: $termino_id})-[r:EQUIVALENTE_A]->(ct:ConceptoTecnico {id: $concepto_id})
        SET r.confianza = GREATEST(0, LEAST(1, r.confianza + $delta)),
            r.ultima_actualizacion = datetime()
        RETURN r
        """
        return self.conn.execute_query(query, {
            "termino_id": termino_id,
            "concepto_id": concepto_id,
            "delta": delta
        })

    
    


class SemanticProcessor:
    def __init__(self, crud: Neo4jCRUD):
        self.crud = crud

    
    """
    Procesa la consulta o comando del usuario utilizando un modelo GPT fine-tuned para
    desambiguar y mapear la consulta a términos y conceptos técnicos. La salida se espera en formato JSON.
    """
    

    """
    Estructura de datos JSON:
    {
  "terminos_mapeados": [
    {
      "termino": "apagar",
      "confianza": 0.85
    },
    {
      "termino": "servidor",
      "confianza": 0.80
    }
  ],
  "contexto_usuario": {
    "historial": [
      "consulta previa 1",
      "consulta previa 2"
    ],
    "preferencias": "mantenimiento"
  },
  "conceptos": [
    {
      "concepto": "apagado_de_servidor",
      "score": 0.90,
      "comando_relacionado": "shutdown now",
      "categoria": "mantenimiento"
    },
    {
      "concepto": "reinicio",
      "score": 0.65,
      "comando_relacionado": "reboot",
      "categoria": "mantenimiento"
    }
  ]
}

    
    """
    def procesar_consulta(self, usuario_id: str, consulta: str):
        """
        Procesa la consulta enviada por el usuario utilizando el modelo GPT fine-tuned.
        
        El proceso consiste en:
          1. Enviar la consulta al modelo GPT fine-tuned.
          2. Recibir la respuesta en formato JSON con la desambiguación de términos y conceptos.
          3. (Opcional) Agregar o complementar con contexto adicional si fuese necesario.

        Parámetros:
            * usuario_id (str): Identificador del usuario (puede utilizarse para personalizar la consulta).
            * consulta (str): Texto de la consulta o comando del usuario.

        Retorna:
            * dict: Diccionario con la respuesta del modelo, que se espera contenga claves como:
                     - 'terminos_mapeados': Términos desambiguados.
                     - 'contexto_usuario': (Opcional) Contexto o historial, si se requiere.
                     - 'conceptos': Conceptos técnicos asociados.
        """
 



class LearningEngine:
    def __init__(self, crud: Neo4jCRUD):
        self.crud = crud

    """
        Detecta términos similares y crea relaciones SINONIMO
        si la similitud es mayor a 0.85.
    """
    def actualizar_relaciones_semanticas(self):
        cypher = """
        MATCH (t1:TerminoColoquial), (t2:TerminoColoquial)
        WHERE t1 <> t2 AND size(t1.embedding) > 0 AND size(t2.embedding) > 0
        WITH t1, t2, gds.similarity.cosine(t1.embedding, t2.embedding) AS sim
        WHERE sim > 0.85 AND NOT EXISTS((t1)-[:SINONIMO]-(t2))
        MERGE (t1)-[r:SINONIMO]->(t2)
        SET r.confianza = sim,
            r.detectado_auto = true,
            r.ultima_actualizacion = datetime()
        """
        return self.crud.conn.execute_query(cypher)


    """
        Refuerza la relación entre un término coloquial y un concepto técnico.
        Si la relación no existe se crea con una confianza inicial de 0.9;
        en caso contrario se incrementa la confianza en 0.1, sin superar 1.0.
    """
    def reforzar_aprendizaje(self, termino: str, concepto_correcto: str):
        cypher = """
        MATCH (t:TerminoColoquial {nombre: $termino})
        MATCH (c:ConceptoTecnico {nombre: $concepto})
        MERGE (t)-[r:EQUIVALENTE_A]->(c)
        ON CREATE SET r.confianza = 0.9, r.fuente = 'manual'
        ON MATCH SET r.confianza = min(r.confianza + 0.1, 1.0)
        """
        return self.crud.conn.execute_query(cypher, {
            "termino": termino,
            "concepto": concepto_correcto
        })

    def actualizar_relaciones_automaticamente(self):
        # Encontrar y actualizar relaciones término-concepto basadas en uso
        query = """
        MATCH (t:TerminoColoquial)-[r:EQUIVALENTE_A]->(ct:ConceptoTecnico)
        WHERE t.uso_count > 3 AND r.confianza < 0.9
        SET r.confianza = r.confianza + 0.05
        RETURN count(r) AS relaciones_actualizadas
        """
        return self.crud.conn.execute_query(query)
    



# Main Entry :D
if __name__ == "__main__":
    # Configurar conexión

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASS = os.getenv("NEO4J_PASS")
   
    neo4j_conn = Neo4jConnection(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASS
    )
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo local ligero
  
    crud = Neo4jCRUD(neo4j_conn, embedding_model)
    processor = SemanticProcessor(crud)
    learning_engine = LearningEngine(crud)

    
    try:

        concepto_tecnico = ConceptoTecnicoModel(
            id="ct-002",
            nombre="apagado_de_seridos",
            descripcion="Apagado completo del servidor",
            comando_relacionado="shutdown now",
            categoria="mantenimiento"
        )
        crud.crear_concepto_tecnico(concepto_tecnico)

        # Crear un término coloquial
        termino = TerminoColoquialModel(
            id="ter-003",
            nombre="desactivar",
            embedding=embedding_model.encode("desactivar")
        )
        crud.crear_termino_coloquial(termino)

        # Establecer la relación entre el término y el concepto con confianza inicial
        crud.crear_relacion_equivalencia("ter-003", "ct-002", 0.75)

    

        print(crud.buscar_equivalencias_semanticas("apagar"))
        

        
        
    finally:
        neo4j_conn.close()