"""
Sistema de Gestión Semántica para Mantenimiento Técnico Autónomo

Funcionamiento:
---------------------------------------------------------------
Arquitectura cognitiva que integra:
1. Base de conocimiento en grafo (Neo4j) con:
   - Términos coloquiales (TerminoColoquial) y sus embeddings
   - Conceptos técnicos (ConceptoTecnico) y comandos asociados
   - Relaciones semánticas dinámicas (confianza, frecuencia)
2. Motor de procesamiento semántico que:
   - Mapea consultas naturales a comandos técnicos usando GPT fine-tuned
   - Genera relaciones contextuales usando modelos de embeddings (all-MiniLM-L6-v2)
3. Subsistema de auto-aprendizaje que:
   - Actualiza pesos relacionales basado en uso
   - Detecta sinónimos mediante similitud vectorial
   - Ajusta confianzas de asociación término-concepto

Guía de Uso:
---------------------------------------------------------------
1. Configurar variables de entorno (.env):
   - Neo4j: NEO4J_URI, NEO4J_USER, NEO4J_PASS
   - OpenAI: OPENAI_API_KEY

2. Operaciones principales:
   - Crear entidades: crear_termino(), crear_concepto_tecnico()
   - Procesar consultas: enviar_consulta(query_text)
   - Actualizar conocimiento: actualizar_relaciones_semanticas()

3. Ejemplo de flujo:
   consulta = "Cómo monitorear red"
   resultado = processor.enviar_consulta(consulta)
   crear_concepto(client, consulta)

Aporte al Proyecto Argos:
---------------------------------------------------------------
Capa fundamental para:
- Interpretación contextual de requerimientos técnicos
- Traducción automática lenguaje natural → comandos ejecutables
- Mantenimiento autónomo de base de conocimiento especializado
- Detección proactiva de patrones de uso y relaciones implícitas

Estructura clave:
- TerminoColoquial: Términos del usuario con embeddings semánticos
- ConceptoTecnico: Entidades técnicas con comandos asociados
- Relaciones ASOCIADO_A: Conexiones dinámicas con metadatos de confianza
- Categorías: Clasificación contextual para gestión de conocimiento
"""







from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
import json
import re 
from openai import OpenAI
import time
from json_repair import repair_json
from jsonschema import validate
import numpy as np 


load_dotenv(Path(__file__).parent / ".env")


def validate_entry(entry):
    return all(key in entry for key in ["input", "output"])


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

class CategoriaModel(BaseModel):
    id:str
    categoria_name: str




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
    
    def crear_termino(self, termino: TerminoColoquialModel):
        # Generar embedding solo si no existe 
        if not termino.embedding:
            termino.embedding = self.embedding_model.encode(termino.nombre).tolist()
        
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

    def crear_categoria(self, categoria: CategoriaModel):
        query = """
        CREATE (c:Categoria {
            id: $id,
            categoria: $categoria_name
        })
        RETURN c
        """
        return self.conn.execute_query(query, categoria.model_dump())


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

    def crear_relacion_categoria(self, concepto_name: str, categoria_name: str):
        query = """
        MATCH (c:ConceptoTecnico {nombre: $concepto_name}), (d:Categoria {categoria: $categoria_name})
        CREATE (c)-[r:ASOCIADO_A{
            desde: datetime(),
            frecuencia: 0
        }]->(d)
        RETURN r
        """
        return self.conn.execute_query(query, {
            "concepto_name": concepto_name,
            "categoria_name": categoria_name
        })


    def crear_relacion_termino(self, termino_name: str, concepto_name: str):
        query = """
        MERGE (t:TerminoColoquial {nombre: $termino_name})
        WITH t
        MATCH (c:ConceptoTecnico {nombre: $concepto_name})
        MERGE (t)-[r:ASOCIADO_A]->(c)
        ON CREATE SET
            r.desde = datetime(),
            r.frecuencia = 1
        ON MATCH SET
            r.frecuencia = r.frecuencia + 1
        RETURN r
        """
        return self.conn.execute_query(query, {
            "termino_name": termino_name,
            "concepto_name": concepto_name
        })

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


    def obtener_usuario_por_id(self, usuario_id: str):
        query = """
        MATCH (u:Usuario {id: $usuario_id})
        RETURN u
        """
        return self.conn.execute_query(query, {"usuario_id": usuario_id})


    def obtener_concepto_por_nombre(self, nombre: str):
        query = """
        MATCH (u:ConceptoTecnico {nombre: $nombre})
        RETURN u
        """
        return self.conn.execute_query(query, {"nombre": nombre})
    
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
    def obtener_conceptos_relacionados(self, termino: str):
        query = """
        MATCH (t:TerminoColoquial {nombre: $termino})-[r:ASOCIADO_A]->(ct:ConceptoTecnico)
        RETURN ct.nombre AS concepto
        """
        return self.conn.execute_query(query, {
            "termino": termino,
        })
    

    def buscar_equivalencias_semanticas(self, query_text: str, top_n=3):
        query_embedding = self.embedding_model.encode(query_text)
        
        cypher = """
        MATCH (t:TerminoColoquial)
        WITH t, gds.similarity.cosine(t.embedding, $embedding) AS similarity
        WHERE similarity > 0.68
        RETURN t.nombre AS termino, similarity
        ORDER BY similarity DESC
        LIMIT $top_n
        """
        return self.conn.execute_query(cypher, {
            "embedding": query_embedding,
            "top_n": top_n
        })

    def categoria_existe(self, nombre_categoria: str) -> bool:
        query = """
        RETURN EXISTS {
            MATCH (c:Categoria {categoria: $categoria_name})
        } AS existe
        """
        resultado = self.conn.execute_query(query, {"categoria_name": nombre_categoria})
        return resultado[0]['existe'] if resultado else False


    def termino_existe(self, nombre_termino: str) -> bool:
        query = """
        RETURN EXISTS {
            MATCH (t:TerminoColoquial {nombre: $termino_name})
        } AS existe
        """
        resultado = self.conn.execute_query(query, {"termino_name": nombre_termino})
        return resultado[0]['existe'] if resultado else False



    def actualizar_contador_termino(self, nombre_termino:str):
        query = """
        MATCH (t:TerminoColoquial {nombre: $termino_name})
        SET t.uso_count = t.uso_count + 1
        RETURN t
        """
        return self.conn.execute_query(query, {
            "termino_name": nombre_termino
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
   
   
class SemanticProcessor:
    def __init__(self, crud: Neo4jCRUD):
        self.crud = crud
    
    """
    Procesa la consulta o comando del usuario utilizando un modelo GPT fine-tuned para
    desambiguar y mapear la consulta a términos y conceptos técnicos. La salida se espera en formato JSON.
    """

    def enviar_consulta(self, client, request_content="",max_intentos=3):
        SYSTEM_CONTENT_PROMPT="""
        Eres un asistente técnico especializado en desambiguar y mapear consultas a conceptos técnicos, especialmente en tareas de mantenimiento de sistemas. Tu tarea es generar respuestas en formato JSON válido, siguiendo estrictamente esta estructura:

        {
          "terminos_mapeados": [
            {"termino": "término_1", "confianza": 0.95},
            {"termino": "término_2", "confianza": 0.85}
          ],
          "contexto_usuario": {
            "historial": ["consulta_anterior_1", "consulta_anterior_2"],
            "preferencias": "categoría_relevante"
          },
          "conceptos": [
            {
              "concepto": "nombre_del_concepto",
              "score": 0.92,
              "comando_relacionado": "comando_asociado",
              "categoria": "categoría_del_concepto"
            }
          ]
        }

        Reglas estrictas:
        1. **terminos_mapeados**: Debe contener al menos 2 términos técnicos relevantes para la consulta, con un valor de confianza entre 0.1 y 1.0.
        2. **contexto_usuario**: 
           - "historial": Lista de 2 consultas anteriores relacionadas.
           - "preferencias": Categoría principal de la consulta (ej: redes, seguridad, desarrollo).
        3. **conceptos**: 
           - Debe incluir al menos 2 conceptos técnicos.
           - Cada concepto debe tener un "score" entre 0.7 y 1.0.
           - El "comando_relacionado" debe ser un comando ejecutable en Linux o sistemas relacionados.
           - La "categoria" debe ser una de: redes, seguridad, desarrollo, mantenimiento, diagnóstico.

        Ejemplo de salida esperada para la consulta "¿Cómo listar archivos en /home?":
        {
          "terminos_mapeados": [
            {"termino": "ls", "confianza": 0.95},
            {"termino": "directorio_home", "confianza": 0.85}
          ],
          "contexto_usuario": {
            "historial": ["cómo crear un archivo en Linux", "cómo cambiar permisos en Linux"],
            "preferencias": "mantenimiento"
          },
          "conceptos": [
            {
              "concepto": "Listado de archivos",
              "score": 0.92,
              "comando_relacionado": "ls /home",
              "categoria": "mantenimiento"
            },
            {
              "concepto": "Navegación de directorios",
              "score": 0.88,
              "comando_relacionado": "cd



        """

        intentos = 0
        while intentos < max_intentos:
            try:
                # Configuración optimizada del request
                response = client.chat.completions.create(
                    model="ft:gpt-3.5-turbo-0125:argos-project:semantic-core-v1:Az6nS1aO",
                    messages=[
                        {"role": "system", "content": SYSTEM_CONTENT_PROMPT},
                        {"role": "user", "content": request_content}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content
                
                # Reparación avanzada de JSON
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    data = json.loads(repair_json(content, skip_json_loads=True))
                    
                # Validación estructural con esquema


                schema = {
                "type": "object",
                "required": ["terminos_mapeados", "contexto_usuario", "conceptos"],
                "properties": {
                    "terminos_mapeados": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["termino", "confianza"],
                            "properties": {
                                "termino": {"type": "string"},
                                "confianza": {"type": "number"}
                            }
                        }
                    },
                    "contexto_usuario": {
                        "type": "object",
                        "required": ["historial", "preferencias"],
                        "properties": {
                            "historial": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "preferencias": {"type": "string"}
                        }
                    },
                    "conceptos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["concepto", "score", "comando_relacionado", "categoria"],
                            "properties": {
                                "concepto": {"type": "string"},
                                "score": {"type": "number"},
                                "comando_relacionado": {"type": "string"},
                                "categoria": {"type": "string"}
                            }
                        }
                    }
                }
            }
            
              
                validate(instance=data, schema=schema)

                return {
                    "terminos_mapeados": data["terminos_mapeados"],
                    "contexto_usuario": data["contexto_usuario"],
                    "conceptos": [
                        {
                            "concepto": term["concepto"],
                            "score": round(term["score"], 2),
                            "comando_relacionado": term["comando_relacionado"],
                            "categoria": term["categoria"]
                        }
                        for term in data["conceptos"]
                        if all(key in term for key in ["concepto", "score", "comando_relacionado", "categoria"])
                    ]
                }
            
                
            except json.JSONDecodeError as e:
                print(f"Error JSON (intento {intentos+1}): {str(e)}")
                # Extracción con regex mejorado
                if conceptos := re.findall(
                    r'"concepto"\s*:\s*"((?:\\"|[^"]*)+)"\s*,\s*"score"\s*:\s*([0-9.]+)',
                    content
                ):
                    return [{"concepto": c[0], "score": float(c[1])} for c in conceptos]
                
            except Exception as e:
                print(f"Error ({intentos+1}/{max_intentos}): {type(e).__name__} - {str(e)}")
                time.sleep(2 ** intentos)
                intentos += 1
    
        raise Exception("Falló después de múltiples intentos")



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




class LearningEngine:
    def __init__(self, crud: Neo4jCRUD):
        self.crud = crud

    """
        Detecta términos similares y crea relaciones SINONIMO
        si la similitud es mayor a 0.80.
    """
    def actualizar_relaciones_semanticas(self):
        cypher = """
        MATCH (t1:TerminoColoquial), (t2:TerminoColoquial)
        WHERE t1 <> t2 AND size(t1.embedding) > 0 AND size(t2.embedding) > 0
        WITH t1, t2, gds.similarity.cosine(t1.embedding, t2.embedding) AS sim
        WHERE sim > 0.70 AND NOT EXISTS((t1)-[:SINONIMO]-(t2))
        MERGE (t1)-[r:SINONIMO]->(t2)
        SET r.confianza = sim,
            r.detectado_auto = true,
            r.ultima_actualizacion = datetime()
        """
        return self.crud.conn.execute_query(cypher)


    def actualizar_confianza_relacion(self):
        # Encontrar y actualizar relaciones término-concepto basadas en uso
        query = """
        MATCH (t:TerminoColoquial)-[r:ASOCIADO_A]->(ct:ConceptoTecnico)
        WHERE t.uso_count > 3 AND r.confianza < 0.9
        SET r.confianza = r.confianza + 0.05
        RETURN count(r) AS relaciones_actualizadas
        """
        return self.crud.conn.execute_query(query)


def crear_concepto(client, query):

    data = processor.enviar_consulta(client, query) 
    print("\n\n Full data %s" % data)
    concepto_name_1 = data["conceptos"][0]["concepto"]
    comando_relacionado_name = data["conceptos"][0]["comando_relacionado"]
    categoria_name = data["conceptos"][0]["categoria"]
    score_confianza = data["conceptos"][0]["score"]
    historial = data["contexto_usuario"]["historial"]
    termino_name = max(data["terminos_mapeados"], key=lambda x: x["confianza"])["termino"]
    print(termino_name) 
    if not crud.obtener_concepto_por_nombre(concepto_name_1):
        concepto = ConceptoTecnicoModel(
            id ="con-001",
            nombre=concepto_name_1,
            descripcion=concepto_name_1,
            comando_relacionado=comando_relacionado_name,
            categoria=categoria_name,
            confianza_base=score_confianza,
            contexto=historial
        )
        # crear termino y relacionalo con el concepto , si el termino ya existe, no crearlo y conectarlo a ese termino y aumentar su contador de uso
        termino = TerminoColoquialModel(
            id="ter-001",
            nombre=termino_name,
            uso_count=0,
            contexto = historial,
        )
        
        crud.crear_concepto_tecnico(concepto)
        #buscar si existe la categoria, si no, crearla
        if not crud.categoria_existe(categoria_name):
            categoria_model = CategoriaModel(
                id ="cat-00" + str("1"),
                categoria_name=str(categoria_name)
            )
            crud.crear_categoria(categoria_model)
        crud.crear_relacion_categoria(concepto_name_1, categoria_name) # self, concepto_name: str, categoria_name: str
        if not crud.termino_existe(termino_name):
            crud.crear_termino(termino) 
        crud.actualizar_contador_termino(termino_name)
        crud.crear_relacion_termino(termino_name, concepto_name_1)
       

    else:
        print("concepto nombre %s ya creado" % concepto_name_1)
        




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
    client = OpenAI(
        organization='org-QUNtEDSO2BkGHAs4NCoCh10Z',
        project='proj_b76vYqzASNCe7XPZ7Zl4flOs',
    )
    
    try:
    
        
        queries = [
            "cómo instalar Java en Linux",]
        
        # crear funciones de busqueda y que devuelva las relaciones que se tiene con el concepto o termino  

        print(crud.obtener_conceptos_relacionados("yum")) #[{'concepto': 'Búsqueda por tiempo'}, {'concepto': 'Búsqueda por tiempo de modificación'}, {'concepto': 'Búsqueda avanzada'}]
        print(crud.buscar_equivalencias_semanticas("yum")) #[{'termino': 'python', 'similarity': 1.0}, {'termino': 'Python3', 'similarity': 0.8516451789961336}]


        #for i in queries:
            #crear_concepto(client, str(i))
        # actualizar relaciones del grafo
        #learning_engine.actualizar_relaciones_semanticas()
        #learning_engine.actualizar_confianza_relacion()
        

   
    finally:
        neo4j_conn.close()  