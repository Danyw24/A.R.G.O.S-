<<<<<<< HEAD
=======
# BusinessAssistant_AI
permite la configuración de una agencia de agentes de IA especializados que colaboran en tareas empresariales y de procesamiento de datos de Excel


![BusinessAssistant_AI](https://github.com/user-attachments/assets/5b79d3da-6591-40f7-a950-31306468e272)

Es una plataforma de agentes de inteligencia artificial basada en modelos predictivos de lenguaje (LLMs) como GPT-4o y GPT-4-mini diseñada para facilitar tareas empresariales y de análisis de datos mediante una colaboración inteligente entre agentes especializados. Esta suite utiliza la biblioteca **Agency Swarm** para gestionar un entorno donde múltiples agentes interactúan y colaboran, proporcionando soluciones optimizadas en la automatización de datos y asesoramiento empresarial.

## Estructura del Proyecto

El proyecto cuenta con varios agentes, cada uno con roles específicos:

- **CEO**: Actúa como el agente central que coordina las tareas y supervisa las decisiones de los agentes especializados.
- **ExcelAgent**: Un agente especializado en manipulación y limpieza de datos en archivos Excel, ideal para realizar operaciones avanzadas de análisis, transformación y corrección de datos.
- **BusinessAdvisor**: Agente de asesoría empresarial, que proporciona recomendaciones y ayuda en la toma de decisiones con base en el análisis de datos y la información proporcionada por los otros agentes.

## Requisitos

Para utilizar este proyecto, necesitas tener instalado:

- Python 3.9.11 o superior
- Biblioteca **Agency Swarm**
- **Gradio** para la interfaz de usuario
- Archivo `.env` con tu clave de OpenAI (`OPENAI_KEY`)

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/BizAssist-AI-Suite.git

2. Ejecución
   ```bash
   python Agency.py

3. Uso

   ![image](https://github.com/user-attachments/assets/264bd3a3-9a1b-47a8-bc89-17760370e9ca)

   ![image](https://github.com/user-attachments/assets/4eaabb8d-c675-4701-8751-51db328a73b8)


   ![image](https://github.com/user-attachments/assets/6d07ceda-6d99-4a2d-b5b3-9aed296f1371)


   ![image](https://github.com/user-attachments/assets/c2e52e4e-2dc9-422a-b3c2-88758b7dda48)

   ![image](https://github.com/user-attachments/assets/3725502c-9b14-4803-8337-6ae868d43c5e)

   ![image](https://github.com/user-attachments/assets/f599d650-6572-42ed-9da6-e897ad9ae6df)





![image](https://github.com/user-attachments/assets/beb3c6fe-0780-4a02-b5b5-fd5bfb8a564d)



## ¿Por qué usar BusinessAssistant_AI?

- **Automatización inteligente**: Los agentes especializados reducen la carga de trabajo manual, lo que permite a los usuarios concentrarse en aspectos estratégicos de sus negocios.
- **Decisiones informadas**: Los agentes proporcionan asesoría basada en datos, utilizando modelos de IA para ofrecer recomendaciones personalizadas.
- **Fácil integración**: BusinessAssistant_AI es fácil de configurar e integrar con flujos de trabajo existentes, y permite la colaboración eficiente entre distintos agentes.
- **Solución poderosa y accesible**: BusinessAssistant_AI es una solución poderosa y accesible para empresas que buscan mejorar sus procesos mediante la inteligencia artificial, optimizando tareas repetitivas y promoviendo una toma de decisiones más precisa y basada en datos.


## Deficiencias del Proyecto

1. **No es un RAG**: La plataforma no tiene la capacidad de leer, analizar y generar contenido en tiempo real desde fuentes externas como lo hace una AI-Retrieval-Augmented Generation (RAG) .
   
2. **No se actualiza automáticamente**: Depende de actualizaciones manuales y no recibe datos en tiempo real.

3. **Dependencia de OpenAI**: La plataforma depende exclusivamente de la API de OpenAI, lo que genera vulnerabilidad si hay interrupciones en el servicio.

4. **Escalabilidad limitada**: Puede enfrentar problemas al agregar más agentes o tareas complejas, limitando su uso en empresas grandes.

5. **Falta de personalización avanzada**: La personalización de respuestas y tareas es limitada.

6. **Interfaz visual básica**: La interfaz de usuario con Gradio es simple, lo que puede no ser suficiente para necesidades más complejas.

7. **Falta de integraciones**: No tiene integraciones con otras herramientas empresariales como CRM o ERP.

8. **No aprende de forma autónoma**: Los agentes no mejoran por sí mismos, requieren intervención manual.

9. **Configuración compleja**: La configuración inicial puede ser difícil para usuarios no técnicos.





>>>>>>> 19e7fff7 (Update README.md)
