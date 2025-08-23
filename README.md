# A.R.G.O.S - Sistema Conversacional de Voz en Tiempo Real

## Introducción

Los sistemas conversacionales en tiempo real se están convirtiendo en una herramienta necesaria para las aplicaciones de interacción humano-máquina, Sin embargo, su adopción enfrenta limitaciones importantes. Por un lado, el acceso a sistemas de alto nivel suele estar limitado a infraestructuras privadas de grandes empresas, por otro, las soluciones basadas exclusivamente en la nube introducen retardos significativos incompatibles con la fluidez conversacional, mientras que las implementaciones en dispositivos Edge se ven limitadas por sus recursos computacionales, Por esto le suma la dificultad de integrar múltiples modelos como –Reconocimiento de voz, detección de actividad vocal y síntesis de voz—todo en un mismo flujo, de procesamiento conversacional. Estas restricciones evidencian la necesidad de marcos equilibrados que combinen precisión, eficiencia y accesibilidad de código abierto.

Este trabajo presenta un sistema conversacional de voz en tiempo real que integra cinco modelos especializados de inteligencia artificial en una arquitectura distribuida edge-cloud para una interacción humano-máquina de baja latencia. El sistema permite un flujo conversacional natural mediante streaming bidireccional de audio y un control inteligente de micrófono, logrando una interacción humano-máquina de baja latencia.

## Contribuciones Principales

**(I)** El diseño de una arquitectura distribuida edge-cloud que integra cinco modelos de inteligencia artificial especializados (VAD, ASR, NLU, TTS y control de audio), logrando una interacción conversacional de baja latencia en tiempo real.

**(II)** La adaptación y fine-tuning de un modelo TTS (Orpheus) para síntesis de voz en castellano, utilizando un dataset propio y optimizaciones con NVIDIA TensorRT en formato FP8, lo que reduce significativamente el tiempo de inferencia y consumo de memoria en GPU.

**(III)** La implementación de un framework modular y de código abierto, disponible en GitHub y HuggingFace, que permite sustituir o extender el LLM central según el requerimiento de la aplicación (p. Ej., Ventas, Servicio técnico, Educación), permitiendo la escalabilidad y personalización de agentes conversacionales.

## Arquitectura Distribuida Edge-Cloud

El sistema implementa una arquitectura híbrida que distribuye la carga computacional entre dispositivos de borde y recursos en la nube, optimizando tanto la latencia como la eficiencia energética.

### Componentes del Sistema

**VAD (Voice Activity Detection)** - Detección de actividad vocal en tiempo real
**ASR (Automatic Speech Recognition)** - Reconocimiento automático de voz 
**NLU (Natural Language Understanding)** - Comprensión de lenguaje natural
**TTS (Text-to-Speech)** - Síntesis de voz con modelo Orpheus optimizado
**Control de Audio** - Gestión inteligente del flujo de audio bidireccional

## Modelo Orpheus-TTS

Orpheus es una familia de modelos LLM de voz de última generación considerados como el "state of the art" (estado del arte), en la generación de voz con un nivel de habla humano.

El modelo utiliza Llama-3.2-1B, con modificaciones estructurales en su arquitectura para procesar tokens de texto y audio entrelazados en un formato codificado SNAC (Codecs de audio Neurales) que posteriormente son procesados por un decoder para generar un espectrograma de audio que contiene los datos de audio completos en una resolución de 24kHz.

### Optimizaciones Implementadas

- **Fine-tuning para castellano** con dataset propio especializado
- **Optimización NVIDIA TensorRT** en formato FP8 para reducir latencia
- **Streaming de audio bidireccional** para interacción en tiempo real
- **Control inteligente de micrófono** con detección de actividad vocal

### Pipeline de Procesamiento