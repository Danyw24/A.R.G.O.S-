# A.R.G.O.S - Sistema Conversacional de Voz en Tiempo Real

<div align="center">
 <img src="https://img.shields.io/badge/A.R.G.O.S-v1.1-blue?style=for-the-badge&logo=robot" alt="A.R.G.O.S Version">
 <img src="https://img.shields.io/badge/Edge--Cloud-AI%20System-green?style=for-the-badge&logo=nvidia" alt="Edge-Cloud System">
 <br>
 <img src="https://img.shields.io/badge/Python-3.8+-3776ab?style=flat&logo=python&logoColor=white" alt="Python">
 <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
 <img src="https://img.shields.io/badge/NVIDIA-TensorRT-76B900?style=flat&logo=nvidia&logoColor=white" alt="TensorRT">
 <img src="https://img.shields.io/badge/Raspberry%20Pi-4B+-A22846?style=flat&logo=raspberry-pi&logoColor=white" alt="Raspberry Pi">
 <img src="https://img.shields.io/badge/🤗-HuggingFace-FFD21E?style=flat" alt="HuggingFace">
 <br>
 <img src="https://img.shields.io/github/license/yourusername/A.R.G.O.S?style=flat" alt="License">
 <img src="https://img.shields.io/github/stars/yourusername/A.R.G.O.S?style=flat&logo=github" alt="Stars">
 <img src="https://img.shields.io/github/issues/yourusername/A.R.G.O.S?style=flat&logo=github" alt="Issues">
 
 <h3>🎯 Sistema conversacional de inteligencia artificial con arquitectura distribuida edge-cloud para interacción humano-máquina de baja latencia</h3>
</div>

---

## 📑 Tabla de Contenidos

- [🔍 Introducción](#-introducción)
- [🎯 Contribuciones Principales](#-contribuciones-principales)
- [🏗️ Arquitectura Distribuida Edge-Cloud](#️-arquitectura-distribuida-edge-cloud)
- [🧠 Modelo Orpheus-TTS](#-modelo-orpheus-tts)
- [🔧 Framework Modular](#-framework-modular)
- [📚 Referencias](#-referencias)
- [🤝 Contribuir](#-contribuir)
- [📜 Licencia](#-licencia)

---

## 🔍 Introducción

Los sistemas conversacionales en tiempo real se están convirtiendo en una herramienta necesaria para las aplicaciones de interacción humano-máquina. Sin embargo, su adopción enfrenta **limitaciones importantes**:

> 🏢 **Acceso restringido**: Los sistemas de alto nivel suelen estar limitados a infraestructuras privadas de grandes empresas  
> ☁️ **Latencia en la nube**: Las soluciones basadas exclusivamente en la nube introducen retardos significativos incompatibles con la fluidez conversacional  
> 📱 **Limitaciones Edge**: Las implementaciones en dispositivos Edge se ven limitadas por sus recursos computacionales  
> 🔗 **Complejidad de integración**: Dificultad de integrar múltiples modelos (reconocimiento de voz, detección de actividad vocal y síntesis de voz) en un mismo flujo de procesamiento conversacional

Estas restricciones evidencian la **necesidad de marcos equilibrados** que combinen precisión, eficiencia y accesibilidad de código abierto.

**A.R.G.O.S** presenta un sistema conversacional de voz en tiempo real que integra **cinco modelos especializados de inteligencia artificial** en una arquitectura distribuida edge-cloud para una interacción humano-máquina de baja latencia. El sistema permite un flujo conversacional natural mediante streaming bidireccional de audio y un control inteligente de micrófono.

---

## 🎯 Contribuciones Principales

### 🏗️ **(I) Arquitectura Distribuida Edge-Cloud**
Diseño de una arquitectura que integra cinco modelos de inteligencia artificial especializados:
- **VAD** (Voice Activity Detection)
- **ASR** (Automatic Speech Recognition) 
- **NLU** (Natural Language Understanding)
- **TTS** (Text-to-Speech)
- **Control de Audio**

Logrando una **interacción conversacional de baja latencia** en tiempo real.

### 🎙️ **(II) Modelo TTS Optimizado (Orpheus)**
Adaptación y fine-tuning del modelo TTS Orpheus para síntesis de voz en **castellano**:
- ✅ **Dataset propio** especializado
- ⚡ **Optimizaciones NVIDIA TensorRT** en formato **FP8**
- 📉 **Reducción significativa** del tiempo de inferencia y consumo de memoria en GPU

### 🔧 **(III) Framework Modular Open Source**
Implementación de un framework modular disponible en **GitHub** y **HuggingFace**:
- 🔄 **Intercambiable**: Permite sustituir o extender el LLM central
- 📈 **Escalable**: Adaptable según requerimientos (Ventas, Servicio técnico, Educación)
- 🛠️ **Personalizable**: Permite la customización de agentes conversacionales

---

## 🏗️ Arquitectura Distribuida Edge-Cloud

El sistema implementa una **arquitectura híbrida** que distribuye la carga computacional entre dispositivos de borde y recursos en la nube, optimizando tanto la latencia como la eficiencia energética.




### 📊 Distribución de Componentes

| **Componente** | **Ubicación** | **Justificación** |
|----------------|---------------|-------------------|
| **🔊 VAD** | Edge | Reduce ancho de banda y latencia inicial |
| **📝 ASR** | Edge/Cloud | Balance entre recursos locales y precisión |
| **🧠 NLU + LLM** | Cloud | Requiere alta capacidad computacional |
| **🎯 TTS** | Cloud | Modelo optimizado con TensorRT FP8 |
| **🎛️ Control Audio** | Edge | Respuesta inmediata requerida |

---

## 🧠 Modelo Orpheus-TTS

**Orpheus** es una familia de modelos LLM de voz de última generación considerados como el **"state of the art"** en la generación de voz con un nivel de habla humano.

### 🔬 Arquitectura Técnica

- **🏗️ Base**: Utiliza **Llama-3.2-1B** con modificaciones estructurales en su arquitectura
- **🔄 Procesamiento**: Tokens de texto y audio entrelazados en formato codificado **SNAC** (Codecs de Audio Neurales)
- **🎵 Salida**: Decoder que genera espectrograma de audio con datos completos en resolución de **24kHz**

### ⚡ Optimizaciones Implementadas

- **🇪🇸 Fine-tuning para castellano** con dataset propio especializado
- **🚀 Optimización NVIDIA TensorRT** en formato **FP8** para reducir latencia
- **📡 Streaming de audio bidireccional** para interacción en tiempo real
- **🎛️ Control inteligente de micrófono** con detección de actividad vocal

### 🔄 Pipeline de Procesamiento
 - 🎤 Micrófono → 🔊 VAD → 📝 ASR → 🧠 NLU → 🤖 LLM → 🎯 TTS (Orpheus) → 🔈 Altavoces



El sistema procesa el audio de entrada detectando primero la **actividad vocal**, transcribe el **habla a texto**, procesa la **intención del usuario**, genera una **respuesta contextual** y finalmente sintetiza la respuesta en **audio natural** usando el modelo Orpheus optimizado.

---

## 🔧 Framework Modular

La **arquitectura modular** de A.R.G.O.S permite:

### 🔄 **Intercambio de Modelos**
- Sustitución del LLM central según la aplicación específica
- Compatibilidad con múltiples proveedores de IA

### 📈 **Escalabilidad**
- Soporte para múltiples usuarios concurrentes
- Distribución de carga automática

### 🎯 **Personalización de Agentes**
- **💼 Ventas**: Agentes especializados en comercialización
- **🛠️ Soporte técnico**: Resolución de problemas especializados
- **📚 Educación**: Tutores virtuales adaptativos

### 🔌 **Extensibilidad**
- Fácil integración de nuevos componentes de IA
- API modular para desarrolladores terceros

---

## 📚 Referencias

<div align="left">

**[1]** Sathish, V., Lin, H., Kamath, A. K., & Nyayachavadi, A. (2024). *LLeMpower: Understanding Disparities in the Control and Access of Large Language Models*. arXiv preprint arXiv:2404.09356.

**[2]** Leow, C. S., Hayakawa, T., Nishizaki, H., & Kitaoka, N. (2020). *Development of a Low-Latency and Real-Time Automatic Speech Recognition System*. University of Yamanashi. Retrieved from https://www.alps-lab.org/wp-content/uploads/2020/11/1570656914.pdf

**[3]** Wang, T., Guo, J., Zhang, B., Yang, G., & Li, D. (2025). *Deploying AI on Edge: Advancement and Challenges in Edge Intelligence*. Mathematics, 13(11), 1878.

**[4]** Shah, B., Jain, S., & Taqa, A. (2025). *Hybrid Cloud Architectures for Multi-Modal AI Systems*. International Journal of Scientific Research, 2455-6211.

**[5]** Anonymous. (2024). *An Initial Investigation of Language Adaptation for TTS Systems under Low-resource Scenarios*. arXiv preprint arXiv:2406.08911. Retrieved from https://arxiv.org/pdf/2406.08911

**[6]** NVIDIA Corporation. *TensorRT - High Performance Deep Learning Inference*. NVIDIA Developer. Retrieved from https://developer.nvidia.com/tensorrt

</div>

---

## 🤝 Contribuir

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📜 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

<div align="center">
  <strong>Desarrollado con ❤️ para la comunidad de IA conversacional</strong>
  <br>
  <sub>Si este proyecto te resulta útil, considera darle una ⭐</sub>
</div>