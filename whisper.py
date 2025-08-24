# utilizar vad para cortar chunks de audio y enviarlos por post al endpoind de baseten 
# Ademas de que tienen que ser http por chunks, tcp no es soportado
#arecord -f S16_LE -c1 -r 16000 -t raw -D default | nc <SERVIDOR_ASR> 4300

import socket
import io
import wave
import requests
import threading
import numpy as np
import contextlib
from collections import deque
import onnxruntime as ort
import shutil
from pathlib import Path
import subprocess
import asyncio
import aiohttp
import uuid
import time
import os
import subprocess
import torch 
from IPython.display import Audio
from pprint import pprint
import base64
torch.set_num_threads(1)


PORT = 4300
CHUNK_SEC = 2.0
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_SIZE = int(SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_SEC)
API_URL = "http://127.0.0.1:8080/transcribe/"
API_KEY = "Bearer TU_API_KEY"


# Configuration
MODEL = "232k54x3"
BASETEN_HOST = f"https://model-nwxn27z3.api.baseten.co/deployment/31zvjgw/predict"
BASETEN_API_KEY = "9Vux7EuB.5y3gD8nLjOzhLjqxB4wu8Rtsd77gPh2J"
PAYLOADS_PER_PROCESS = 1
NUM_PROCESSES = 1
MAX_REQUESTS_PER_PROCESS = 8

"9Vux7EuB.5y3gD8nLjOzhLjqxB4wu8Rtsd77gPh2J"

prompt_types = ["short", "medium", "long"]


base_request_payload = {
    "max_tokens": 4096,
    "voice": "david",
    "top_p" : 0.85,
    "repetition_penalty": 1.4,
    "temperature": 0.8,
    "stop_token_ids": [128258, 128009],
}



def extract_onnx_model():
    """Extrae el modelo ONNX de la cache de torch.hub y lo guarda localmente"""
    
    
    cache_dir = Path.home() / ".cache" / "silero_vad_onnx"
    onnx_path = cache_dir / "silero_vad.onnx"
    
    if onnx_path.exists():
        print(f"[+] Modelo ONNX ya existe en: {onnx_path}")
        print("[!] Cargando utils de Silero...")
        _, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            onnx=True,
            opset_version=16,
            force_reload=False
        )
        return str(onnx_path), vad_utils
    
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("[+] Descargando modelo ONNX por primera vez...")
    
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        onnx=True,
        opset_version=16,
        force_reload=False
    )
    
    
    torch_cache = Path.home() / ".cache" / "torch" / "hub"
    
    
    for onnx_file in torch_cache.rglob("*.onnx"):
        if "silero" in str(onnx_file).lower() or "vad" in str(onnx_file).lower():
            shutil.copy2(onnx_file, onnx_path)
            print(f"[+] Modelo ONNX copiado a: {onnx_path}")
            break
    
    
    return str(onnx_path), vad_utils

def load_silero_model() -> None:
    """Carga el modelo usando ONNX Runtime para máximo rendimiento"""
    global vad_session, vad_utils
    import time
    import librosa
    
    # Extraer modelo ONNX si no existe
    onnx_path, vad_utils = extract_onnx_model()
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    providers = [('CPUExecutionProvider', {'intra_op_num_threads': 1, 'inter_op_num_threads': 1})]
    
    try:
        vad_session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        print("[+] Modelo ONNX cargado con ONNX Runtime")
        
        try:
            audio_data, _ = librosa.load("temp_16000.wav", sr=16000, mono=True)
            test_audio = audio_data[:512].reshape(1, -1).astype(np.float32)
        except:
            test_audio = np.random.randn(1, 512).astype(np.float32)

        input_names = [inp.name for inp in vad_session.get_inputs()]
        if 'state' in input_names:
            state = np.zeros((2, 1, 128), dtype=np.float32)
            inputs = {'input': test_audio, 'state': state}
            # Agregar sample rate si es requerido
            if 'sr' in input_names:
                inputs['sr'] = np.array([16000], dtype=np.int64)

        elif 'h' in input_names and 'c' in input_names:
            h = np.zeros((1, 128), dtype=np.float32)
            c = np.zeros((1, 128), dtype=np.float32)
            inputs = {'input': test_audio, 'h': h, 'c': c}
            # Agregar sample rate si es requerido
            if 'sr' in input_names:
                inputs['sr'] = np.array([16000], dtype=np.int64)
        else:
            inputs = {'input': test_audio}
            # Agregar sample rate si es requerido
            if 'sr' in input_names:
                inputs['sr'] = np.array([16000], dtype=np.int64)
                        
        start_time = time.time()
        outputs = vad_session.run(None, inputs)
        test_time = (time.time() - start_time) * 1000
        
        print(f"[+] Test inicial: {test_time:.1f}ms | VAD: {outputs[0][0][0]:.3f}")
        
    except Exception as e:
        print(f"❌ Error cargando modelo ONNX: {e}")
        raise
    
    # Desempacar utils
    get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = vad_utils
    globals().update({
        'get_speech_timestamps': get_speech_timestamps,
        'save_audio': save_audio,
        'read_audio': read_audio,
        'VADIterator': VADIterator,
        'collect_chunks': collect_chunks
    })



def vad_inference_fast(audio_chunk, sample_rate=16000, state=None):
    """Inferencia VAD ultrarrápida usando ONNX Runtime con manejo de estado"""
    global vad_session
    
    # Preparar input (debe ser float32)
    if isinstance(audio_chunk, np.ndarray):
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
    else:
        audio_chunk = np.array(audio_chunk, dtype=np.float32)
    
    # Reshape si es necesario (batch_size=1)
    if len(audio_chunk.shape) == 1:
        audio_chunk = audio_chunk.reshape(1, -1)
    
    try:
        # Preparar inputs según el modelo
        input_names = [inp.name for inp in vad_session.get_inputs()]
        
        if 'state' in input_names:
            # Estado combinado
            if state is None:
                state = np.zeros((2, 1, 128), dtype=np.float32)
            inputs = {'input': audio_chunk, 'state': state}
            # Agregar sample rate si es requerido
            if 'sr' in input_names:
                inputs['sr'] = np.array([16000], dtype=np.int64)
        elif 'h' in input_names and 'c' in input_names:
            # Estados separados - CORRIGIENDO DIMENSIONES
            if state is None:
                h = np.zeros((1, 128), dtype=np.float32)
                c = np.zeros((1, 128), dtype=np.float32)
            else:
                h, c = state[0], state[1]
            inputs = {'input': audio_chunk, 'h': h, 'c': c}
            # Agregar sample rate si es requerido
            if 'sr' in input_names:
                inputs['sr'] = np.array([16000], dtype=np.int64)
        else:
            # Solo audio
            inputs = {'input': audio_chunk}
            # Agregar sample rate si es requerido
            if 'sr' in input_names:
                inputs['sr'] = np.array([16000], dtype=np.int64)
        
        # Inferencia directa con ONNX Runtime
        outputs = vad_session.run(None, inputs)
        
        # Retornar probabilidad y nuevo estado si existe
        speech_prob = outputs[0][0][0] if len(outputs[0].shape) > 1 else outputs[0][0]
        new_state = outputs[1] if len(outputs) > 1 else None
        
        return speech_prob, new_state
        
    except Exception as e:
        print(f"❌ Error en inferencia VAD fast: {e}")
        return 0.0, state


def get_speech_timestamps_onnx(wav, sample_rate=16000, threshold=0.5, 
                                min_speech_duration_ms=100, min_silence_duration_ms=100,
                                window_size_samples=512, padding_ms=50):
    """
    Versión adaptada de get_speech_timestamps que usa ONNX Runtime directamente
    Con mejoras para evitar cortes en el habla natural
    """
    global vad_session
    
    if vad_session is None:
        raise ValueError("VAD session no está inicializada")
    
    # Parámetros
    min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
    min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
    padding_samples = int(sample_rate * padding_ms / 1000)  # Padding antes/después
    
    speech_probs = []
    state = None
    
    for start in range(0, len(wav), window_size_samples):
        end = min(start + window_size_samples, len(wav))
        chunk = wav[start:end]
        
        # Pad si es necesario
        if len(chunk) < window_size_samples:
            chunk = np.pad(chunk, (0, window_size_samples - len(chunk)))
        
        # Inferencia VAD
        prob, state = vad_inference_fast(chunk, sample_rate, state)
        speech_probs.append((start, end, prob))
        
    speech_timestamps = []
    current_speech_start = None
    silence_start = None
    silence_duration = 0
    
    for start, end, prob in speech_probs:
        is_speech = prob > threshold
        
        if is_speech:
            if current_speech_start is None:
                # Inicio de nuevo segmento de voz
                current_speech_start = max(0, start - padding_samples)  # Con padding
            
            # Resetear contador de silencio
            silence_start = None
            silence_duration = 0
            
        else:  # Silencio detectado
            if current_speech_start is not None:
                # Estamos en un segmento de voz, empezar a contar silencio
                if silence_start is None:
                    silence_start = start
                
                silence_duration = start - silence_start
                
                # Solo cerrar si el silencio es suficientemente largo
                if silence_duration >= min_silence_samples:
                    speech_end = min(len(wav), silence_start + padding_samples)  # Con padding
                    speech_duration = speech_end - current_speech_start
                    
                    # Verificar duración mínima del segmento de voz
                    if speech_duration >= min_speech_samples:
                        speech_timestamps.append({
                            'start': current_speech_start,
                            'end': speech_end
                        })
                    
                    current_speech_start = None
                    silence_start = None
                    silence_duration = 0
    
    # Manejar segmento final si queda abierto
    if current_speech_start is not None:
        speech_end = len(wav)
        speech_duration = speech_end - current_speech_start
        if speech_duration >= min_speech_samples:
            speech_timestamps.append({
                'start': current_speech_start,
                'end': speech_end
            })
    
    # Fusionar segmentos muy cercanos (post-procesamiento)
    if len(speech_timestamps) > 1:
        merged_timestamps = []
        current_segment = speech_timestamps[0]
        
        for next_segment in speech_timestamps[1:]:
            gap = next_segment['start'] - current_segment['end']
            # Si el gap es menor a min_silence_duration, fusionar
            if gap < min_silence_samples:
                current_segment['end'] = next_segment['end']
            else:
                merged_timestamps.append(current_segment)
                current_segment = next_segment
        
        merged_timestamps.append(current_segment)
        speech_timestamps = merged_timestamps
    
    return speech_timestamps


def handle_connection(conn, session, semph):
    # ==========================================
    # CONFIGURACIÓN DE PARÁMETROS OPTIMIZADOS
    # ==========================================
    
    # Parámetros de frame
    FRAME_DURATION = 0.064
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION * BYTES_PER_SAMPLE)
    
    # Parámetros de calibración
    CALIB_SEC = 2.0
    CALIB_FRAMES = int(CALIB_SEC / FRAME_DURATION)
    
    # --- MEJORA: Parámetros de umbral dinámico ---
    ADAPTIVE_THRESHOLD = True
    BASE_THRESHOLD_PERCENTILE = 50
    # El multiplicador ahora se ajustará dinámicamente tras la calibración
    VOICE_MULTIPLIER_MIN = 1.25  # Mínimo para ambientes silenciosos
    VOICE_MULTIPLIER_MAX = 1.6   # Máximo para ambientes ruidosos
    FALLBACK_THRESHOLD = 1850.0
    
    # Parámetros de detección de habla
    MIN_SPEECH_DURATION_MS = 200
    MAX_SILENCE_DURATION_MS = 700
    MIN_SPEECH_FRAMES = int(MIN_SPEECH_DURATION_MS / (FRAME_DURATION * 1000))
    MAX_SILENCE_FRAMES = int(MAX_SILENCE_DURATION_MS / (FRAME_DURATION * 1000))
    
    # --- MEJORA: Confirmación de voz para evitar falsos positivos ---
    SPEECH_CONFIRMATION_FRAMES = 2 # Requiere 2 frames seguidos para iniciar la detección
    
    # Parámetros de análisis VAD
    RMS_SMOOTHING_WINDOW = 3
    MIN_VAD_CHUNK_SIZE = 512
    MIN_AUDIO_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.4)

    # ==========================================
    # INICIALIZACIÓN DE VARIABLES
    # ==========================================
    
    buffer = b""
    speech_buffer = b""
    calibration_buffer = b""
    
    silence_frames_after_speech = 0
    speech_frame_count = 0
    processed_segments = 0
    frames_collected = 0
    
    # --- MEJORA: Estado de detección y contador para confirmación ---
    is_speaking = False
    potential_speech_frames = 0
    
    rms_samples = deque(maxlen=CALIB_FRAMES * 2)
    rms_history = deque(maxlen=RMS_SMOOTHING_WINDOW)
    
    SILENCE_THRESHOLD = None
    VOICE_MULTIPLIER = VOICE_MULTIPLIER_MIN # Valor inicial
    is_calibrated = False
    
    # ==========================================
    # FASE 1: CALIBRACIÓN INTELIGENTE
    # ==========================================
    
    print("🔧 Iniciando calibración inteligente del umbral RMS...")
    print(f"   ⏱️ Duración: {CALIB_SEC}s ({CALIB_FRAMES} frames)")
    print("   🤫 Mantén silencio durante la calibración...")
    
    try:
        # Recolección de muestras para calibración
        start_time = time.time()
        while frames_collected < CALIB_FRAMES:
            try:
                data = conn.recv(8192)
                if not data:
                    print("❌ Error: conexión cerrada durante calibración")
                    return
                
                calibration_buffer += data
                
                while len(calibration_buffer) >= FRAME_SIZE and frames_collected < CALIB_FRAMES:
                    frame = calibration_buffer[:FRAME_SIZE]
                    calibration_buffer = calibration_buffer[FRAME_SIZE:]
                    
                    try:
                        audio_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
                        if len(audio_np) > 0:
                            rms = np.sqrt(np.mean(audio_np**2))
                            if rms > 50:
                                rms_samples.append(rms)
                                frames_collected += 1
                    except Exception as e:
                        print(f"   ⚠️ Error procesando frame: {e}")
                        continue
                    
                    if frames_collected % 10 == 0 and frames_collected > 0:
                        progress = (frames_collected / CALIB_FRAMES) * 100
                        elapsed = time.time() - start_time
                        print(f"   📊 Progreso: {progress:.1f}% ({elapsed:.1f}s)")
                        
            except Exception as e:
                print(f"   ⚠️ Error en calibración: {e}")
                continue
        
        if len(rms_samples) >= 15:
            rms_array = np.array(rms_samples)
            
            mean_rms = np.mean(rms_array)
            std_rms = np.std(rms_array)
            median_rms = np.median(rms_array)
            p50_rms = np.percentile(rms_array, BASE_THRESHOLD_PERCENTILE)
            
            # --- MEJORA: Cálculo de multiplicador dinámico ---
            # Se ajusta el multiplicador basado en la variabilidad del ruido (std).
            # Más variabilidad = ambiente más ruidoso = multiplicador más alto.
            noise_variability = std_rms / mean_rms if mean_rms > 0 else 0
            VOICE_MULTIPLIER = np.clip(
                VOICE_MULTIPLIER_MIN + noise_variability * 2.0, 
                VOICE_MULTIPLIER_MIN, 
                VOICE_MULTIPLIER_MAX
            )

            if ADAPTIVE_THRESHOLD:
                base_noise = p50_rms
                adaptive_threshold = base_noise * VOICE_MULTIPLIER
                
                # Límites de seguridad para el umbral
                SILENCE_THRESHOLD = float(np.clip(adaptive_threshold, 1600, 2300))
            else:
                SILENCE_THRESHOLD = FALLBACK_THRESHOLD
            
            is_calibrated = True
            
            print(f"\n[✅] Calibración completada exitosamente:")
            print(f"   📊 Ruido (Media/Std): {mean_rms:.1f} / {std_rms:.1f}")
            print(f"   📈 Multiplicador dinámico calculado: {VOICE_MULTIPLIER:.2f}")
            print(f"   🎯 Umbral de silencio final: {SILENCE_THRESHOLD:.1f}")
            
        else:
            print("❌ Error: insuficientes muestras. Usando umbral de respaldo.")
            SILENCE_THRESHOLD = FALLBACK_THRESHOLD
            is_calibrated = True
    
    except Exception as e:
        print(f"❌ Error crítico en calibración: {e}. Usando umbral de emergencia.")
        SILENCE_THRESHOLD = FALLBACK_THRESHOLD
        is_calibrated = True
    
    # ==========================================
    # FASE 2: DETECCIÓN DE VOZ MEJORADA
    # ==========================================
    
    print(f"\n[🎤] Iniciando detección de voz optimizada...")
    print(f"   🎯 Umbral activo: {SILENCE_THRESHOLD:.1f}")
    
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
                
            buffer += data
            
            while len(buffer) >= FRAME_SIZE:
                frame = buffer[:FRAME_SIZE]
                buffer = buffer[FRAME_SIZE:]
                
                audio_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
                rms = np.sqrt(np.mean(audio_np**2)) if len(audio_np) > 0 else 0
                
                rms_history.append(rms)
                smoothed_rms = float(np.mean(rms_history))
                
                # --- MEJORA: Lógica de estados para detección ---
                if is_speaking:
                    # --- ESTADO: YA ESTAMOS GRABANDO VOZ ---
                    if smoothed_rms > SILENCE_THRESHOLD:
                        # La voz continúa, seguimos grabando
                        speech_buffer += frame
                        speech_frame_count += 1
                        silence_frames_after_speech = 0
                    else:
                        # Silencio detectado después de voz, empezamos a contar
                        silence_frames_after_speech += 1
                        speech_buffer += frame # Añadir el silencio final para un corte natural

                        if silence_frames_after_speech >= MAX_SILENCE_FRAMES:
                            # Fin del segmento de voz por silencio prolongado
                            print("🔇", end="", flush=True)
                            
                            # Procesar si el audio es suficientemente largo
                            if speech_frame_count >= MIN_SPEECH_FRAMES and len(speech_buffer) >= MIN_AUDIO_BYTES:
                                processed_segments += 1
                                segment_duration = len(speech_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                                print(f"\n📝 Segmento {processed_segments} finalizado ({segment_duration:.2f}s). Procesando...")
                                
                                try:
                                    # (El código de procesamiento VAD y envío a Baseten va aquí, sin cambios)
                                    wav_buffer = io.BytesIO()
                                    with wave.open(wav_buffer, 'wb') as wf:
                                        wf.setnchannels(1)
                                        wf.setsampwidth(2)
                                        wf.setframerate(SAMPLE_RATE)
                                        wf.writeframes(speech_buffer)
                                    wav_buffer.seek(0)
                                    
                                    # Aquí iría tu lógica de `get_speech_timestamps_onnx` y `send_to_baseten`
                                    send_to_baseten(wav_buffer) # Placeholder para tu función de envío
                                    print("   📤 Enviado a Baseten")
                                    
                                except Exception as e:
                                    print(f"   ❌ Error procesando segmento: {e}")
                            else:
                                # El audio grabado es muy corto, se descarta
                                print(f"\n⏭️ Segmento descartado (demasiado corto: {len(speech_buffer)} bytes)")

                            # --- Reseteo de estado para la próxima detección ---
                            is_speaking = False
                            speech_buffer = b""
                            speech_frame_count = 0
                            silence_frames_after_speech = 0
                            potential_speech_frames = 0
                else:
                    # --- ESTADO: ESPERANDO VOZ ---
                    if smoothed_rms > SILENCE_THRESHOLD:
                        # Umbral superado, posible inicio de voz
                        potential_speech_frames += 1
                        speech_buffer += frame # Guardar temporalmente por si se confirma

                        if potential_speech_frames >= SPEECH_CONFIRMATION_FRAMES:
                            # Se confirma el inicio de voz
                            is_speaking = True
                            speech_frame_count = potential_speech_frames
                            potential_speech_frames = 0
                            print("🗣️", end="", flush=True)
                    else:
                        # No es voz, reseteamos el buffer de confirmación
                        potential_speech_frames = 0
                        speech_buffer = b""

    except KeyboardInterrupt:
        print(f"\n🛑 Interrupción detectada.")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
    finally:
        print(f"\n📊 Resumen: {processed_segments} segmentos procesados.")
        conn.close()
        print("🔌 Conexión cerrada.")

 
def start_arecord():
    try:
        pipeline = (
            "arecord -f S16_LE -c1 -r 16000 -t raw -D plughw:3,0 | "
            "sox -t raw -r 16000 -e signed -b 16 -c 1 - -t raw - gain 18 | "
            "nc 127.0.0.1 4300"
        )
        return subprocess.Popen(pipeline, shell=True)
    except Exception as e:
        print(f"Error iniciando arecord con amplificación: {e}")
        try:
            fallback = "arecord -f S16_LE -c1 -r 16000 -t raw -D plughw:3,0 | nc 127.0.0.1 4300"
            return subprocess.Popen(fallback, shell=True)
        except Exception as e2:
            print(f"Error en fallback: {e2}")
            return subprocess.Popen([
                "arecord -f S16_LE -c1 -r 16000 -t raw -D default | nc 127.0.0.1 4300"
            ], shell=True)
            
            
def send_to_baseten(wav_data):
    """
    Envia un paquete por POST a la API para ser transcrito por el modelo argos-voice,
    y recibe directamente el audio streaming para reproducir en tiempo real.
    """
    # Combinar base_request_payload con los datos específicos de este request
    payload = {**base_request_payload}  # Expandir el payload base
    
    # Agregar datos específicos del archivo WAV y voz (codificado en base64)
    payload.update({
        "file": base64.b64encode(wav_data.getvalue()).decode('utf-8'),  # Codificar en base64
        "voice": "david"
    })
    
    # Headers para JSON
    headers = {
        "Authorization": f"Api-Key {BASETEN_API_KEY}",
        "Content-Type": "application/json"
    }
    
        # CAMBIO 1: Función para mute/unmute del micrófono
    def mute_microphone():
        try:
            subprocess.run(["amixer", "sset", "Capture", "0%"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("🔇 Micrófono silenciado")
        except:
            pass
    
    def unmute_microphone():
        try:
            subprocess.run(["amixer", "sset", "Capture", "85%"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("🎤 Micrófono reactivado")
        except:
            pass
    
    
    try:
        # Hacer request con streaming habilitado al host correcto
        response = requests.post(BASETEN_HOST, json=payload, headers=headers, stream=True)
        
        if response.status_code == 200:
            print("✅ Conectado al servidor, iniciando streaming de audio...")
            mute_microphone()
            # Generar ID único para este request
            run_id = f"stream_{int(time.time())}"
            
            # Preparar directorio de salida
            output_dir = os.path.join(os.path.dirname(__file__), "Outputs")
            os.makedirs(output_dir, exist_ok=True)
            fn = os.path.join(output_dir, f"output_response_{run_id}.wav")
            
            # Limpiar procesos previos de aplay
            try:
                subprocess.run(["pkill", "-f", "aplay"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL, 
                            timeout=2)
            except:
                pass
            
            # Buffer para guardar archivo completo
            audio_buffer = bytearray()
            chunk_count = 0
            
            # Iniciar aplay para streaming directo
            aplay_process = None
            try:
                # Configurar aplay para audio de 24kHz mono
                aplay_process = subprocess.Popen([
                    "aplay", 
                    "-f", "S16_LE",      # 16-bit signed little endian
                    "-r", "24000",       # 24kHz sample rate
                    "-c", "1",           # mono
                    "-t", "raw",         # raw format (sin headers WAV)
                    "-"                  # leer desde stdin
                ], 
                stdin=subprocess.PIPE, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL)
                
                first_chunk = True
                
                # Procesar chunks de audio streaming
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        audio_buffer.extend(chunk)
                        chunk_count += 1
                        
                        if first_chunk:
                            print(f"🎵 Primer chunk de audio recibido, iniciando reproducción...")
                            first_chunk = False
                        
                        # Enviar chunk directamente a aplay para reproducción inmediata
                        if aplay_process and aplay_process.poll() is None:
                            try:
                                aplay_process.stdin.write(chunk)
                                aplay_process.stdin.flush()
                            except BrokenPipeError:
                                print("⚠️ Error en pipe de aplay")
                                break
                        
                        if chunk_count % 10 == 0:
                            print(f"🎵 Reproduciendo chunk #{chunk_count}...")
                
                # Cerrar pipe para terminar reproducción
                if aplay_process:
                    aplay_process.stdin.close()
                    aplay_process.wait(timeout=5)
                
                print(f"✅ Audio completado: {len(audio_buffer)} bytes en {chunk_count} chunks")
                
            except Exception as play_error:
                print(f"⚠️ Error durante reproducción streaming: {play_error}")
            finally:
                # Limpiar proceso aplay
                if aplay_process and aplay_process.poll() is None:
                    try:
                        aplay_process.terminate()
                        aplay_process.wait(timeout=2)
                    except:
                        try:
                            aplay_process.kill()
                        except:
                            pass
                unmute_microphone()
            # Guardar archivo WAV completo para respaldo
            try:
                with wave.open(fn, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(audio_buffer)
                print(f"💾 Audio guardado en: {fn}")
            except Exception as save_error:
                print(f"⚠️ Error guardando archivo: {save_error}")
                
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        print("⚠️ Error HTTP:", e)
        unmute_microphone()
    finally:
        # Limpieza final
        try:
            subprocess.run(["pkill", "-f", "aplay"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=1)
        except:
            pass



async def stream_to_buffer(
    session: aiohttp.ClientSession, label: str, payload: dict
) -> bytes:
    """Send one streaming request, accumulate into bytes, and log timings."""
    req_id = str(uuid.uuid4())
    payload = {**payload, "request_id": req_id}

    t0 = time.perf_counter()

    try:
        async with session.post(
            BASETEN_HOST,
            json=payload,  # payload - change to waveform or .wav audio to be processed as speech
            headers={"Authorization": f"Api-Key {BASETEN_API_KEY}"},
        ) as resp:
            if resp.status != 200:
                print(f"[{label}] ← HTTP {resp.status}")
                return b""

            buf = bytearray()
            idx = 0
            # *** CORRECTED: async for on the AsyncStreamIterator ***
            async for chunk in resp.content.iter_chunked(4_096):
                elapsed_ms = (time.perf_counter() - t0) * 1_000
                if idx in [0, 10]:
                    print(
                        f"[{label}] ← chunk#{idx} ({len(chunk)} B) @ {elapsed_ms:.1f} ms"
                    )
                buf.extend(chunk)
                idx += 1

            total_s = time.perf_counter() - t0
            print(f"[{label}] ← done {len(buf)} B in {total_s:.2f}s")
            return bytes(buf)

    except Exception as e:
        print(f"[{label}] ⚠️ exception: {e!r}")
        return b""



def run_session_sync(prompt: str, ptype: str, run_id: str):
    """Ejecuta síntesis de voz con streaming y reproduce el audio en tiempo real"""
    payload = {**base_request_payload, "prompt": prompt}
    headers = {"Authorization": f"Api-Key {BASETEN_API_KEY}"}
    
    try:
        # Streaming request
        r = requests.post(BASETEN_HOST, json=payload, headers=headers, stream=True)
        
        if r.status_code != 200:
            print(f"❌ TTS HTTP {r.status_code}: {r.text}")
            return
        
        # Preparar directorios
        output_dir = os.path.join(os.path.dirname(__file__), "Outputs")
        os.makedirs(output_dir, exist_ok=True)
        fn = os.path.join(output_dir, f"output_{ptype}_{run_id}.wav")
        
        # Limpiar procesos previos
        try:
            subprocess.run(["pkill", "-f", "aplay"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL, 
                        timeout=2)
        except:
            pass
        
        # Buffer para guardar archivo completo
        audio_buffer = bytearray()
        chunk_count = 0
        
        # Iniciar aplay en modo pipe para streaming directo
        aplay_process = None
        try:
            # Configurar aplay para recibir datos raw por stdin
            aplay_process = subprocess.Popen([
                "aplay", 
                "-f", "S16_LE",      # 16-bit signed little endian
                "-r", "24000",       # 24kHz sample rate
                "-c", "1",           # mono
                "-t", "raw",         # raw format (sin headers WAV)
                "-"                  # leer desde stdin
            ], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL)
            
            first_chunk = True
            
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    audio_buffer.extend(chunk)
                    chunk_count += 1
                    
                    if first_chunk:
                        print(f"[{ptype}_{run_id}] ← first chunk received, starting playback")
                        first_chunk = False
                    
                    # Enviar chunk directamente a aplay para reproducción inmediata
                    if aplay_process and aplay_process.poll() is None:
                        try:
                            aplay_process.stdin.write(chunk)
                            aplay_process.stdin.flush()
                        except BrokenPipeError:
                            print(f"[{ptype}_{run_id}] ⚠️ aplay pipe broken")
                            break
                    
                    if chunk_count % 10 == 0:
                        print(f"[{ptype}_{run_id}] ← chunk#{chunk_count} streaming...")
            
            # Cerrar pipe para terminar reproducción
            if aplay_process:
                aplay_process.stdin.close()
                aplay_process.wait(timeout=5)
            
            print(f"[{ptype}_{run_id}] ← completed {len(audio_buffer)} B in {chunk_count} chunks")
            
        except Exception as play_error:
            print(f"[{ptype}_{run_id}] ⚠️ streaming playback error: {play_error}")
        finally:
            # Limpiar proceso aplay si aún está corriendo
            if aplay_process and aplay_process.poll() is None:
                try:
                    aplay_process.terminate()
                    aplay_process.wait(timeout=2)
                except:
                    try:
                        aplay_process.kill()
                    except:
                        pass
        
        # Guardar archivo WAV completo para respaldo
        try:
            with wave.open(fn, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_buffer)
            print(f"[{ptype}_{run_id}] ➔ saved {fn}")
        except Exception as save_error:
            print(f"[{ptype}_{run_id}] ⚠️ save error: {save_error}")
        
    except Exception as e:
        print(f"⚠️ Error en run_session_sync: {e}")
    finally:
        # Limpieza final
        try:
            subprocess.run(["pkill", "-f", "aplay"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=1)
        except:
            pass
        


async def warmup()->  tuple[aiohttp.ClientSession, asyncio.Semaphore]: 
    semph = asyncio.Semaphore(MAX_REQUESTS_PER_PROCESS)
    connector = aiohttp.TCPConnector(limit_per_host=128, limit=128)

    session =  aiohttp.ClientSession(connector=connector)
        # 2) Hacemos warmup UNA sola vez
    await run_session(session, "calentamiento", "warmup", 90, semph)
    return session, semph




async def run_session(
    session: aiohttp.ClientSession,
    prompt: str,
    ptype: str,
    run_id: int,
    semaphore: asyncio.Semaphore,
) -> None:
    """Wrap a single prompt run in its own error‐safe block and save audio as WAV."""
    label = f"{ptype}_run{run_id}"
    async with semaphore:
        try:
            # send the request
            payload = {"prompt": f"{prompt}"}
            buf = await stream_to_buffer(session, label, payload)
            if not buf:
                print(f"[{label}] 🛑 no data received")
                return

            # ensure the Outputs directory exists
            output_dir = os.path.join(os.path.dirname(__file__), "Outputs")
            os.makedirs(output_dir, exist_ok=True)

            # write the entire buffer into a WAV file inside Outputs/
            fn = os.path.join(output_dir, f"output_{ptype}_run{run_id}.wav")
            with wave.open(fn, "wb") as wf:
                wf.setnchannels(1)  # mono
                wf.setsampwidth(2)  # 16-bit samples
                wf.setframerate(24000)  # 24 kHz sample rate
                wf.writeframes(buf)
            print(f"[{label}] ➔ saved {fn}")

        except Exception as e:
            print(f"[{label}] 🛑 failed: {e!r}")



async def main():
    global global_session, global_semph , loop
    global_session, global_semph = await warmup()
    loop = asyncio.get_running_loop()
    load_silero_model()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", PORT))
    s.listen(1)
    print(f"🎙️ Servidor escuchando TCP en {PORT}...")

    arecord_proc = start_arecord()
    if arecord_proc is None:
        raise RuntimeError("No se pudo iniciar arecord.")
    print("[🎙️] arecord | nc iniciado.")
    print(arecord_proc)
    while True:
        conn, addr = s.accept()
        print(f"[+] Conexión desde {addr}")
        
        threading.Thread(
            target=handle_connection,
            args=(conn, global_session, global_semph, )
        ).start()
        await global_session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("[🛑] Proceso terminado por el usuario.")

