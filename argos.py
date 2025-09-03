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
import re
import signal
from gpiozero import LED, Button
import time
from threading import Lock

# Variables para debounce
last_press_time = 0
debounce_delay = 0.3  # 300ms entre presiones válidas
toggle_lock = Lock()  # Para thread safety

led1 = LED(17)  # LED 1 - GPIO 17
led2 = LED(27)  # LED 2 - GPIO 27  
boton = Button(4)  # Botón - GPIO 2 (con pull-up interno)



PORT = 4300
CHUNK_SEC = 2.0
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_SIZE = int(SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_SEC)
API_URL = "http://127.0.0.1:8080/transcribe/"
API_KEY = "Bearer TU_API_KEY"


# Configuration
MODEL = "232k54x3"
BASETEN_HOST = f"https://model-nwxn27z3.api.baseten.co/deployment/qzmvrkw/predict"
BASETEN_API_KEY = "nc1GV4ms.1Uu1sTz51VVph6dQvU0v7lZWwVKEJBEd"
PAYLOADS_PER_PROCESS = 1
NUM_PROCESSES = 1
MAX_REQUESTS_PER_PROCESS = 8
muted = False

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


def toggle_mute():
    """Alterna entre mute/unmute con debounce para evitar múltiples activaciones"""
    global muted, last_press_time
    
    current_time = time.time()
    
    with toggle_lock:
        if current_time - last_press_time < debounce_delay:
            print("⏭️ Presión ignorada (debounce)")
            return
        last_press_time = current_time
        
        muted = not muted
        
        if muted:
            print("🔇 MUTE")
            led2.on()
            try:
                mute_microphone()
            except Exception as e:
                print(f"❌ Error en mute: {e}")
        else:
            print("🎤 UNMUTE")
            led2.off()
            try:
                unmute_microphone()
            except Exception as e:
                print(f"❌ Error en unmute: {e}")
        time.sleep(0.1)


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
    FRAME_DURATION = 0.064  # Duración de cada chunk de audio en segundos
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION * BYTES_PER_SAMPLE)
    
    # Parámetros de calibración
    CALIB_SEC = 2.0  # Duración de la calibración en segundos
    CALIB_FRAMES = int(CALIB_SEC / FRAME_DURATION)
    
    # --- MEJORA: Parámetros de Umbral Adaptativo Simple ---
    NOISE_OFFSET = 250.0 
    FALLBACK_THRESHOLD = 1850.0
    
    # Parámetros de detección de habla (VAD)
    MIN_SPEECH_DURATION_MS = 200
    MAX_SILENCE_DURATION_MS = 700
    MIN_SPEECH_FRAMES = int(MIN_SPEECH_DURATION_MS / (FRAME_DURATION * 1000))
    MAX_SILENCE_FRAMES = int(MAX_SILENCE_DURATION_MS / (FRAME_DURATION * 1000))
    
    # Confirmación de voz para evitar falsos positivos
    SPEECH_CONFIRMATION_FRAMES = 2
    
    # Parámetros de análisis VAD
    MIN_AUDIO_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.4)
    
    # --- NUEVOS PARÁMETROS PARA SILERO VAD ---
    VAD_THRESHOLD = 0.6  # Umbral de confianza para Silero VAD (0.0-1.0)
    MIN_VAD_SPEECH_DURATION_MS = 150  # Duración mínima de segmento confirmado por VAD
    VAD_PADDING_MS = 50  # Padding antes/después de segmentos VAD
# ==========================================
    # INICIALIZACIÓN DE VARIABLES
    # ==========================================
    
    buffer = b""
    speech_buffer = b""
    calibration_buffer = b""
    processed_segments = 0
    silence_frames_after_speech = 0
    speech_frame_count = 0
    frames_collected = 0
    
    # Estado de detección y contador para confirmación
    is_speaking = False
    potential_speech_frames = 0
    
    # Almacenamiento de muestras RMS para la calibración
    rms_samples = []
    
    SILENCE_THRESHOLD = None
    is_calibrated = False

    # === VARIABLES DE DEPURACIÓN ===
    debug_frames_received = 0
    debug_valid_frames = 0
    debug_rms_below_50 = 0
    debug_buffer_underruns = 0
    debug_processing_errors = 0

    # ==========================================
    # FASE 1: CALIBRACIÓN ADAPTATIVA
    # ==========================================
    
    print("🔧 Iniciando calibración de umbral adaptativo...")
    print(f"   ⏱️ Duración: {CALIB_SEC}s ({CALIB_FRAMES} frames)")
    print("   🤫 Por favor, mantén silencio durante la calibración...")
    
    try:
        start_time = time.time()
        while frames_collected < CALIB_FRAMES:
            try:
                data = conn.recv(8192)
                if not data:
                    print("❌ Error: Conexión cerrada durante la calibración.")
                    return
                
                calibration_buffer += data
                debug_frames_received += 1
                
                # === DEPURACIÓN: Verificar tamaño del buffer ===
                if debug_frames_received % 10 == 0:
                    print(f"   📊 Debug - Buffer: {len(calibration_buffer)} bytes, Frames recibidos: {debug_frames_received}")
                
                while len(calibration_buffer) >= FRAME_SIZE and frames_collected < CALIB_FRAMES:
                    frame = calibration_buffer[:FRAME_SIZE]
                    calibration_buffer = calibration_buffer[FRAME_SIZE:]
                    
                    try:
                        audio_np = np.frombuffer(frame, dtype=np.int16)
                        debug_valid_frames += 1
                        
                        if len(audio_np) > 0:
                            rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                            
                            # === DEPURACIÓN: Analizar RMS ===
                            if frames_collected % 5 == 0:  # Cada 5 frames
                                print(f"   🔍 Frame {frames_collected}: RMS={rms:.1f}, Audio samples={len(audio_np)}")
                            
                            if rms > 50:
                                rms_samples.append(rms)
                                if len(rms_samples) % 3 == 0:  # Cada 3 muestras válidas
                                    print(f"   ✅ Muestras RMS válidas: {len(rms_samples)}/15 requeridas")
                            else:
                                debug_rms_below_50 += 1
                                
                            frames_collected += 1
                    except Exception as e:
                        debug_processing_errors += 1
                        print(f"   ⚠️ Error procesando frame {frames_collected}: {e}")
                        continue
            
            except BlockingIOError:
                debug_buffer_underruns += 1
                time.sleep(0.01)
                continue
            except Exception as e:
                print(f"   ⚠️ Error de red durante calibración: {e}")
                break

        # === DEPURACIÓN FINAL ===
        elapsed_time = time.time() - start_time
        print(f"\n📈 ESTADÍSTICAS DE CALIBRACIÓN:")
        print(f"   ⏱️ Tiempo transcurrido: {elapsed_time:.2f}s")
        print(f"   📦 Frames recibidos de red: {debug_frames_received}")
        print(f"   ✅ Frames válidos procesados: {debug_valid_frames}")
        print(f"   🎯 Frames objetivo: {CALIB_FRAMES}")
        print(f"   📊 Muestras RMS válidas (>50): {len(rms_samples)}")
        print(f"   📉 RMS muy bajos (<50): {debug_rms_below_50}")
        print(f"   🔄 Buffer underruns: {debug_buffer_underruns}")
        print(f"   ❌ Errores de procesamiento: {debug_processing_errors}")
        
        if len(rms_samples) > 0:
            print(f"   📋 Primeras RMS: {[f'{x:.1f}' for x in rms_samples[:10]]}")
            print(f"   📈 RMS min/max: {min(rms_samples):.1f} / {max(rms_samples):.1f}")

        # Cálculo del umbral
        if len(rms_samples) >= 15:
            average_noise = np.mean(rms_samples)
            SILENCE_THRESHOLD = average_noise + NOISE_OFFSET
            is_calibrated = True
            
            print(f"\n[✅] Calibración completada:")
            print(f"   🔊 Ruido ambiente promedio (RMS): {average_noise:.1f}")
            print(f"   🎯 Umbral de silencio adaptativo: {SILENCE_THRESHOLD:.1f}")
        else:
            print(f"\n❌ DIAGNÓSTICO: Insuficientes muestras RMS válidas:")
            print(f"   📊 Obtenidas: {len(rms_samples)} / Requeridas: 15")
            
            # Diagnóstico específico
            if debug_frames_received < 10:
                print(f"   🚨 CAUSA: Muy pocos frames de red recibidos ({debug_frames_received})")
            elif debug_valid_frames < CALIB_FRAMES // 2:
                print(f"   🚨 CAUSA: Muchos frames corruptos o mal formateados")
            elif debug_rms_below_50 > debug_valid_frames * 0.8:
                print(f"   🚨 CAUSA: Audio demasiado silencioso (RMS < 50)")
                print(f"   💡 SOLUCIÓN: Aumentar ganancia del micrófono")
            elif debug_processing_errors > debug_valid_frames * 0.3:
                print(f"   🚨 CAUSA: Errores de procesamiento numpy")
            else:
                print(f"   🚨 CAUSA: Calibración muy lenta o timeout")
            
            print(f"   🔄 Usando umbral de respaldo: {FALLBACK_THRESHOLD}")
            SILENCE_THRESHOLD = FALLBACK_THRESHOLD
            is_calibrated = True
    
    except Exception as e:
        print(f"❌ Error en calibración: {e}. Usando umbral de emergencia.")
        print(f"🔍 Tipo de error: {type(e).__name__}")
        SILENCE_THRESHOLD = FALLBACK_THRESHOLD
        is_calibrated = True
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
            
                if is_speaking:
                    # ESTADO: YA ESTAMOS GRABANDO VOZ
                    if rms > SILENCE_THRESHOLD:
                        speech_buffer += frame
                        speech_frame_count += 1
                        silence_frames_after_speech = 0
                    else:
                        silence_frames_after_speech += 1
                        speech_buffer += frame

                        if silence_frames_after_speech >= MAX_SILENCE_FRAMES:
                            print("🔇", end="", flush=True)
                            
                            # INTEGRACIÓN DE SILERO VAD - VALIDACIÓN FINAL
                            if speech_frame_count >= MIN_SPEECH_FRAMES and len(speech_buffer) >= MIN_AUDIO_BYTES:
                                
                                # Convertir buffer a numpy array para Silero VAD
                                try:
                                    speech_audio_np = np.frombuffer(speech_buffer, dtype=np.int16).astype(np.float32)
                                    # Normalizar a rango [-1, 1] para Silero
                                    speech_audio_normalized = speech_audio_np / 32768.0
                                    
                                    # APLICAR SILERO VAD para confirmar si realmente hay voz
                                    print(f"\n🤖 Aplicando Silero VAD (umbral: {VAD_THRESHOLD})...")
                                    
                                    vad_segments = get_speech_timestamps_onnx(
                                        speech_audio_normalized,
                                        sample_rate=SAMPLE_RATE,
                                        threshold=VAD_THRESHOLD,
                                        min_speech_duration_ms=MIN_VAD_SPEECH_DURATION_MS,
                                        min_silence_duration_ms=100,
                                        window_size_samples=512,
                                        padding_ms=VAD_PADDING_MS
                                    )
                                    
                                    if vad_segments and len(vad_segments) > 0:
                                        # SILERO CONFIRMA QUE HAY VOZ
                                        processed_segments += 1
                                        segment_duration = len(speech_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                                        total_vad_duration = sum((seg['end'] - seg['start']) / SAMPLE_RATE for seg in vad_segments)
                                        
                                        print(f"✅ VAD confirmó voz: {len(vad_segments)} segmento(s)")
                                        print(f"📝 Segmento {processed_segments} ({segment_duration:.2f}s total, {total_vad_duration:.2f}s voz)")
                                        
                                        # Crear WAV para envío - CON VALIDACIONES
                                        wav_buffer = io.BytesIO()
                                        try:
                                            with wave.open(wav_buffer, 'wb') as wf:
                                                wf.setnchannels(1)
                                                wf.setsampwidth(2)
                                                wf.setframerate(SAMPLE_RATE)
                                                wf.writeframes(speech_buffer)
                                            
                                            # Validar que el WAV se creó correctamente
                                            wav_buffer.seek(0)
                                            wav_size = len(wav_buffer.getvalue())
                                            
                                            if wav_size < 1000:  # WAV debe tener al menos 1KB
                                                print(f"❌ WAV muy pequeño ({wav_size} bytes), omitiendo envío")
                                                continue
                                            
                                            print(f"📦 WAV creado: {wav_size} bytes")
                                            wav_buffer.seek(0)  # Reset para lectura
                                            
                                        except Exception as wav_error:
                                            print(f"❌ Error creando WAV: {wav_error}")
                                            continue
                                        
                                        # Enviar a Baseten CON MANEJO DE ERRORES
                                        try:
                                            send_to_baseten(wav_buffer)
                                            print("   📤 Enviado a Baseten exitosamente")
                                        except Exception as send_error:
                                            print(f"   ❌ Error enviando a Baseten: {send_error}")
                                            # No interrumpir el programa por errores de envío
                                        
                                    else:
                                        # SILERO NO DETECTA VOZ - FALSO POSITIVO
                                        print("❌ VAD rechazó el segmento: No hay voz suficiente")
                                        print(f"   📊 RMS promedio era: {np.mean([np.sqrt(np.mean(np.frombuffer(speech_buffer[i:i+FRAME_SIZE], dtype=np.int16).astype(np.float32)**2)) for i in range(0, len(speech_buffer), FRAME_SIZE)]):.1f}")
                                        
                                except Exception as vad_error:
                                    print(f"❌ Error en Silero VAD: {vad_error}")
                                    print("   🔄 Enviando sin validación VAD...")
                                    
                                    # Fallback: enviar sin validación VAD - CON VALIDACIONES
                                    try:
                                        wav_buffer = io.BytesIO()
                                        with wave.open(wav_buffer, 'wb') as wf:
                                            wf.setnchannels(1)
                                            wf.setsampwidth(2)
                                            wf.setframerate(SAMPLE_RATE)
                                            wf.writeframes(speech_buffer)
                                        
                                        # Validar WAV antes de envío
                                        wav_buffer.seek(0)
                                        wav_size = len(wav_buffer.getvalue())
                                        
                                        if wav_size >= 1000:  # Mínimo 1KB
                                            processed_segments += 1
                                            segment_duration = len(speech_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                                            print(f"📝 Segmento {processed_segments} ({segment_duration:.2f}s) - Sin validación VAD")
                                            
                                            wav_buffer.seek(0)
                                            send_to_baseten(wav_buffer)
                                            print("   📤 Enviado a Baseten (fallback)")
                                        else:
                                            print(f"❌ WAV fallback muy pequeño ({wav_size} bytes), omitido")
                                            
                                    except Exception as fallback_error:
                                        print(f"❌ Error en fallback: {fallback_error}")
                            else:
                                # Segmento muy corto, descartado antes del VAD
                                print(f"\n⏭️ Segmento descartado (demasiado corto: {len(speech_buffer)} bytes)")

                            # Reset de estado
                            is_speaking = False
                            speech_buffer = b""
                            speech_frame_count = 0
                            silence_frames_after_speech = 0
                            potential_speech_frames = 0
                else:
                    # ESTADO: ESPERANDO VOZ
                    if rms > SILENCE_THRESHOLD:
                        potential_speech_frames += 1
                        speech_buffer += frame

                        if potential_speech_frames >= SPEECH_CONFIRMATION_FRAMES:
                            is_speaking = True
                            speech_frame_count = potential_speech_frames
                            potential_speech_frames = 0
                            print("🗣️", end="", flush=True)
                            led1.on()
                    else:
                        potential_speech_frames = 0
                        speech_buffer = b""
                        led1.off()

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
        # Detectar dispositivo USB automáticamente
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True, check=True)
        usb_match = re.search(r'card\s+(\d+):\s+.*(?:USB|MIC).*device\s+(\d+):', result.stdout, re.IGNORECASE)
        device = f"plughw:{usb_match[1]},{usb_match[2]}" if usb_match else "default"
        
        print(f"[🎤] Usando dispositivo: {device}")
        
        # Probar con filtro primero
        for cmd in [
            f"arecord -f S16_LE -c1 -r 16000 -t raw -D {device} | python audio_filter.py | nc 127.0.0.1 4300",
            f"arecord -f S16_LE -c1 -r 16000 -t raw -D {device} | nc 127.0.0.1 4300",
            "arecord -f S16_LE -c1 -r 16000 -t raw | nc 127.0.0.1 4300"
        ]:
            try:
                return subprocess.Popen(cmd, shell=True)
            except: continue
                
    except:
        return subprocess.Popen("arecord -f S16_LE -c1 -r 16000 -t raw | nc 127.0.0.1 4300", shell=True)
    
    
    
def mute_microphone():
    """Silencia el micrófono pausando todo el pipeline de audio"""
    try:
        # Buscar el proceso shell que ejecuta el pipeline completo
        result = subprocess.run(["pgrep", "-f", "arecord.*audio_filter"], 
                                capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    # Pausar el proceso shell y todos sus hijos
                    os.kill(int(pid), signal.SIGSTOP)
                    # También pausar procesos hijos (arecord, python, nc)
                    subprocess.run(f"pkill -STOP -P {pid}", shell=True, 
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except:
                    continue
            print("🔇 Pipeline de audio silenciado")
            return True
        else:
            # Fallback: buscar solo arecord
            result2 = subprocess.run(["pgrep", "-f", "arecord"], 
                                        capture_output=True, text=True, timeout=2)
            if result2.returncode == 0 and result2.stdout.strip():
                pids = result2.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGSTOP)
                    except:
                        continue
                print("🔇 Micrófono silenciado (fallback)")
                return True
            else:
                print("⚠️ No se encontraron procesos de audio")
                return False
    except Exception as e:
        print(f"❌ Error silenciando micrófono: {e}")
        return False
    
    

def unmute_microphone():
    """Reactiva el micrófono reanudando todo el pipeline de audio"""
    try:
        # Buscar el proceso shell que ejecuta el pipeline completo
        result = subprocess.run(["pgrep", "-f", "arecord.*audio_filter"], 
                                capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    # Reanudar el proceso shell y todos sus hijos
                    os.kill(int(pid), signal.SIGCONT)
                    # También reanudar procesos hijos (arecord, python, nc)
                    subprocess.run(f"pkill -CONT -P {pid}", shell=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except:
                    continue
            print("🎤 Pipeline de audio reactivado")
            return True
        else:
            # Fallback: buscar solo arecord
            result2 = subprocess.run(["pgrep", "-f", "arecord"], 
                                    capture_output=True, text=True, timeout=2)
            if result2.returncode == 0 and result2.stdout.strip():
                pids = result2.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGCONT)
                    except:
                        continue
                print("🎤 Micrófono reactivado (fallback)")
                return True
            else:
                print("⚠️ No se encontraron procesos de audio")
                return False
    except Exception as e:
        print(f"❌ Error reactivando micrófono: {e}")
        return False
    
            
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
    boton.when_pressed = toggle_mute
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

