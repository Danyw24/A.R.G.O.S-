import numpy as np
import sys
from scipy.signal import butter, lfilter, iirnotch
import time # Para depuración si es necesario

# --- PARÁMETROS GLOBALES ---
SAMPLE_RATE = 16000
CHUNK_SIZE_MS = 64
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SIZE_MS / 1000)
CHUNK_BYTES = CHUNK_SAMPLES * 2

# --- CADENA DE PROCESAMIENTO: AJUSTA ESTOS VALORES ---

# 1. Filtro Pasa-banda (Voz)
LOWCUT_FREQ = 250.0
HIGHCUT_FREQ = 3500.0
FILTER_ORDER = 4

# 2. Filtros Notch (Ruido Eléctrico)
NOTCH_FREQS = [60.0, 120.0, 180.0]
NOTCH_Q = 35.0

# 3. Reductor de Transitorios (NUEVO) - Suaviza los picos agudos
# Umbral en la escala int16. Si un pico supera esto, se reduce.
# Ajústalo para que sea más alto que tu voz normal pero más bajo que los golpes.
TRANSIENT_THRESHOLD = 8000.0 
# Cuánto reducir los picos que superan el umbral. 2.0 es una reducción suave.
TRANSIENT_RATIO = 2.5 

# 4. Puerta de Ruido (Noise Gate)
GATE_THRESHOLD = 800.0
GATE_ATTENUATION = 0.1

# 5. Compresor - Nivela el volumen de tu voz
COMPRESSOR_THRESHOLD_DB = -18.0
COMPRESSOR_RATIO = 6.0
COMPRESSOR_ATTACK_MS = 5.0 # Podemos volver a un ataque más natural
COMPRESSOR_RELEASE_MS = 150.0

# 6. Ganancia de Salida (Makeup Gain)
MAKEUP_GAIN_DB = 38.0

# 7. Limitador
LIMITER_THRESHOLD_DB = -1.0

# --- CLASES Y FUNCIONES DE PROCESAMIENTO ---

def design_filters():
    filters = []
    nyquist = 0.5 * SAMPLE_RATE
    low = LOWCUT_FREQ / nyquist
    high = HIGHCUT_FREQ / nyquist
    b, a = butter(FILTER_ORDER, [low, high], btype='band')
    filters.append({'b': b, 'a': a, 'state': np.zeros(max(len(a), len(b)) - 1)})
    for freq in NOTCH_FREQS:
        if freq < nyquist:
            w = freq / nyquist
            b, a = iirnotch(w, NOTCH_Q)
            filters.append({'b': b, 'a': a, 'state': np.zeros(max(len(a), len(b)) - 1)})
    return filters

def float_to_db(x):
    # Evita log(0)
    safe_x = np.maximum(1e-9, np.abs(x))
    return 20 * np.log10(safe_x / 32767.0)

def db_to_float(db):
    return 10**(db / 20.0)

# (NUEVA FUNCIÓN)
def apply_transient_reduction(signal_float, threshold, ratio):
    """
    Reduce picos rápidos (transitorios) que superan un umbral.
    Actúa como un compresor súper rápido y simple.
    """
    # Identifica las muestras que superan el umbral
    above_threshold_mask = np.abs(signal_float) > threshold
    
    # Para esas muestras, calcula la cantidad a reducir
    # La fórmula es: umbral + (distancia_desde_el_umbral / ratio)
    reduced_amplitude = threshold + (np.abs(signal_float[above_threshold_mask]) - threshold) / ratio
    
    # Aplica la reducción manteniendo el signo original
    signal_float[above_threshold_mask] = np.sign(signal_float[above_threshold_mask]) * reduced_amplitude
    
    return signal_float

class AudioProcessor:
    def __init__(self):
        self.gate_open = True
        self.compressor_gain = 1.0

    def process_chunk(self, signal_float):
        # --- 4. Puerta de Ruido (Noise Gate) ---
        if np.max(np.abs(signal_float)) < GATE_THRESHOLD:
            if self.gate_open:
                signal_float *= np.linspace(1, GATE_ATTENUATION, CHUNK_SAMPLES, dtype=np.float32)
            else:
                signal_float *= GATE_ATTENUATION
            self.gate_open = False
        else:
            if not self.gate_open:
                signal_float *= np.linspace(GATE_ATTENUATION, 1, CHUNK_SAMPLES, dtype=np.float32)
            self.gate_open = True

        # --- 5. Compresor ---
        thresh_linear = db_to_float(COMPRESSOR_THRESHOLD_DB) * 32767.0
        attack_samples = max(1, (COMPRESSOR_ATTACK_MS / 1000.0) * SAMPLE_RATE)
        release_samples = max(1, (COMPRESSOR_RELEASE_MS / 1000.0) * SAMPLE_RATE)
        alpha_attack = np.exp(-1.0 / attack_samples)
        alpha_release = np.exp(-1.0 / release_samples)

        output_signal = np.zeros_like(signal_float)
        for i in range(CHUNK_SAMPLES):
            sample = signal_float[i]
            sample_db = float_to_db(sample)
            
            if sample_db > COMPRESSOR_THRESHOLD_DB:
                target_db = COMPRESSOR_THRESHOLD_DB + (sample_db - COMPRESSOR_THRESHOLD_DB) / COMPRESSOR_RATIO
                target_gain = db_to_float(target_db - sample_db)
            else:
                target_gain = 1.0
            
            if target_gain < self.compressor_gain:
                self.compressor_gain = alpha_attack * self.compressor_gain + (1 - alpha_attack) * target_gain
            else:
                self.compressor_gain = alpha_release * self.compressor_gain + (1 - alpha_release) * target_gain
            
            output_signal[i] = sample * self.compressor_gain
        
        # --- 6. Ganancia de Salida (Makeup Gain) ---
        makeup_gain_linear = db_to_float(MAKEUP_GAIN_DB)
        amplified_signal = output_signal * makeup_gain_linear

        # --- 7. Limitador ---
        limiter_thresh_linear = db_to_float(LIMITER_THRESHOLD_DB)
        clipped_signal = 32767.0 * limiter_thresh_linear * np.tanh(amplified_signal / (32767.0 * limiter_thresh_linear))

        return clipped_signal

def main():
    filters = design_filters()
    processor = AudioProcessor()
    
    # ... (el print de estado se puede actualizar para incluir el Reductor de Transitorios)
    print("🎙️ Filtro de audio profesional v2 iniciado:", file=sys.stderr)
    print(f"  - Cadena: Filtros -> Reductor Transitorios -> Gate -> Compresor -> Ganancia -> Limitador", file=sys.stderr)
    print(f"  - Reductor de Transitorios: Ratio {TRANSIENT_RATIO}:1 @ {TRANSIENT_THRESHOLD}", file=sys.stderr)


    try:
        while True:
            in_bytes = sys.stdin.buffer.read(CHUNK_BYTES)
            if len(in_bytes) != CHUNK_BYTES:
                break
            
            signal_int16 = np.frombuffer(in_bytes, dtype=np.int16)
            signal_float = signal_int16.astype(np.float32)
            
            # --- 1. y 2. Aplicar Filtros Correctivos ---
            current_signal = signal_float
            for f in filters:
                current_signal, f['state'] = lfilter(f['b'], f['a'], current_signal, zi=f['state'])

            # --- 3. Aplicar Reductor de Transitorios (NUEVO) ---
            tamed_signal = apply_transient_reduction(current_signal, TRANSIENT_THRESHOLD, TRANSIENT_RATIO)

            # --- 4 a 7. Aplicar el resto de la cadena de procesamiento ---
            processed_signal_float = processor.process_chunk(tamed_signal)
            
            processed_signal_int16 = np.clip(
                processed_signal_float, -32768, 32767
            ).astype(np.int16)
            
            sys.stdout.buffer.write(processed_signal_int16.tobytes())
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        print("\n🛑 Filtro detenido por el usuario", file=sys.stderr)
    except Exception as e:
        print(f"\n❌ Error en filtro: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()