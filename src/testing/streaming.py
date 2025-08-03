import requests
import pyaudio
import wave
import time
import os


# ——— Request parameters ———

max_tokens = 2000

def stream_audio(text: str, voice: str):
    """
    Stream the audio directly to your speakers
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    resp = requests.post(
    "https://model-4w5rzzpq.api.baseten.co/environments/production/predict",
    headers={"Authorization": "Api-Key tMl"},
    json={'voice': 'tara', 'prompt': 'In todays fast-paced world, finding balance between work and personal life is more important than ever. With the constant demands of technology, remote communication, ', 'max_tokens': 10000},
    )

    resp.raise_for_status()

    for chunk in resp.iter_content(chunk_size=4096):
        if chunk:
            stream.write(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()


def save_audio(text: str, voice: str, output_path: str = "output.wav"):
    """
    Save the audio to a WAV file.
    """
    start_time = time.monotonic()

    resp = requests.post(
        f"https://model-{orpheus_model_id}.api.baseten.co/environments/production/predict",
        headers={"Authorization": f"Api-Key {api_key}"},
        json={"voice": voice, "prompt": text, "max_tokens": max_tokens},
        stream=False,
    )
    resp.raise_for_status()

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        chunk_counter = 0

        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            chunk_counter += 1
            frame_count = len(chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(chunk)

        duration = total_frames / wf.getframerate()

    end_time = time.monotonic()
    elapsed = end_time - start_time
    print(f"Generated {duration:.2f}s of audio in {chunk_counter} chunks in {elapsed:.2f}s.")


if __name__ == "__main__":
    voice = "tara"

    original_text = """
    Nothing beside remains. Round the decay of that colossal wreck, boundless and bare,
    The lone and level sands stretch far away.
    """

    print("🔊 Streaming live:")
    stream_audio(original_text, voice)

    print("\n💾 Saving to output.wav:")
    save_audio(original_text, voice)

    print("Done!")





















from orpheus_tts import OrpheusModel
import wave
import time
model = OrpheusModel(model_name ="danyw24/argos-4b-0.2-es")




prompt = "Los psicodélicos muestran que la conciencia no es algo fijo, sino un patrón de actividad cerebral. Estudios con psilocibina y LSD revelan que al aumentar la conectividad y desactivar la red por defecto, el sentido del yo se desarma. Así se difuminan los límites entre sujeto y objeto, dejando en claro que nuestra realidad es una construcción flexible, no una verdad fija."
start_time = time.monotonic()
syn_tokens = model.generate_speech(
            prompt=prompt,
            repetition_penalty=1.1,
            stop_token_ids=[128258],
            max_tokens=3072,
            temperature=0.8,
            top_p=0.9
        )

with wave.open("output.wav", "wb") as wf:
   wf.setnchannels(1)
   wf.setsampwidth(2)
   wf.setframerate(24000)

   total_frames = 0
   chunk_counter = 0
   for audio_chunk in syn_tokens: # output streaming
      chunk_counter += 1
      frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
      total_frames += frame_count
      wf.writeframes(audio_chunk)
   duration = total_frames / wf.getframerate()

   end_time = time.monotonic()
   print(f"It took {end_time - start_time} seconds to generate {duration:.2f} seconds of audio")

from IPython.display import Audio

# Reproduce el archivo de audio subido
Audio("output.wav")  # puede ser .wav, .ogg, etc.
