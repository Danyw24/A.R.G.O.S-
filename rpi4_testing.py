from gpiozero import LED, Button
from signal import pause
import time

# Configuración de pines
led1 = LED(17)  # LED 1 - GPIO 17
led2 = LED(27)  # LED 2 - GPIO 18  
led3 = LED(22)  # LED 3 - GPIO 22
boton = Button(4)  # Botón - GPIO 2 (con pull-up interno)

# Estado del sistema
muted = False

def toggle_mute():
    """Alterna entre mute/unmute al presionar el botón"""
    global muted
    
    # Cambiar estado
    muted = not muted
    
    if muted:
        print("🔇 MUTE - LEDs apagados")
        # Apagar todos los LEDs
        led1.off()
        led2.off()
        led3.off()
        
        # Aquí puedes agregar tu función de mute del micrófono
        # mute_microphone()
        
    else:
        print("🎤 UNMUTE - LEDs encendidos")
        # Encender todos los LEDs
        led1.on()
        led2.on()
        led3.on()
        
        # Aquí puedes agregar tu función de unmute del micrófono  
        # unmute_microphone()

# Configurar callback del botón
boton.when_pressed = toggle_mute

# Estado inicial - UNMUTE (LEDs encendidos)
print("🎤 Sistema iniciado - UNMUTE")
led1.on()
led2.on()
led3.on()

print("Presiona el botón para alternar MUTE/UNMUTE")
print("Ctrl+C para salir")

try:
    # Mantener el programa corriendo
    pause()
except KeyboardInterrupt:
    print("\n🛑 Saliendo...")
    # Apagar todos los LEDs al salir
    led1.off()
    led2.off()
    led3.off()
    print("LEDs apagados - Programa terminado")