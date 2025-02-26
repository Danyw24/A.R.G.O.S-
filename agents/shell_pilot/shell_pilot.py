import argparse
import os
import platform
import subprocess

def system_info():
    return {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Architecture": platform.architecture()[0],
        "Processor": platform.processor(),
        "Hostname": platform.node(),
        "Uptime": subprocess.getoutput("uptime") if platform.system() == "Linux" else "N/A"
    }

# argumentos
parser = argparse.ArgumentParser(description="Shell Pilot - Información del Sistema")
parser.add_argument("command", help="Comando a ejecutar (info, cpu, os, hostname)")

args = parser.parse_args()

# Ejecutar según el argumento recibido
if args.command == "info":
    print(system_info())
elif args.command == "cpu":
    print(f"Processor: {platform.processor()}")
elif args.command == "os":
    print(f"OS: {platform.system()} - Version: {platform.version()}")
elif args.command == "hostname":
    print(f"Hostname: {platform.node()}")
else:
    print("Comando no reconocido. Usa: info, cpu, os, hostname")


print(system_info())
