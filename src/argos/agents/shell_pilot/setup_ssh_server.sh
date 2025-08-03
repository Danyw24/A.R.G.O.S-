#!/bin/bash

# Configuración
PORT="22"  # Puerto default en ssh

echo "[+] Iniciando configuración del servidor SSH..."

# Verificado si OpenSSH Server está instalado
if ! command -v sshd &> /dev/null; then
    echo "[+] OpenSSH Server no está instalado. Instalando..."
    sudo apt update && sudo apt install -y openssh-server
else
    echo "[✓] OpenSSH Server ya está instalado."
fi

# Verificando si el servicio está activo
if systemctl is-active --quiet ssh; then
    echo "[✓] SSH ya está en ejecución."
else
    echo "[+] Iniciando el servicio SSH..."
    sudo systemctl start ssh
fi

# Habilitar SSH para que inicie con el sistema
sudo systemctl enable ssh

# Configurar el firewall (UFW) para permitir SSH
if command -v ufw &> /dev/null; then
    echo "[+] Configurando firewall para permitir conexiones SSH..."
    sudo ufw allow $PORT/tcp
    sudo ufw reload
else
    echo "[!] UFW no está instalado. Asegúrate de permitir el puerto $PORT manualmente."
fi

# Verificar si el puerto está abierto
echo "[+] Verificando si el puerto $PORT está abierto..."
if ss -tuln | grep -q ":$PORT"; then
    echo "[✓] El puerto $PORT está abierto y escuchando conexiones SSH."
else
    echo "[✗] El puerto $PORT no está abierto. Revisa la configuración manualmente."
fi

# Mostrar la IP del servidor
echo "[+] Configuración completa. Puedes conectarte con:"
echo "    ssh usuario@$(hostname -I | awk '{print $1}')"

exit 0
