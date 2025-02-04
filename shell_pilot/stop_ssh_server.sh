#!/bin/bash

PORT="22"  # Puerto por default :+ 

echo "[+] Deteniendo el servidor SSH..."

# Detener el servicio SSH
if systemctl is-active --quiet ssh; then
    sudo systemctl stop ssh
    echo "[✓] Servicio SSH detenido."
else
    echo "[!] SSH ya estaba detenido."
fi

# Deshabilitar SSH para que no inicie con el sistema
sudo systemctl disable ssh
echo "[✓] SSH deshabilitado en el arranque."

# Configurar el firewall (UFW) para bloquear SSH
if command -v ufw &> /dev/null; then
    echo "[+] Bloqueando el puerto $PORT en el firewall..."
    sudo ufw deny $PORT/tcp
    sudo ufw reload
    echo "[✓] Puerto $PORT bloqueado en el firewall."
else
    echo "[!] UFW no está instalado. Si usás otro firewall, bloqueá el puerto manualmente."
fi

echo "[+] SSH ha sido detenido y bloqueado correctamente."
exit 0
