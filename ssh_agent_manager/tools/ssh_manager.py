from agency_swarm import BaseTool
from pydantic import Field
from typing import List
import paramiko
import asyncio

 
class ssh_manager(BaseTool):
    """
    Herramienta para manejar comunicación y control via ssh
    """
    host: str = Field(..., description="Direccion ip del host en formato ipv4, ejemplo: 192.168.0.22")
    user: str = Field(..., description="Nombre de usuario")
    password: str = Field(..., description="Contraseña")
    commands: List[str] = Field(..., description="Lista de comandos linux")

    async def run(self):
        try:
            tareas = [self.execute_remote_command(command= cmd) for cmd in self.commands]
            stdoutput = await asyncio.gather(*tareas)  # Ejecuta múltiples comandos en paralelo
            return stdoutput

        except Exception as e:
            return [f"Error al ejecutar comando remoto: {e}"]


    async def execute_remote_command(self, command):
        """Ejecuta un comando en la máquina remota vía SSH"""
     
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            client.connect(hostname=self.host, username=self.user, password=self.password, timeout=5)
            stdin, stdout, stderr = client.exec_command(command)

            salida = stdout.read().decode()
            error = stderr.read().decode()

            if error:
                return f"[ERROR en {self.host}]: {error}"
            else:
                return f"[{self.host}]: {salida}"

        except Exception as e:
            return f"Error en la conexión con {self.host}: {e}"
        finally:
            client.close()
            
