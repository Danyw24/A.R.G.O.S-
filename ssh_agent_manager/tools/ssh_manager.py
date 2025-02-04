from agency_swarm import BaseTool
from pydantic import Field, field_validator, IPvAnyAddress
from typing import List, Dict, Optional
import asyncssh
import asyncio
import logging
from logging import Logger
from contextlib import asynccontextmanager

class ssh_manager(BaseTool):
    """
    Herramienta segura para ejecución remota de comandos SSH con gestión avanzada de conexiones
    
    Características:
    - Conexiones persistentes reutilizables
    - Validación de entrada estricta
    - Soporte para autenticación por clave SSH
    - Timeouts configurables
    - Ejecución paralela segura
    - Registro detallado de operaciones
    
    Ejemplo de uso:
    ```python
    tool = SSHTool(
        host="192.168.0.22",
        user="admin",
        auth_type="password",
        password="segura123",
        commands=["ls -l", "df -h"]
    )
    results = await tool.run()
    ```
    """
    
    host: IPvAnyAddress = Field(..., description="Dirección IP válida del host remoto")
    user: str = Field(..., min_length=1, description="Usuario SSH válido")
    auth_type: str = Field("password", description="Tipo de autenticación: 'password' o 'key'")
    password: Optional[str] = Field(None, description="Contraseña para autenticación", min_length=8)
    key_file: Optional[str] = Field(None, description="Ruta a clave SSH privada")
    commands: List[str] = Field(..., min_items=1, description="Lista de comandos a ejecutar")
    connection_timeout: int = Field(10, description="Timeout de conexión en segundos")
    command_timeout: int = Field(15, description="Timeout por comando en segundos")
    max_parallel: int = Field(5, description="Máximo de comandos concurrentes")

    def __init__(self, **data):
        super().__init__(**data)
        self.logger: Logger = logging.getLogger("SSHTool")
        self._validate_auth()
        
    @field_validator('auth_type')
    def validate_auth_type(cls, v):
        if v not in ['password', 'key']:
            raise ValueError("Tipo de autenticación inválido")
        return v

    @field_validator('commands')
    def validate_commands(cls, v):
        forbidden = ['rm ', 'dd ', 'shutdown', 'reboot']
        for cmd in v:
            if any(cmd.startswith(f) for f in forbidden):
                raise ValueError(f"Comando prohibido: {cmd}")
        return v

    def _validate_auth(self):
        if self.auth_type == "password" and not self.password:
            raise ValueError("Se requiere contraseña para autenticación por password")
        if self.auth_type == "key" and not self.key_file:
            raise ValueError("Se requiere archivo de clave para autenticación por key")

    @asynccontextmanager  
    async def _get_connection(self):
        """Gestión segura de conexiones SSH reutilizables"""
        conn_params = {
            "host": str(self.host),
            "username": self.user,
            "password": self.password if self.auth_type == "password" else None,
            "client_keys": [self.key_file] if self.auth_type == "key" else None,
            "known_hosts": None  # En producción usar un archivo de hosts conocidos 
        }
        
        try:
            async with asyncssh.connect(
                **conn_params,
                connect_timeout=self.connection_timeout # establecido en 10 segundos
            ) as conn:
                yield conn
        except asyncssh.Error as e:
            self.logger.error(f"Error de conexión SSH: {str(e)}")
            raise  

    async def _execute_command(self, conn: asyncssh.SSHClientConnection, command: str) -> Dict:
        """Ejecuta un comando individual con gestión de timeouts"""
        try:
            result = await asyncio.wait_for(
                conn.run(command), # [commands]
                timeout=self.command_timeout # 10 secs
            )
            return {
                "command": command,
                "output": result.stdout, 
                "error": result.stderr,
                "status": "success" if result.exit_status == 0 else "error",
                "exit_code": result.exit_status
            }
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout ejecutando comando: {command}") # Error de timeout
            return {
                "command": command,
                "output": "",
                "error": "Timeout excedido",
                "status": "timeout",
                "exit_code": -1 
            }
        except Exception as e:
            self.logger.error(f"Error ejecutando comando {command}: {str(e)}")
            return {
                "command": command,
                "output": "",
                "error": str(e),
                "status": "error",
                "exit_code": -1
            }

    async def run(self) -> List[Dict]:
        """Ejecuta comandos de forma segura y paralela"""
        results = []
        
        try:
            async with self._get_connection() as conn:
                semaphore = asyncio.Semaphore(self.max_parallel) # maximo 5 comandos en paralelo
                
                async def limited_execution(cmd):
                    async with semaphore:
                        return await self._execute_command(conn, cmd)
                
                tasks = [limited_execution(cmd) for cmd in self.commands]
                results = await asyncio.gather(*tasks)
                
        except Exception as e:
            self.logger.critical(f"Error crítico: {str(e)}")
            results = [{
                "command": cmd,
                "output": "",
                "error": f"Error de conexión: {str(e)}",
                "status": "connection_error",
                "exit_code": -1
            } for cmd in self.commands]
            
        self._log_summary(results)
        return results

    def _log_summary(self, results: List[Dict]):
        stats = {
            "total": len(results),
            "success": sum(1 for r in results if r['status'] == "success"),
            "errors": sum(1 for r in results if r['status'] == "error"),
            "timeouts": sum(1 for r in results if r['status'] == "timeout")
        }
        self.logger.info(
            f"Resumen ejecución SSH - "
            f"Comandos: {stats['total']}, "
            f"Éxitos: {stats['success']}, "
            f"Errores: {stats['errors']}, "
            f"Timeouts: {stats['timeouts']}"
        )