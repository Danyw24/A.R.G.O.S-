from pydantic import Field, BaseModel
from agency_swarm import BaseTool
import pandas as pd
import os

class ReadExcelFile(BaseTool):
    """
    Lee datos de un archivo Excel y devuelve el contenido.
    """
    file: str = Field(..., description="Ruta del archivo Excel que se va a leer")

    def run(self):
        # Verificar si el archivo existe
        if not os.path.exists(self.file):
            return f'Error: El archivo {self.file} no existe.'

        # Leer el archivo Excel
        try:
            df = pd.read_excel(self.file)
            # Convertir el DataFrame a un diccionario
            data = df.to_dict(orient='records')
            return data
        except Exception as e:
            return f'Error al leer el archivo: {str(e)}'
