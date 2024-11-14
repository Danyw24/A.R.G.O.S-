from agency_swarm import BaseTool
import pandas as pd
from pydantic import Field

class ExcelReader(BaseTool):
    """
    Herramienta que permite cargar y procesar datos de un archivo Excel.
    """
    file_path: str = Field(..., description="Ruta del archivo Excel para cargar los datos")
    
    def run(self):
        # Lee el archivo Excel usando Pandas
        try:
            df = pd.read_excel(self.file_path)
            return df  # Devolver el DataFrame o procesarlo según el caso
        except Exception as e:
            return f"Error al leer el archivo: {e}"
