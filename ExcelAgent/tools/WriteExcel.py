from pydantic import Field, field_validator

from agency_swarm import BaseTool
import pandas as pd
import os

class WriteExcel(BaseTool):
    """
    Creates a excel file and write data in it
    """
    file: str = Field(..., description="Ruta del archivo Excel donde se cargarán los datos")
    data: list = Field(..., description="Lista de diccionarios con los datos a cargar en el Excel")

    def run (self):
        #Verificar si el archivo existe
        if os.path.exists(self.file):
            df = pd.read_excel(self.file)
            new_data_df = pd.DataFrame(self.data)
            df = new_data_df
        #Si no existe, crear un nuevo archivo
        else:
            df = pd.DataFrame(self.data)
        

        df.to_excel(self.file, index=False, engine='openpyxl')
        return f'Los datos se han guardado correctamente en {self.file}'