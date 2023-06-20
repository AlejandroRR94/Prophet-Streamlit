import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Carga de datos de ejemplo
@st.cache_data
def load_data():
    """
    Carga por defecto los valores de data/raw/yahoo_data.xlsx
    """
    print('Directory Name:     ', os.path.dirname(__file__))
    directory = Path(os.path.dirname(__file__)).parent.absolute().parent.absolute()
    data_directory = os.path.join(directory, "data", "raw")
    trial_data = st.selectbox("Archivos de prueba", [file for file in os.listdir(data_directory)])
    if trial_data is not None:
        df = pd.read_excel(os.path.join(directory, trial_data))  # Reemplazar por la ruta correcta del archivo
    else:
        df = pd.read_excel(os.path.join(directory, "data/raw/yahoo_data.xlsx"))  # Reemplazar por la ruta correcta del archivo
    df.dropna(axis=0, inplace=True)

    return df

def select_variables(data:pd.DataFrame):
    """
    Seleccionamos dos variables mediante desplegables
    """
    selected_y_var = st.selectbox('**VARIABLE A PREDECIR?**',
                                    list(data.columns)
                                    )


    time_variable = st.selectbox('**VARIABLE TEMPORAL?**',
                                    list(data.drop([selected_y_var], 
                                                    axis = 1).columns)
                                                    )
    return selected_y_var, time_variable