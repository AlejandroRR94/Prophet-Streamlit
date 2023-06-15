import streamlit as st
import pandas as pd
from prophet import Prophet
import os

# Configuración inicial de Streamlit
st.set_page_config(page_title="Predicciones de Prophet", layout="wide")

# Título de la aplicación
st.title("Predicciones de Prophet")
st.markdown("Aplicación de Streamlit para producir predicciones de Prophet sobre archivos que contienen series temporales")

# DATACELL/prophet_app_st/data/raw/yahoo_data.xlsx


# Carga de datos de ejemplo
@st.cache_data
def load_data():
    real_path = os.path.relpath()
    df = pd.read_excel("../../data/raw/yahoo_data.xlsx")  # Reemplazar por la ruta correcta del archivo
    df.dropna(axis=0, inplace=True)


    #Preparamos los datos en el formato reqerido por Prophet
    # data = df[[time_variable, selected_y_var]]
    # data = data.rename(columns={time_variable: 'ds', selected_y_var: 'y'})
    return df


data_file = st.file_uploader("Selecciona tu archivo excel para predecir la serie temporal \n(por defecto cargamos una serie temporal de yahoo)")
if data_file is not None:
    data = pd.read_excel(data_file)
else:
    data = load_data()

# MOSTRAMOS LOS DATOS
st.write(data.head())

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


# Selección de variables
selected_y_var, time_variable = select_variables(data)
data = data[[time_variable, selected_y_var]]
data = data.rename(columns={time_variable: 'ds', selected_y_var: 'y'})

# Filtrado del conjunto de datos
# data_selected = data[selected_variables]
# Configuración de los parámetros ajustables
changepoint_prior_scale = st.sidebar.slider("changepoint_prior_scale", 0.0, 1.0, 0.05, 0.01)
seasonality_prior_scale = st.sidebar.slider("seasonality_prior_scale", 0.0, 10.0, 1.0, 0.1)

# Creación del modelo Prophet
model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale)

# Ajuste del modelo a los datos
model.fit(data)

# Generación de predicciones
future = model.make_future_dataframe(periods=365)  # Predicciones para los próximos 365 días
forecast = model.predict(future)

# Gráfico interactivo de los datos y las predicciones
st.write("### Datos y Predicciones")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Gráfico interactivo de las componentes
st.write("### Componentes")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)
