import streamlit as st
import pandas as pd
from prophet import Prophet

from prophet.serialize import model_to_json, model_from_json

from utils import *
import os
from pathlib import Path

# Configuración inicial de Streamlit
st.set_page_config(page_title="Predicciones de Prophet", layout="wide")

# Título de la aplicación
st.title("Predicciones de Prophet")
st.markdown("Aplicación de Streamlit para producir predicciones de Prophet sobre archivos que contienen series temporales")

data_file = st.file_uploader("Selecciona tu archivo excel para predecir la serie temporal \n(por defecto cargamos una serie temporal de yahoo)")

print(f"Directorio del modelo: {os.path.dirname(__file__)}")
model_directory = os.path.dirname(__file__)
weights_directory = os.path.join(model_directory, "weights")
directory = Path(os.path.dirname(__file__)).parent.absolute().parent.absolute()
data_directory = os.path.join(directory, "data", "raw")
trial_data = st.selectbox("Archivos de prueba", [file for file in os.listdir(data_directory) if "git" not in file])

if data_file is not None:
    try:
        # if data_file is not None:
        data = pd.read_excel(data_file)
    except Exception as e:
        data = pd.read_csv(data_file)
else:
    data = load_data(trial_data)

# MOSTRAMOS LOS DATOS
st.write("Todos los datos")
if "Country" in list(data.columns):
    st.write(data.groupby("Country").head())
else:
    st.write(data.head())
st.write(data.shape)



# Configuración de los parámetros ajustables
st.sidebar.write("PARÁMETROS")
st.sidebar.markdown("Modificar los siguientes parámetros para alterar el ajuste y la predicción del modelo")

changepoint_prior_scale = st.sidebar.slider("changepoint_prior_scale", 0.0, 1.0, 0.05, 0.01)
seasonality_prior_scale = st.sidebar.slider("seasonality_prior_scale", 0.0, 10.0, 1.0, 0.1)


if "Country" in list(data.columns):
    country = st.sidebar.selectbox("Paises", data["Country"].unique())
    data_country = data[data.Country == country]
    # MOSTRAMOS LOS DATOS
    st.write(f"Datos de {country}")
    st.write(data_country.head())
    st.write(data_country.shape)
    data = data_country.copy()


# Selección de variables
st.write("DISCLAIMER: La variable temporal puede situarse como la target por defecto, al seleccionar las variables tal y como lo haríamos, el error se subsana")
selected_y_var, time_variable = select_variables(data)
data = data[[time_variable, selected_y_var]]
data = data.rename(columns={time_variable: 'ds', selected_y_var: 'y'})


days_to_forecast = st.sidebar.number_input(label='Días a predecir', min_value = 1, max_value=365, value = 30)



# Creación del modelo Prophet
model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale)


retrain = st.selectbox("¿Quieres reentrenar el modelo que cargues?", [True, False, "Nuevo Modelo"])
# Guardamos los pesos del modelo
weights_name= os.path.join(weights_directory, f"serialized_model_{selected_y_var}_{trial_data.split('.')[0]}.json")

if len(os.listdir(weights_directory))>0:
    model_name = st.selectbox("Selecciona unos pesos para el modelo", [d.split("/")[-1] for d in os.listdir(weights_directory) if "git" not in d])

# Condicionales para cargar o escribir los pesos del modelo
if os.path.isfile(weights_name)==True and retrain==False: 
    with open(weights_name, "r") as fin:
        model = model_from_json(fin.read())

elif os.path.isfile(weights_name)==True and retrain==True: 
    model.fit(data)
    with open(weights_name, "w") as fout:
        fout.write(model_to_json(model))

elif retrain=="Nuevo Modelo":
    # Ajuste del modelo a los datos
    model.fit(data)
    with open(os.path.join(weights_directory, weights_name), "w") as fout:
        fout.write(model_to_json(model))


# Generación de predicciones

# days_to_forecast = st.sidebar.slider("days_to_forecast", 1, 365, 1, 1)

future = model.make_future_dataframe(periods=days_to_forecast)  # Predicciones para los próximos 365 días
forecast = model.predict(future)

# Gráfico interactivo de los datos y las predicciones
st.write("### Datos y Predicciones")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Gráfico interactivo de las componentes
st.write("### Componentes")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)
