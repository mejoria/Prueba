# -*- coding: utf-8 -*-
"""
Módulo API para servir predicciones de ventas usando FastAPI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib  # Para cargar el modelo entrenado

# Importar funciones necesarias (ajustar según sea necesario)
# from .data_processing import preprocess_input  # Necesitaremos una función para preprocesar la entrada de la API
# from .model_training import load_model # Necesitaremos una función para cargar el modelo

app = FastAPI(title="API de Predicción de Ventas", version="0.1.0")

# --- Modelo de datos para la entrada de la API ---
class PredictionInput(BaseModel):
    QUANTITYORDERED: int
    PRICEEACH: float
    ORDERLINENUMBER: int
    ORDERDATE: str  # Recibimos la fecha como string, el preprocesamiento la manejará
    PRODUCTLINE: str

# --- Carga del Modelo y Preprocesador ---
MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
model = None
preprocessor = None

try:
    model = joblib.load(MODEL_PATH)
    print(f"Modelo cargado exitosamente desde {MODEL_PATH}")
except FileNotFoundError:
    print(f"ADVERTENCIA: Modelo no encontrado en {MODEL_PATH}. Endpoint /predict/ no funcionará correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo desde {MODEL_PATH}: {e}")

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"Preprocesador cargado exitosamente desde {PREPROCESSOR_PATH}")
except FileNotFoundError:
    print(f"ADVERTENCIA: Preprocesador no encontrado en {PREPROCESSOR_PATH}. Endpoint /predict/ no funcionará correctamente.")
except Exception as e:
    print(f"Error al cargar el preprocesador desde {PREPROCESSOR_PATH}: {e}")


# --- Función de Preprocesamiento para la API ---
def preprocess_input(data: pd.DataFrame):
    """
    Preprocesa los datos de entrada usando el preprocesador cargado.
    """
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocesador no disponible.")
    try:
        # Asegúrate de que las columnas coincidan con las esperadas por el preprocesador
        # Puede ser necesario reordenar o añadir columnas faltantes con valores por defecto
        # antes de transformar.
        print(f"Preprocesando datos de entrada: {data.to_dict()}")
        data_processed = preprocessor.transform(data)
        print("Datos de entrada preprocesados exitosamente.")
        return data_processed
    except Exception as e:
        print(f"Error durante el preprocesamiento de la entrada: {e}")
        # Sé más específico sobre el error si es posible (ej., columna faltante)
        raise HTTPException(status_code=400, detail=f"Error al preprocesar datos de entrada: {e}")


# --- Endpoint de Predicción ---
@app.post("/predict/")
async def predict_sales(input_data: PredictionInput):
    """
    Recibe datos de un producto y retorna la predicción de ventas.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Modelo o preprocesador no disponible.")

    try:
        # 1. Convertir entrada a DataFrame (o formato esperado por el preprocesador/modelo)
        #    Esto necesita adaptarse a cómo tu modelo espera los datos.
        # Convertir Pydantic model a dict, luego a DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        print(f"Datos de entrada recibidos: {input_data.to_dict()}")

        # 2. Preprocesar los datos de entrada
        #    Necesitarás una función `preprocess_input` similar a la de `data_processing`
        #    pero adaptada para una sola entrada y usando el preprocesador ajustado (fit).
        #    Por ahora, usaremos los datos directamente como placeholder.
        # Asegurarse de que la ingeniería de características se aplique si es necesario
        # antes de pasar al preprocesador. Esto depende de cómo esté estructurado
        # tu data_processing.feature_engineering y tu preprocessor.
        # Si feature_engineering (ej. extraer mes/año de ORDERDATE) se hace *antes*
        # del ColumnTransformer guardado, necesitamos hacerlo aquí también.
        # Por simplicidad ahora, asumimos que el preprocesador maneja todo.
        preprocessed_data = preprocess_input(input_df) # Usar la función de preprocesamiento

        # 3. Realizar la predicción
        #    Asegúrate de que el formato de `preprocessed_data` sea el correcto para model.predict()
        #    Muchos modelos de scikit-learn esperan un array numpy o similar.
        #    El formato exacto dependerá de tu pipeline de preprocesamiento.
        #    Para el DummyModel, esperamos una lista de diccionarios.
        # El preprocesador devuelve un array numpy (generalmente), que es lo que esperan los modelos sklearn
        prediction = model.predict(preprocessed_data)

        # Asumiendo que la predicción devuelve una lista/array con un solo valor
        predicted_sales = prediction[0]

        print(f"Predicción realizada: {predicted_sales}")
        # Devolver las entradas originales junto con la predicción
        return {"input_data": input_data.dict(), "predicted_sales": predicted_sales}

    except Exception as e:
        print(f"Error durante la predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno al procesar la predicción: {e}")

# --- Endpoint Raíz (Opcional) ---
@app.get("/")
async def read_root():
    return {"message": "Bienvenido a la API de Predicción de Ventas"}

# --- Ejecución (para desarrollo local) ---
if __name__ == "__main__":
    import uvicorn
    # Nota: Ejecutar con `uvicorn src.api:app --reload` desde el directorio raíz del proyecto
    # es generalmente preferible para desarrollo.
    print("Iniciando servidor Uvicorn para desarrollo...")
    uvicorn.run(app, host="127.0.0.1", port=8000)