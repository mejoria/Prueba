# -*- coding: utf-8 -*-
"""
Módulo para el entrenamiento y evaluación de modelos de machine learning
para la predicción de ventas.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
# Importar modelos a medida que se implementen
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR

# --- Funciones de Entrenamiento ---

from sklearn.linear_model import LinearRegression

def train_linear_regression(X_train, y_train):
    """
    Entrena un modelo de Regresión Lineal.

    Args:
        X_train (pd.DataFrame): Datos de entrenamiento (features).
        y_train (pd.Series): Datos de entrenamiento (target).

    Returns:
        LinearRegression: Modelo de Regresión Lineal entrenado.
    """
    print("Entrenando modelo de Regresión Lineal...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Modelo de Regresión Lineal entrenado.")
    return model

from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train):
    """
    Entrena un modelo de Random Forest Regressor.

    Args:
        X_train (pd.DataFrame): Datos de entrenamiento (features).
        y_train (pd.Series): Datos de entrenamiento (target).

    Returns:
        RandomForestRegressor: Modelo de Random Forest Regressor entrenado.
    """
    print("Entrenando modelo de Random Forest...")
    model = RandomForestRegressor(random_state=42)  # Añadido random_state para reproducibilidad
    model.fit(X_train, y_train)
    print("Modelo de Random Forest entrenado.")
    return model

from sklearn.svm import SVR

def train_svr(X_train, y_train):
    """
    Entrena un modelo de Support Vector Regressor.

    Args:
        X_train (pd.DataFrame): Datos de entrenamiento (features).
        y_train (pd.Series): Datos de entrenamiento (target).

    Returns:
        SVR: Modelo de Support Vector Regressor entrenado.
    """
    print("Entrenando modelo SVR...")
    model = SVR()
    model.fit(X_train, y_train)
    print("Modelo SVR entrenado.")
    return model

# --- Funciones de Evaluación ---

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el rendimiento del modelo utilizando MSE y R2.

    Args:
        model: Modelo entrenado.
        X_test (pd.DataFrame): Datos de prueba (features).
        y_test (pd.Series): Datos de prueba (target).

    Returns:
        dict: Diccionario con las métricas de evaluación (MSE y R2).
    """
    print("Evaluando modelo...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  MSE: {mse:.4f}")
    print(f"  R^2: {r2:.4f}")
    return {"MSE": mse, "R2": r2}

# --- Funciones para Guardar/Cargar ---

def save_model_and_preprocessor(model, preprocessor, model_path="models/best_model.pkl", preprocessor_path="models/preprocessor.pkl"):
    """
    Guarda el modelo entrenado y el preprocesador en archivos .pkl.

    Args:
        model: El modelo de machine learning entrenado.
        preprocessor: El objeto preprocesador ajustado (e.g., ColumnTransformer).
        model_path (str): Ruta donde guardar el modelo.
        preprocessor_path (str): Ruta donde guardar el preprocesador.
    """
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

        # Guardar modelo
        joblib.dump(model, model_path)
        print(f"Modelo guardado exitosamente en {model_path}")

        # Guardar preprocesador
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Preprocesador guardado exitosamente en {preprocessor_path}")

    except Exception as e:
        print(f"Error al guardar el modelo o preprocesador: {e}")

# --- Función Principal para el Modelado ---

def model_selection_and_evaluation(X_train, X_test, y_train, y_test, preprocessor): # Añadido preprocessor
    """
    Entrena, evalúa diferentes modelos, selecciona el mejor basado en R2 y lo guarda.

    Args:
        X_train (pd.DataFrame): Datos de entrenamiento (features preprocesados).
        X_test (pd.DataFrame): Datos de prueba (features preprocesados).
        y_train (pd.Series): Datos de entrenamiento (target).
        y_test (pd.Series): Datos de prueba (target).
        preprocessor: El objeto preprocesador ajustado para guardarlo.

    Returns:
        tuple: (mejor_modelo, resultados_evaluacion), donde:
               - mejor_modelo: El modelo con el R2 más alto.
               - resultados_evaluacion: Diccionario con métricas de todos los modelos.
    """
    print("Iniciando selección y evaluación de modelos...")
    modelos = {}
    resultados_evaluacion = {}

    # 1. Entrenar modelos
    modelos['Regresión Lineal'] = train_linear_regression(X_train, y_train)
    modelos['Random Forest'] = train_random_forest(X_train, y_train)
    modelos['SVR'] = train_svr(X_train, y_train)

    # 2. Evaluar modelos
    for nombre, modelo in modelos.items():
        print(f"\n--- Evaluación de {nombre} ---")
        resultados_evaluacion[nombre] = evaluate_model(modelo, X_test, y_test)

    # 3. Seleccionar el mejor modelo basado en R2
    mejor_modelo_nombre = max(resultados_evaluacion, key=lambda k: resultados_evaluacion[k]['R2'])
    mejor_modelo = modelos[mejor_modelo_nombre]

    print(f"\n--- Mejor Modelo Seleccionado: {mejor_modelo_nombre} ---")

    # 4. Guardar el mejor modelo y el preprocesador
    save_model_and_preprocessor(mejor_modelo, preprocessor) # Llamada a la nueva función

    print("Selección y evaluación de modelos completada.")
    return mejor_modelo, resultados_evaluacion

if __name__ == '__main__':
    # Ejemplo de uso o pruebas rápidas (opcional)
    print("Módulo model_training cargado.")