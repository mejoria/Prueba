# -*- coding: utf-8 -*-
"""
Script principal para orquestar el flujo de trabajo de predicción de ventas:
carga de datos, preprocesamiento, entrenamiento y evaluación de modelos.
"""

import pandas as pd
from google.cloud import bigquery
# Importar los módulos de procesamiento de datos y entrenamiento de modelos
from . import data_processing
from . import model_training

def main():
    """
    Función principal que orquesta el flujo de trabajo.
    """
    print("Iniciando flujo de trabajo...")

    # 1. Carga de datos (desde BigQuery)
    bq_project = 'mejoria-prueba'
    bq_dataset = 'datos_prueba'
    bq_table = 'sales_data_sample'
    df = data_processing.load_data_from_bigquery(bq_project, bq_dataset, bq_table)
    if df is None:
        print("Flujo de trabajo interrumpido debido a error en la carga de datos.")
        return

    # 2. Preprocesamiento de datos
    X_train_prep, X_test_prep, y_train, y_test, preprocessor = data_processing.preprocess_data(df.copy()) # .copy() para evitar modificar el original
    if X_train_prep is None: # Verifica si preprocess_data retornó None debido a error
        print("Flujo de trabajo interrumpido debido a error en el preprocesamiento de datos.")
        return

    # 3. Selección y evaluación de modelos
    if X_train_prep is not None and X_test_prep is not None: # Verifica nuevamente antes de modelar
        mejor_modelo, resultados_evaluacion = model_training.model_selection_and_evaluation(
            X_train_prep, X_test_prep, y_train, y_test, preprocessor # Pasar el preprocesador
        )

        # 4. Imprimir resultados
        print("\n--- Resultados Finales ---")
        print(f"Mejor Modelo: {mejor_modelo.__class__.__name__}")
        print("\nMétricas de Evaluación Detalladas:")
        for nombre_modelo, metricas in resultados_evaluacion.items():
            print(f"  {nombre_modelo}:")
            for metrica, valor in metricas.items():
                print(f"    {metrica}: {valor:.4f}")

    print("\nFlujo de trabajo completado.")

if __name__ == "__main__":
    main()