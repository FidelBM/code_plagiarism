# Resumen de la codebase

## Propósito

Este proyecto construye y evalúa modelos de machine learning para detectar similitud o reutilización de código fuente. La codebase trabaja con programas en C y Java del dataset SOCO14, genera pares positivos y negativos de código, aplica preprocesamiento textual y léxico, entrena modelos baseline y modelos basados en embeddings de CodeBERT, y guarda métricas, predicciones y artefactos entrenados.

## Datos

El entrenamiento contiene 79 archivos C y 259 archivos Java. El conjunto de prueba contiene 18,499 archivos C y 11,977 archivos Java, organizados por escenarios.

Los archivos `.qrel` (`SOCO14-c.qrel`, `SOCO14-java.qrel` y sus versiones de test) contienen referencias a pares relacionados. En `01.ipynb` se cargan estos pares positivos, se validan contra los archivos disponibles y se enriquecen con el código fuente completo.

## Modelos de machine learning utilizados

1. `TfidfVectorizer`
   - Usado para transformar texto de código en vectores TF-IDF en el baseline.
   - No es un clasificador, pero es una pieza central de representación de features.

2. `LogisticRegression`
   - Usado en `03.ipynb` como baseline clásico.
   - Usado también en `04.ipynb` como clasificador sobre embeddings de CodeBERT.

3. `LinearSVC`
   - Usado en `03.ipynb` como baseline clásico.
   - Usado también en `04.ipynb` como clasificador sobre embeddings de CodeBERT.
   - Fue el mejor baseline clásico según `03_baseline_model_results.csv`.

4. `RandomForestClassifier`
   - Usado en `03.ipynb` con features simples.
   - Usado en `04.ipynb` sobre features derivadas de embeddings.
   - Fue el mejor modelo de la etapa transformer clásica según `04_transformer_embedding_model_results.csv`.

5. `microsoft/codebert-base`
   - Modelo transformer usado como extractor de embeddings de código.
   - Se carga con `AutoTokenizer` y `AutoModel` de Hugging Face Transformers.
   - Alimenta tanto los clasificadores clásicos de `04.ipynb` como la MLP de `05.ipynb`.

6. `custom_mlp_code_reuse_detector`
   - Red neuronal MLP construida con `tf.keras.models.Sequential`.
   - Usa capas `Dense`, activación sigmoide de salida y optimizador `Adam`.
   - Entrenada con callbacks como `EarlyStopping` y guardado del mejor checkpoint.

## Resultados principales de validación

| Etapa | Mejor modelo | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | Linear SVM | 0.9318 | 0.8800 | 1.0000 | 0.9362 |
| CodeBERT + clásico | Embedding Random Forest | 0.9091 | 0.9091 | 0.9091 | 0.9091 |
| MLP custom | Custom MLP threshold 0.40 | 0.8636 | 0.8333 | 0.9091 | 0.8696 |

## Observaciones

- El pipeline está organizado de forma incremental: raw data, preprocesamiento, baseline, embeddings transformer y red neuronal.
- La mejor métrica F1 reportada en validación corresponde al baseline `Linear SVM`.
- CodeBERT se usa como extractor de representaciones, no como modelo fine-tuned extremo a extremo.
- La MLP final optimiza el threshold de decisión, pero en los resultados actuales queda por debajo del baseline clásico y del Random Forest sobre embeddings.
- Los notebooks guardan artefactos suficientes para reproducir comparaciones, revisar predicciones y reutilizar modelos entrenados.

## Fuentes
- https://aclanthology.org/2023.ranlp-1.34.pdf
