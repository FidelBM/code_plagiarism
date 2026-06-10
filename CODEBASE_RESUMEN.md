# Resumen de la codebase

## Propósito

Este proyecto construye y evalúa modelos de machine learning para detectar similitud o reutilización de código fuente. La codebase trabaja con programas en C y Java del dataset SOCO14, genera pares positivos y negativos de código, aplica preprocesamiento textual y léxico, entrena modelos baseline y modelos basados en embeddings de CodeBERT, y guarda métricas, predicciones y artefactos entrenados.

## Estructura general

- `01.ipynb`: construye el dataset raw de pares de código.
- `02.ipynb`: limpia, normaliza y genera features para modelado.
- `03.ipynb`: entrena modelos baseline con TF-IDF y features simples.
- `04.ipynb`: genera embeddings con CodeBERT y entrena clasificadores clásicos sobre esos embeddings.
- `05.ipynb`: entrena una red neuronal MLP custom sobre features derivadas de embeddings.
- `data/train`: código fuente de entrenamiento en C y Java, junto con archivos `.qrel` de pares positivos.
- `data/test`: código fuente de prueba separado por lenguaje y escenario.
- `data/processed`: CSVs intermedios y finales usados por los notebooks.
- `data/embeddings`: embeddings NumPy generados a partir de CodeBERT.
- `data/models`: modelos entrenados y metadata serializada.
- `data/reports`: métricas, comparaciones y predicciones de validación.

## Datos

El entrenamiento contiene 79 archivos C y 259 archivos Java. El conjunto de prueba contiene 18,499 archivos C y 11,977 archivos Java, organizados por escenarios.

Los archivos `.qrel` (`SOCO14-c.qrel`, `SOCO14-java.qrel` y sus versiones de test) contienen referencias a pares relacionados. En `01.ipynb` se cargan estos pares positivos, se validan contra los archivos disponibles y se enriquecen con el código fuente completo.

## Flujo de notebooks

### `01.ipynb` - Construcción raw

Este notebook carga archivos C y Java, detecta lenguaje y escenario en test, lee los `.qrel`, arma pares positivos y genera pares negativos balanceados. Luego divide el dataset en entrenamiento y validación con `train_test_split`.

Artefactos principales:

- `data/processed/00_train_codes_raw.csv`
- `data/processed/00_test_codes_raw.csv`
- `data/processed/01_train_positive_pairs_raw.csv`
- `data/processed/01_train_pairs_raw.csv`
- `data/processed/01_train_model_raw.csv`
- `data/processed/01_val_model_raw.csv`

### `02.ipynb` - Preprocesamiento

Este notebook agrega features de longitud, elimina comentarios de C/Java, tokeniza el código, normaliza tokens y calcula medidas de similitud como Jaccard y overlap de tokens. Produce vistas separadas para baseline y transformer.

Artefactos principales:

- `data/processed/02_train_preprocessed.csv`
- `data/processed/02_val_preprocessed.csv`
- `data/processed/02_train_baseline.csv`
- `data/processed/02_val_baseline.csv`
- `data/processed/02_train_transformer.csv`
- `data/processed/02_val_transformer.csv`

### `03.ipynb` - Baseline clásico

Entrena modelos clásicos con representación TF-IDF del código y/o features manuales. El mejor modelo guardado fue `Linear SVM`, con F1 de validación aproximado de `0.9362`.

Artefactos principales:

- `data/models/03_baseline_model.joblib`
- `data/reports/03_baseline_model_results.csv`
- `data/reports/03_val_predictions_baseline.csv`

### `04.ipynb` - CodeBERT + clasificadores clásicos

Usa `microsoft/codebert-base` para convertir cada fragmento de código en embeddings. Luego construye features por par usando combinaciones de embeddings, por ejemplo similitud coseno, diferencias absolutas y productos elemento a elemento. Sobre esas features entrena clasificadores clásicos. El mejor modelo reportado fue `Embedding Random Forest`, con F1 de validación aproximado de `0.9091`.

Artefactos principales:

- `data/embeddings/04_train_emb1_microsoft_codebert_base_len256.npy`
- `data/embeddings/04_train_emb2_microsoft_codebert_base_len256.npy`
- `data/embeddings/04_val_emb1_microsoft_codebert_base_len256.npy`
- `data/embeddings/04_val_emb2_microsoft_codebert_base_len256.npy`
- `data/models/04_transformer_embedding_model.joblib`
- `data/reports/04_transformer_embedding_model_results.csv`
- `data/reports/04_val_predictions_transformer_embeddings.csv`
- `data/reports/04_comparison_baseline_vs_transformer.csv`

### `05.ipynb` - MLP custom con embeddings

Reutiliza `microsoft/codebert-base` para generar embeddings y entrena una red neuronal MLP custom con Keras/TensorFlow. El notebook evalúa distintos thresholds de decisión y selecciona el mejor por F1. El mejor threshold reportado fue `0.40`, con F1 de validación aproximado de `0.8696`.

Artefactos principales:

- `data/models/04B_custom_mlp_best.keras`
- `data/models/05_custom_mlp_final.keras`
- `data/models/05_custom_mlp_metadata.joblib`
- `data/reports/04B_custom_mlp_training_history.csv`
- `data/reports/04B_custom_mlp_threshold_results.csv`
- `data/reports/04B_custom_mlp_comparison.csv`
- `data/reports/05_custom_mlp_val_predictions.csv`

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
