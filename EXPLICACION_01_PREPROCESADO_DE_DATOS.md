# Explicacion del notebook `01-Preprocesado_de_datos.ipynb`

Este documento explica a profundidad que hace el notebook `01-Preprocesado_de_datos.ipynb`, con enfasis en el recorrido del dataset desde su forma raw hasta los archivos tabulares que se usan despues para entrenar y validar modelos de deteccion de similitud o reutilizacion de codigo.

La idea central del notebook es convertir una coleccion de archivos fuente y archivos de referencia `qrel` en un dataset supervisado de pares de programas. Cada fila final representa dos codigos comparados entre si y una etiqueta:

- `label = 1`: el par aparece como relacion positiva en los archivos `qrel`; se interpreta como codigo similar, reutilizado o relacionado.
- `label = 0`: el par se genera artificialmente como ejemplo negativo; se interpreta como codigo no marcado como similar.

En esta etapa el codigo se conserva en formato raw. Es decir, todavia no se limpia, no se normaliza, no se tokeniza, no se eliminan comentarios y no se extraen embeddings ni caracteristicas. El objetivo de este primer notebook es dejar armado el dataset de pares con los textos originales del codigo.

## 1. Entrada raw del proyecto

El notebook parte de la carpeta `data`, que contiene principalmente:

- `data/train`: corpus de entrenamiento.
- `data/test`: corpus de prueba.
- `data/processed`: carpeta donde se guardan las salidas generadas por el notebook.

En el entrenamiento, la estructura esperada es mas simple:

```text
data/train/
  c/
    000.c
    001.c
    ...
  java/
    000.java
    001.java
    ...
  SOCO14-c.qrel
  SOCO14-java.qrel
```

Los archivos `.c` y `.java` son los programas originales. Los archivos `.qrel` contienen pares de programas que el dataset considera relacionados. Por ejemplo:

```text
005.c 006.c
007.c 032.c
003.java 004.java
005.java 006.java
```

En el conjunto de prueba, la estructura es mas profunda porque los programas estan separados por lenguaje y por escenario:

```text
data/test/
  c/
    A1/
    A2/
    B1/
    B2/
    C1/
    C2/
  java/
    A1/
    A2/
    B1/
    B2/
    C1/
    C2/
  soco14-test-c-update.qrel
  soco14-test-java-update.qrel
  README
```

El notebook lee tambien los archivos de prueba en formato raw, pero el dataset final que deja listo para modelado en este notebook se construye con los pares de entrenamiento y se divide en `train` y `validation`.

## 2. Configuracion de rutas

El notebook define rutas base con `Path`:

```python
DATASET_PATH = Path("data")
TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"
PROCESSED_PATH = DATASET_PATH / "processed"
```

Esto permite referirse a las carpetas sin escribir rutas completas en cada paso. Tambien se crea `data/processed` si no existe, porque ahi se guardan los CSV intermedios y finales.

## 3. Lectura del codigo raw

La funcion mas basica de la etapa es `read_raw_code(file_path)`. Su trabajo es abrir un archivo de codigo y devolver su contenido completo como texto.

La lectura se hace con `encoding="utf-8"` y `errors="ignore"`. Esto es importante porque los corpus de codigo pueden traer caracteres raros, comentarios con acentos, simbolos copiados desde otros editores o archivos con codificacion mezclada. Con `errors="ignore"`, el notebook evita que un caracter problematico detenga todo el proceso.

En este punto no se transforma el codigo. Si un archivo contiene imports, espacios, saltos de linea, comentarios, nombres de variables originales o formato irregular, todo eso queda guardado tal como viene.

## 4. Construccion de `00_train_codes_raw.csv`

La funcion `load_train_codes_raw(train_path)` recorre las carpetas de entrenamiento para C y Java. Usa una configuracion interna:

```python
language_configs = {
    "c": {"folder": "c", "extension": ".c"},
    "java": {"folder": "java", "extension": ".java"}
}
```

Para cada archivo encontrado, el notebook crea un registro con estas columnas:

- `split`: siempre vale `train`.
- `language`: `c` o `java`.
- `scenario`: en entrenamiento se fija como `train`.
- `file_name`: nombre del archivo, por ejemplo `005.c` o `003.java`.
- `file_id`: nombre sin extension, por ejemplo `005`.
- `file_path`: ruta relativa al archivo original.
- `code_raw`: contenido completo del archivo.

El resultado mostrado por el notebook es:

```text
train_codes_raw: 338 filas x 7 columnas
```

Distribucion por lenguaje:

```text
java    259
c        79
```

Esta tabla todavia no contiene pares. Es un inventario de programas individuales, cada uno con su codigo crudo. Se guarda como:

```text
data/processed/00_train_codes_raw.csv
```

Conceptualmente, este archivo es la primera forma tabular del corpus raw de entrenamiento.

## 5. Construccion de `00_test_codes_raw.csv`

El notebook tambien carga los archivos de `data/test` con `load_test_codes_raw(test_path)`.

A diferencia de entrenamiento, test se recorre recursivamente con `rglob("*")`, porque los archivos estan dentro de subcarpetas como `A1`, `A2`, `B1`, etc. La funcion descarta:

- archivos `.qrel`;
- el archivo `README`;
- entradas que no son archivos.

Para cada archivo valido se detecta:

- lenguaje, usando la ruta o la extension;
- escenario, usando carpetas o nombres que coincidan con patrones como `A1`, `B2`, `C1`.

Las columnas son equivalentes a las de entrenamiento:

- `split`
- `language`
- `scenario`
- `file_name`
- `file_id`
- `file_path`
- `code_raw`

El notebook reporta:

```text
test_codes_raw: 30476 filas x 7 columnas
```

Distribucion por lenguaje y escenario:

```text
c_cpp  A1    4683
c_cpp  A2    5195
c_cpp  B1    4303
c_cpp  B2    3873
c_cpp  C1     300
c_cpp  C2     145
java   A1    3238
java   A2    3093
java   B1    3268
java   B2    2166
java   C1     124
java   C2      88
```

La salida se guarda como:

```text
data/processed/00_test_codes_raw.csv
```

Este archivo deja disponible el corpus de prueba en formato tabular raw, aunque el armado supervisado principal de este notebook ocurre con entrenamiento.

## 6. Lectura de los archivos `qrel`

Los archivos `qrel` son esenciales porque contienen las relaciones positivas. En este proyecto se usan:

```text
data/train/SOCO14-c.qrel
data/train/SOCO14-java.qrel
```

Cada linea contiene, como minimo, dos identificadores de archivo que forman un par positivo. El notebook incluye una funcion llamada `load_qrel_flexible(qrel_path, language)` para leerlos.

La funcion es "flexible" porque contempla dos posibles formatos:

1. Lineas con nombres completos de archivo:

```text
005.c 006.c
003.java 004.java
```

2. Lineas con identificadores numericos:

```text
5 6
3 4
```

Si la linea ya trae `.c` o `.java`, toma esos nombres directamente. Si solo encuentra numeros, los rellena con ceros a la izquierda usando `zfill(3)` y agrega la extension correspondiente. Por ejemplo, para lenguaje C:

```text
5 -> 005.c
```

Para cada par valido, se crea una fila con:

- `language`
- `file_1`
- `file_2`
- `label`, siempre `1`
- `source_qrel`, ruta del qrel de origen
- `line_number`, linea del qrel de donde salio la relacion

Al unir los qrel de C y Java, el notebook obtiene:

```text
positive_pairs_raw: 110 filas
```

Distribucion:

```text
java    84
c       26
```

En este punto los pares positivos solo contienen nombres de archivo y metadatos. Todavia no tienen el texto del codigo.

## 7. Validacion de existencia de archivos

Antes de unir los pares positivos con el codigo raw, el notebook valida que los archivos mencionados en los qrel existan dentro de `train_codes_raw`.

Para eso construye:

```python
available_train_files = set(train_codes_raw["file_name"])
```

Luego revisa si `file_1` o `file_2` tienen nombres que no esten en ese conjunto. El notebook reporta:

```text
Faltantes en file_1: 0
Faltantes en file_2: 0
```

Esta validacion es importante porque evita construir filas incompletas. Si un qrel mencionara un archivo inexistente, no se podria asociar `code_1_raw` o `code_2_raw`, y el ejemplo quedaria inutil para entrenamiento.

## 8. Union de pares positivos con codigo raw

Despues de validar los nombres, el notebook crea diccionarios desde `train_codes_raw`:

```python
code_map = train_codes_raw.set_index("file_name")["code_raw"].to_dict()
path_map = train_codes_raw.set_index("file_name")["file_path"].to_dict()
language_map = train_codes_raw.set_index("file_name")["language"].to_dict()
```

Estos mapas permiten convertir un nombre de archivo en su codigo y su ruta.

Para cada par positivo:

- `file_1` se transforma en `code_1_raw`;
- `file_2` se transforma en `code_2_raw`;
- tambien se agregan `path_1`, `path_2`, `language_1` y `language_2`.

Luego se limpian pares incompletos con `dropna` sobre `code_1_raw` y `code_2_raw`, y se conserva solo aquello donde ambos archivos pertenecen al mismo lenguaje.

Finalmente se reorganizan las columnas para dejar:

```text
language, scenario, file_1, file_2, path_1, path_2, code_1_raw, code_2_raw, label
```

El archivo resultante se guarda como:

```text
data/processed/01_train_positive_pairs_raw.csv
```

Este CSV ya representa ejemplos supervisados positivos completos: cada fila tiene dos programas en texto raw y una etiqueta `1`.

## 9. Construccion del conjunto de positivos conocidos

Antes de generar negativos, el notebook crea `positive_set`. Este conjunto contiene todos los pares positivos ya conocidos.

Cada par se ordena internamente:

```python
pair_key = tuple(sorted([row["file_1"], row["file_2"]]))
```

Esto hace que `(A, B)` y `(B, A)` se consideren el mismo par. Es una decision necesaria porque la similitud entre dos codigos es simetrica: si `005.c` es similar a `006.c`, tambien lo seria `006.c` frente a `005.c`.

El notebook reporta:

```text
len(positive_set) = 110
```

Ese conjunto se usa como filtro para no generar accidentalmente un negativo que en realidad aparece como positivo.

## 10. Generacion de pares negativos

La funcion `generate_negative_pairs_raw(...)` genera ejemplos con `label = 0`.

El procedimiento es:

1. Agrupar los programas de entrenamiento por `language`.
2. Elegir aleatoriamente un lenguaje.
3. Tomar dos archivos aleatorios de ese mismo lenguaje.
4. Construir una llave ordenada del par.
5. Rechazar el par si ya esta en `positive_set`.
6. Rechazar el par si ya fue generado antes como negativo.
7. Si pasa los filtros, guardarlo con su codigo raw y `label = 0`.

El muestreo usa `seed=42`, lo cual permite reproducir el resultado.

El numero de negativos se iguala al numero de positivos:

```python
n_positives = len(positive_pairs_with_code)
n_negatives = n_positives
```

El notebook reporta:

```text
Positivos: (110, 9)
Negativos: (110, 9)
```

Esto produce un dataset balanceado por etiqueta. Esa decision es util en una primera etapa porque evita que el modelo aprenda con una clase dominante. Si hubiera muchos mas negativos que positivos, un modelo podria obtener buena exactitud prediciendo casi todo como `0`, pero no necesariamente aprenderia a detectar reutilizacion.

## 11. Construccion de `01_train_pairs_raw.csv`

Despues de tener positivos y negativos, el notebook los concatena:

```python
train_pairs_raw = pd.concat(
    [positive_pairs_with_code, negative_pairs_raw],
    ignore_index=True
)
```

Luego mezcla las filas con:

```python
train_pairs_raw.sample(frac=1, random_state=42)
```

Esto evita que el CSV quede primero con todos los positivos y despues con todos los negativos.

Tambien asigna un identificador unico por fila:

```python
train_pair_000000
train_pair_000001
...
```

Las columnas finales de `train_pairs_raw` son:

```text
pair_id
language
scenario
file_1
file_2
path_1
path_2
code_1_raw
code_2_raw
label
```

El balance final es:

```text
label 0    110
label 1    110
```

Por lenguaje y etiqueta:

```text
c     label 0     57
c     label 1     26
java  label 0     53
java  label 1     84
```

El archivo se guarda como:

```text
data/processed/01_train_pairs_raw.csv
```

Este archivo es el dataset raw completo de pares de entrenamiento antes de separar entrenamiento y validacion.

## 12. Division en entrenamiento y validacion

El dataset `01_train_pairs_raw.csv` se divide con `train_test_split`:

```python
train_model_raw, val_model_raw = train_test_split(
    train_pairs_raw,
    test_size=0.2,
    random_state=42,
    stratify=train_pairs_raw["label"]
)
```

Los puntos importantes son:

- `test_size=0.2`: el 20% se separa para validacion.
- `random_state=42`: la particion es reproducible.
- `stratify=train_pairs_raw["label"]`: se mantiene el balance de etiquetas en ambos subconjuntos.

El resultado es:

```text
Train model:      176 filas x 10 columnas
Validation model: 44 filas x 10 columnas
```

Balance en entrenamiento:

```text
label 0    88
label 1    88
```

Balance en validacion:

```text
label 0    22
label 1    22
```

Estos son los archivos finales de este notebook:

```text
data/processed/01_train_model_raw.csv
data/processed/01_val_model_raw.csv
```

Son los datasets que quedan listos para la siguiente etapa del proyecto. Siguen siendo raw porque contienen `code_1_raw` y `code_2_raw` sin normalizacion, pero ya estan estructurados como ejemplos supervisados de comparacion entre dos programas.

## 13. Validaciones finales

Al final, el notebook valida que los datasets finales tengan las columnas necesarias:

```text
pair_id
language
file_1
file_2
code_1_raw
code_2_raw
label
```

Tambien verifica que `code_1_raw` y `code_2_raw` no tengan valores nulos en train ni validation.

Si todo esta correcto, imprime:

```text
Dataset raw listo para pasar a preprocesado.
```

Esta frase resume el objetivo real del notebook: no producir todavia features finales, sino dejar listo el dataset raw supervisado para que otros notebooks puedan limpiarlo, tokenizarlo, vectorizarlo o convertirlo en embeddings.

## 14. Resumen del flujo completo

El flujo de transformacion puede leerse asi:

```text
Archivos fuente raw
  data/train/c/*.c
  data/train/java/*.java
        |
        v
Inventario tabular de programas individuales
  00_train_codes_raw.csv
        |
        +---------------------------+
        |                           |
        v                           v
Archivos qrel                 Codigo raw por archivo
  SOCO14-c.qrel               code_map, path_map
  SOCO14-java.qrel            language_map
        |                           |
        +-------------+-------------+
                      v
Pares positivos con codigo raw
  01_train_positive_pairs_raw.csv
                      |
                      v
Generacion de pares negativos del mismo lenguaje
  label = 0, evitando pares positivos conocidos
                      |
                      v
Union y mezcla de positivos y negativos
  01_train_pairs_raw.csv
                      |
                      v
Split estratificado 80/20
        +-------------+-------------+
        |                           |
        v                           v
01_train_model_raw.csv        01_val_model_raw.csv
```

En paralelo, el notebook tambien convierte el corpus de prueba individual a:

```text
00_test_codes_raw.csv
```

pero no usa ese archivo para crear `01_train_model_raw.csv` ni `01_val_model_raw.csv`.

## 15. Que significa "dataset utilizado" en este notebook

Dentro del flujo del proyecto, el dataset que queda listo para ser usado por los notebooks posteriores es:

```text
data/processed/01_train_model_raw.csv
data/processed/01_val_model_raw.csv
```

Cada fila de estos archivos representa una comparacion entre dos programas:

```text
pair_id, language, scenario, file_1, file_2, path_1, path_2,
code_1_raw, code_2_raw, label
```

Estos archivos ya tienen la forma que necesita un modelo supervisado:

- entrada 1: `code_1_raw`;
- entrada 2: `code_2_raw`;
- objetivo: `label`.

Lo que todavia no tienen es preprocesamiento del contenido. Esa decision es intencional: este notebook separa la etapa de construccion del dataset de la etapa de transformacion del codigo. Asi se puede conservar una version auditable del corpus original y despues experimentar con diferentes representaciones sin volver a recorrer todo el dataset raw.

## 16. Nota sobre los archivos de test

En `data/processed` existen archivos como `01_test_pairs_raw.csv` y `01_test_model_raw.csv`, pero en las celdas visibles de `01-Preprocesado_de_datos.ipynb` el flujo documentado termina guardando:

```text
00_train_codes_raw.csv
00_test_codes_raw.csv
01_train_positive_pairs_raw.csv
01_train_pairs_raw.csv
01_train_model_raw.csv
01_val_model_raw.csv
```

Por eso, para explicar estrictamente este notebook, los archivos finales principales son `01_train_model_raw.csv` y `01_val_model_raw.csv`. Los archivos `01_test_*` pueden formar parte de una ejecucion posterior, una version extendida del proceso o una etapa adicional, pero no son producidos por las celdas mostradas en este notebook.

## 17. Tamano final de los conjuntos

Con las salidas generadas para entrenamiento y validacion, el dataset supervisado construido por este notebook tiene:

```text
N train + validation = 220 pares
```

La distribucion es:

```text
Train:
  negativos = 88
  positivos = 88
  total     = 176

Validation:
  negativos = 22
  positivos = 22
  total     = 44
```

Por lo tanto:

```text
N = 88 + 88 + 22 + 22 = 220
```

Para testing, los archivos ya existentes en `data/processed` son:

```text
01_test_pairs_raw.csv
01_test_model_raw.csv
```

Ambos contienen el mismo tamano:

```text
Testing:
  negativos = 497
  positivos = 497
  total     = 994
```

Si se reporta el total completo incluyendo train, validation y testing, entonces:

```text
N total = 220 + 994 = 1214 pares
```

La distribucion global queda balanceada:

```text
Negativos totales = 88 + 22 + 497 = 607
Positivos totales = 88 + 22 + 497 = 607
N total           = 1214
```

La diferencia metodologica importante es que `N = 220` corresponde a los pares usados para entrenar y validar durante el desarrollo del modelo, mientras que `N testing = 994` corresponde al conjunto reservado para evaluacion final.
