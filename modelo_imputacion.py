import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURACIÓN ---
RUTA_CARPETA = 'data/raw' 
aglomerados_a_analizar = [13, 32] # Córdoba y CABA
años_modelo = range(2016, 2026)

# Variables clave del EPH
COL_INGRESO = 'P21'
COL_PONDERADOR = 'PONDIIO'
COL_EDAD = 'CH06'
COL_SEXO = 'CH04'
COL_NIVEL_ED = 'NIVEL_ED'
COL_HORAS = 'PP3E_TOT'
COL_CAT_OCUP = 'CAT_OCUP'
COL_AGLOMERADO = 'AGLOMERADO'

print("="*80)
print(" 🤖 DESARROLLO DE MODELO DE REGRESIÓN (IMPUTACIÓN DE INGRESOS)")
print("="*80)

# -------------------------------------------------------------------------
# 1. CARGA Y PREPARACIÓN DE DATOS (Pooling de años para robustez)
# -------------------------------------------------------------------------
list_df = []

# Para el modelo, tomamos una muestra de los últimos años (ej. 2021-2024)
# para que la inflación no distorsione tanto, o deflactamos. 
# Dado que ya sabemos deflactar, usaremos los datos NOMINALES + variable dummy de AÑO 
# para capturar el efecto inflacionario (Técnica estándar en regresión).

print("Cargando bases de datos...")
for año in años_modelo:
    año_sufijo = str(año)[2:]
    search_patterns = [
        os.path.join(RUTA_CARPETA, f'*T?{año_sufijo}.txt'), 
        os.path.join(RUTA_CARPETA, f'*4to.trim_{año}.txt'), 
    ]
    files = sorted(list(set([f for p in search_patterns for f in glob.glob(p)])))
    
    if not files: continue
    
    try:
        dfs = [pd.read_csv(f, encoding='latin-1', sep=';', decimal=',', on_bad_lines='skip') for f in files]
        df_anual = pd.concat(dfs, ignore_index=True)
        df_anual.columns = df_anual.columns.str.upper().str.strip()
        df_anual['ANO_ENCUESTA'] = año # Variable de control temporal
        list_df.append(df_anual)
    except: continue

df_total = pd.concat(list_df, ignore_index=True)

# --- PREPROCESAMIENTO Y LIMPIEZA DE TIPOS ---

# Convertir columnas críticas a numérico, forzando errores a NaN
cols_a_numerico = [COL_INGRESO, COL_EDAD, COL_HORAS, COL_NIVEL_ED, COL_CAT_OCUP, COL_AGLOMERADO, 'ESTADO']

for col in cols_a_numerico:
    if col in df_total.columns:
        df_total[col] = pd.to_numeric(df_total[col], errors='coerce')

# Filtros básicos de la población objetivo (Ocupados)
df_model = df_total[
    (df_total[COL_AGLOMERADO].isin(aglomerados_a_analizar)) &
    (df_total['ESTADO'] == 1) & 
    (df_total[COL_EDAD] >= 14) &
    (df_total[COL_HORAS] > 0) & # Ahora sí funciona porque es numérico
    (df_total[COL_NIVEL_ED] < 9) & 
    (df_total[COL_CAT_OCUP] > 0)
].copy()

# --- FEATURE ENGINEERING (Variables del TP de Referencia) ---
# 1. Edad al Cuadrado (Para capturar la curva de experiencia)
df_model['EDAD_SQ'] = df_model[COL_EDAD] ** 2

# 2. Logaritmo del Ingreso (Para normalizar la distribución - CLAVE)
# Filtramos ingresos > 0 para el entrenamiento
df_train_valid = df_model[df_model[COL_INGRESO] > 0].copy()
df_train_valid['LOG_P21'] = np.log(df_train_valid[COL_INGRESO])

# -------------------------------------------------------------------------
# 2. ENTRENAMIENTO DEL MODELO
# -------------------------------------------------------------------------

# División Train/Test (80% / 20%)
X_train, X_test = train_test_split(df_train_valid, test_size=0.2, random_state=42)

# Definición de la Fórmula (Sintaxis estilo R, como usa el profesor)
# Log_Ingreso ~ Edad + Edad^2 + Sexo + Nivel_Ed + Horas + Categoria + Aglomerado + Año
formula = (
    "LOG_P21 ~ "
    "CH06 + EDAD_SQ + "             # Edad (Lineal y Cuadrática)
    "C(CH04) + "                    # Sexo (Categórica)
    "C(NIVEL_ED) + "                # Nivel Educativo (Categórica)
    "np.log(PP3E_TOT) + "           # Log de Horas (Elasticidad)
    "C(CAT_OCUP) + "                # Categoría Ocupacional
    "C(AGLOMERADO) + "              # Efecto Regional
    "C(ANO_ENCUESTA)"               # Control de Inflación (Dummy por año)
)

print("\nEntrenando modelo WLS (Mínimos Cuadrados Ponderados)...")
# Usamos WLS (Weighted Least Squares) usando PONDIIO como peso
model = smf.wls(formula, data=X_train, weights=X_train[COL_PONDERADOR]).fit()

# -------------------------------------------------------------------------
# 3. EVALUACIÓN Y VISUALIZACIÓN
# -------------------------------------------------------------------------

# Predicción en Test
pred_log = model.predict(X_test)
pred_nivel = np.exp(pred_log) # Volver a pesos ($)
y_real = X_test[COL_INGRESO]

# Métricas
r2 = r2_score(np.log(y_real), pred_log)
rmse = np.sqrt(mean_squared_error(y_real, pred_nivel))

print("\n" + "-"*60)
print("📊 EVALUACIÓN DEL MODELO")
print("-"*60)
print(f"R² (Ajuste del modelo): {r2:.4f} (Explica el {r2*100:.1f}% de la variabilidad)")
print(f"RMSE (Error medio en pesos): ${rmse:,.0f}")
print("-"*60)

# --- Interpretación de Coeficientes (Para el informe) ---
print("\n📝 INTERPRETACIÓN DE VARIABLES CLAVE:")
coefs = model.params
print(f"• Educación Universitaria (vs Primaria): +{coefs.get('C(NIVEL_ED)[T.6]', 0)*100:.1f}%")
print(f"• Brecha de Género (Mujer vs Hombre): {coefs.get('C(CH04)[T.2]', 0)*100:.1f}%")
print(f"• Por cada 10% más de horas trabajadas, el salario sube: {coefs.get('np.log(PP3E_TOT)', 0)*10:.1f}%")


# --- GRÁFICO 1: VALORES PREDICHOS VS REALES (Como el TP de referencia) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pred_log, y=np.log(y_real), alpha=0.1, color='blue')
plt.plot([0, 15], [0, 15], 'r--', lw=2) # Línea de identidad
plt.xlabel("Log Ingreso Predicho")
plt.ylabel("Log Ingreso Real")
plt.title("Evaluación: Predicción vs Realidad (Escala Logarítmica)")
plt.grid(True)
plt.savefig('modelo_prediccion_vs_real.png')
plt.close()
print("✅ Gráfico 'modelo_prediccion_vs_real.png' generado.")


# --- GRÁFICO 2: RESIDUOS (Validación de supuestos) ---
residuos = model.resid
plt.figure(figsize=(10, 6))
sns.histplot(residuos, kde=True, color='green')
plt.title("Distribución de los Errores del Modelo")
plt.xlabel("Error (Log)")
plt.grid(True)
plt.savefig('modelo_residuos.png')
plt.close()
print("✅ Gráfico 'modelo_residuos.png' generado.")


# -------------------------------------------------------------------------
# 4. IMPUTACIÓN (Aplicación del modelo)
# -------------------------------------------------------------------------

# Identificar no respondentes (P21 <= 0 o NaN)
# IMPORTANTE: Solo podemos imputar para categorías que el modelo conoce (1, 2, 3)
df_no_respondentes = df_model[
    ((df_model[COL_INGRESO].isna()) | (df_model[COL_INGRESO] <= 0)) &
    (df_model[COL_CAT_OCUP].isin([1, 2, 3])) # <--- FILTRO CLAVE AGREGADO
].copy()

# Feature Engineering en el set de imputación
df_no_respondentes['EDAD_SQ'] = df_no_respondentes[COL_EDAD] ** 2

if not df_no_respondentes.empty:
    print(f"\nImputando ingresos para {len(df_no_respondentes):,} casos de no respuesta...")
    
    # Predecir
    pred_imputacion_log = model.predict(df_no_respondentes)
    df_no_respondentes['P21_IMPUTADO'] = np.exp(pred_imputacion_log)
    
    # Guardar muestra
    cols_export = ['ANO_ENCUESTA', 'AGLOMERADO', 'CH04', 'NIVEL_ED', 'P21_IMPUTADO']
    df_no_respondentes[cols_export].head(20).to_csv('tabla_imputaciones_ejemplo.csv', index=False)
    print("✅ Tabla de imputaciones 'tabla_imputaciones_ejemplo.csv' generada.")
    
    # Mostrar resumen del modelo completo para copiar al informe
    with open('modelo_resumen.txt', 'w') as f:
        f.write(model.summary().as_text())
    print("✅ Resumen estadístico guardado en 'modelo_resumen.txt'.")

else:
    print("No se encontraron casos para imputar.")

print("\n¡Proceso finalizado con éxito!")