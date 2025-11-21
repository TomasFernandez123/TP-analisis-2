import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. CONFIGURACIÓN Y DICCIONARIO DE VARIABLES ---

RUTA_CARPETA = 'data/raw' 
aglomerados_a_analizar = [13, 32]
anos_analisis = range(2016, 2026)

# Definición de Variables del Modelo (Mapeo con la EPH)
VARS = {
    # Variable Dependiente (Y)
    'INGRESO': 'P21',          # Ingreso ocupación principal
    
    # Variables Independientes (X)
    'EDAD': 'CH06',            # Edad (Experiencia)
    'SEXO': 'CH04',            # 1=Varón, 2=Mujer
    'EDUCACION': 'NIVEL_ED',   # 1 a 7 (Primaria a Universitaria)
    'HORAS': 'PP3E_TOT',       # Cantidad de horas trabajadas
    'CATEGORIA': 'CAT_OCUP',   # Patrón, Cuenta Propia, Empleado
    'REGION': 'AGLOMERADO',    # Para diferenciar CABA de Córdoba
    
    # Variables Técnicas
    'ESTADO': 'ESTADO',        # Para filtrar solo a los Ocupados (Estado=1)
    'PONDERADOR': 'PONDIIO'    # Ponderador específico para ingresos
}

ipc_data = {
    'Año': [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    'IPC': [100.0, 125.0, 184.5, 284.4, 392.2, 590.1, 1145.9, 3584.8, 7687.3, 10079.43]
}
df_ipc = pd.DataFrame(ipc_data)

# Calculamos el factor para llevar todo a precios de Diciembre 2025
ipc_base_2025 = df_ipc[df_ipc['Año'] == 2025]['IPC'].iloc[0]
df_ipc['Factor_Deflactacion'] = df_ipc['IPC'] / ipc_base_2025

# -------------------------------------------------------------------------
# 2. CARGA, LIMPIEZA Y PREPARACIÓN DE DATOS
# -------------------------------------------------------------------------

list_df = []
print("\n⏳ Cargando y procesando bases de datos (2016-2025)...")

for año in anos_analisis:
    año_sufijo = str(año)[2:]
    
    # Patrones de búsqueda para encontrar los archivos trimestrales
    search_patterns = [
        os.path.join(RUTA_CARPETA, f'*T?{año_sufijo}.txt'), 
        os.path.join(RUTA_CARPETA, f'*4to.trim_{año}.txt'), 
    ]
    # Buscamos todos los archivos que coincidan
    files = sorted(list(set([f for p in search_patterns for f in glob.glob(p)])))
    
    if not files:
        continue
    
    try:
        # Cargamos cada trimestre y lo unificamos en un año
        dfs = [pd.read_csv(f, encoding='latin-1', sep=';', decimal=',', on_bad_lines='skip') for f in files]
        df_anual = pd.concat(dfs, ignore_index=True)
        
        # Estandarizamos nombres de columnas (a mayúsculas y sin espacios)
        df_anual.columns = df_anual.columns.str.upper().str.strip()
        df_anual['ANO_ENCUESTA'] = año 
        
        list_df.append(df_anual)
    except Exception as e:
        print(f"  [!] Error cargando año {año}: {e}")
        continue

# Unimos todos los años en un solo gran DataFrame
df_total = pd.concat(list_df, ignore_index=True)

# --- LIMPIEZA CRÍTICA (Solución a errores previos) ---
# Convertimos a número todas las columnas que usaremos. Si hay error, pone NaN.
cols_a_limpiar = [
    VARS['INGRESO'], VARS['EDAD'], VARS['HORAS'], VARS['EDUCACION'], 
    VARS['CATEGORIA'], VARS['REGION'], VARS['ESTADO'], VARS['PONDERADOR']
]

for col in cols_a_limpiar:
    if col in df_total.columns:
        df_total[col] = pd.to_numeric(df_total[col], errors='coerce')

# --- DEFLACIÓN (Llevar todo a Pesos de 2025) ---
# Unimos con la tabla de IPC para tener el factor correspondiente a cada fila
df_total = df_total.merge(df_ipc[['Año', 'Factor_Deflactacion']], 
                          left_on='ANO_ENCUESTA', right_on='Año', how='left')

# Calculamos el Ingreso Real
df_total['P21_REAL'] = df_total[VARS['INGRESO']] / df_total['Factor_Deflactacion']

# --- FILTROS DE POBLACIÓN (Universo de Estudio) ---
# Nos quedamos solo con: Ocupados, de nuestros Aglomerados, Edad laboral, etc.
df_model = df_total[
    (df_total[VARS['REGION']].isin(aglomerados_a_analizar)) & # Solo 13 y 32
    (df_total[VARS['ESTADO']] == 1) &      # Solo Ocupados
    (df_total[VARS['EDAD']] >= 14) &       # Mayores de 14
    (df_total[VARS['HORAS']] > 0) &        # Que trabajen horas positivas
    (df_total[VARS['EDUCACION']] < 9) &    # Nivel educativo válido
    (df_total[VARS['CATEGORIA']].isin([1, 2, 3])) # Excluimos trabajadores sin pago
].copy()

df_model['PONDIIO_ANUAL'] = df_model[VARS['PONDERADOR']] / 4  # Ajuste anual del ponderador trimestral

# --- CREACIÓN DE VARIABLES PARA EL MODELO (Feature Engineering) ---
# 1. Edad al Cuadrado (Curva de experiencia)
df_model['EDAD_SQ'] = df_model[VARS['EDAD']] ** 2

# 2. Logaritmo del Ingreso Real (Variable a predecir)
# Solo tomamos ingresos positivos para poder calcular el logaritmo
df_train_valid = df_model[df_model['P21_REAL'] > 0].copy()
df_train_valid['LOG_P21_REAL'] = np.log(df_train_valid['P21_REAL'])

print(f"✅ Parte 2 Completada.")
print(f"   Datos limpios y listos para el modelo: {len(df_train_valid):,} registros.")
print(f"   Rango de Ingreso Real (Dic 2025): ${df_train_valid['P21_REAL'].min():,.0f} - ${df_train_valid['P21_REAL'].max():,.0f}")

# -------------------------------------------------------------------------
# 3. ENTRENAMIENTO Y EVALUACIÓN DEL MODELO (CORREGIDO: ESCALA ORIGINAL)
# -------------------------------------------------------------------------

print("\n⚙️ Entrenando modelo de regresión...")

# A. División Train/Test
X_train, X_test = train_test_split(
    df_train_valid, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_train_valid[VARS['REGION']]
)

# B. Fórmula (Log-Lineal)
formula = (
    "LOG_P21_REAL ~ "
    f"{VARS['EDAD']} + EDAD_SQ + "          
    f"C({VARS['SEXO']}) + "                 
    f"C({VARS['EDUCACION']}) + "            
    f"np.log({VARS['HORAS']}) + "           
    f"C({VARS['CATEGORIA']}) + "            
    f"C({VARS['REGION']})"                  
)

# C. Ajuste del Modelo (WLS)
modelo = smf.wls(
    formula, 
    data=X_train, 
    weights=X_train['PONDIIO_ANUAL']
).fit()

print("✅ Modelo entrenado.")

# D. Evaluación de Métricas (EN PESOS - REQUISITO DOCENTE)

# 1. Predecir en Logaritmos
pred_test_log = modelo.predict(X_test)

# 2. Transformar a Pesos Reales (Anti-Log)
pred_test_pesos = np.exp(pred_test_log)
y_test_pesos = X_test['P21_REAL']

# 3. Calcular Métricas en la escala original
r2_pesos = r2_score(y_test_pesos, pred_test_pesos)
rmse_pesos = np.sqrt(mean_squared_error(y_test_pesos, pred_test_pesos))

# ---- NUEVO: Calcular ECM (MSE) y EMA (MAE) ----
ecm_pesos = mean_squared_error(y_test_pesos, pred_test_pesos)
ema_pesos = mean_absolute_error(y_test_pesos, pred_test_pesos)

print("\n📊 RESULTADOS DE LA EVALUACIÓN (EN PESOS REALES):")
print(f"   R² (Test Set): {r2_pesos:.4f} (Capacidad de predecir el ingreso en $)")
print(f"   ECM (MSE): {ecm_pesos:,.0f}")
print(f"   EMA (MAE): {ema_pesos:,.0f}")
print(f"   Error Promedio (RMSE): ${rmse_pesos:,.0f} (Pesos constantes de 2025)")
print("   Nota: El R² en pesos es menor que en logaritmos, pero es la métrica real solicitada.")

# Interpretación de coeficientes
coefs = modelo.params
print("\n📝 INTERPRETACIÓN ECONÓMICA (Coeficientes):")
print(f"   • Retorno a la Universidad (vs Primaria): +{np.exp(coefs.get(f'C({VARS['EDUCACION']})[T.6]', 0)) * 100 - 100:.1f}%")
print(f"   • Brecha de Género (Mujer vs Varón): {np.exp(coefs.get(f'C({VARS['SEXO']})[T.2]', 0)) * 100 - 100:.1f}%")
print(f"   • Prima por vivir en CABA (vs Córdoba): +{np.exp(coefs.get(f'C({VARS['REGION']})[T.32]', 0)) * 100 - 100:.1f}%")

# -------------------------------------------------------------------------
# 4. GRÁFICOS DE DIAGNÓSTICO (Diagnóstico de Regresión)
# -------------------------------------------------------------------------

print("\n📊 Generando gráficos de diagnóstico del modelo...")

# Obtenemos los valores clave del modelo ya entrenado
predicciones = modelo.fittedvalues
residuos = modelo.resid

# --- GRÁFICO 1: RESIDUOS VS. VALORES PREDICHOS (Homocedasticidad) ---
# Este gráfico muestra si el error es constante o si varía con el ingreso.
plt.figure(figsize=(10, 6))
plt.scatter(predicciones, residuos, alpha=0.2, color='blue', s=10)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2) # Línea de referencia en 0
plt.xlabel('Valores Predichos (Log Ingreso Real)')
plt.ylabel('Residuos (Errores)')
plt.title('Gráfico de Residuos vs. Predichos (Homocedasticidad)')
plt.grid(True, alpha=0.3)
plt.savefig('modelo_grafico_residuos.png', dpi=150)
plt.close()
print("✅ Gráfico 'modelo_grafico_residuos.png' generado.")


# --- GRÁFICO 2: Q-Q PLOT (Normalidad) ---
# Compara la distribución de tus residuos con una distribución Normal teórica.
# Si los puntos siguen la línea roja, los errores son normales (lo ideal).
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
sm.qqplot(residuos, line='45', fit=True, ax=ax, alpha=0.2, markerfacecolor='blue', markeredgecolor='none')
ax.set_title('Q-Q Plot de Residuos (Normalidad)')
plt.grid(True, alpha=0.3)
plt.savefig('modelo_qqplot.png', dpi=150)
plt.close()
print("✅ Gráfico 'modelo_qqplot.png' generado.")

# -------------------------------------------------------------------------
# 5. IMPUTACIÓN FINAL Y GRÁFICO COMPARATIVO
# -------------------------------------------------------------------------

print("\n⚙️ Comenzando imputación y análisis comparativo...")

# 1. Identificar NO RESPONDENTES (Ingreso <= 0 o NaN)
# IMPORTANTE: Filtrar CATEGORÍAS VÁLIDAS (1, 2, 3) para evitar error de Patsy
df_a_imputar = df_model[
    ((df_model[VARS['INGRESO']].isna()) | (df_model[VARS['INGRESO']] <= 0)) &
    (df_model[VARS['CATEGORIA']].isin([1, 2, 3]))
].copy()

if not df_a_imputar.empty:
    print(f"   Casos a imputar: {len(df_a_imputar):,} ({len(df_a_imputar)/len(df_model)*100:.1f}% de la muestra)")
    
    # 2. Predecir el Ingreso Real (Logaritmo)
    pred_log_imput = modelo.predict(df_a_imputar)
    
    # 3. Convertir a Pesos Reales (Dic 2025) y Nominales
    df_a_imputar['P21_IMPUTADO_REAL'] = np.exp(pred_log_imput)
    df_a_imputar['P21_IMPUTADO_NOMINAL'] = df_a_imputar['P21_IMPUTADO_REAL'] * df_a_imputar['Factor_Deflactacion']
    df_a_imputar['TIPO_DATO'] = 'Imputado' # Etiqueta para el gráfico

    # Preparar datos REALES para comparar
    df_reales = df_train_valid.copy()
    df_reales['P21_IMPUTADO_REAL'] = df_reales['P21_REAL'] # Usamos la misma columna para unificar
    df_reales['TIPO_DATO'] = 'Real (Declarado)'

    # Unir ambos sets para el gráfico
    df_comparativo = pd.concat([
        df_reales[[VARS['EDUCACION'], 'P21_IMPUTADO_REAL', 'TIPO_DATO']],
        df_a_imputar[[VARS['EDUCACION'], 'P21_IMPUTADO_REAL', 'TIPO_DATO']]
    ])

    # Mapeo de Nivel Educativo para que se lea bien en el gráfico
    map_educacion = {
        1: 'Primaria Inc.', 2: 'Primaria Comp.', 3: 'Secundaria Inc.',
        4: 'Secundaria Comp.', 5: 'Univ. Inc.', 6: 'Univ. Comp.', 7: 'S/Instrucción'
    }
    df_comparativo['NIVEL_EDUCATIVO_TXT'] = df_comparativo[VARS['EDUCACION']].map(map_educacion)
    
    # Ordenar niveles
    orden_niveles = ['S/Instrucción', 'Primaria Inc.', 'Primaria Comp.', 'Secundaria Inc.', 
                     'Secundaria Comp.', 'Univ. Inc.', 'Univ. Comp.']

    # --- GRÁFICO COMPARATIVO: INGRESO REAL PROMEDIO POR EDUCACIÓN ---
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df_comparativo, 
        x='NIVEL_EDUCATIVO_TXT', 
        y='P21_IMPUTADO_REAL', 
        hue='TIPO_DATO',
        order=orden_niveles,
        estimator=np.median, # Usamos la Mediana que es más robusta
        errorbar=None,       # Sin barras de error para limpieza
        palette={'Real (Declarado)': '#1f77b4', 'Imputado': '#ff7f0e'}
    )
    
    plt.title('Comparación: Ingreso Real Mediano (Declarado vs. Imputado) por Nivel Educativo', fontsize=14)
    plt.xlabel('Nivel Educativo', fontsize=12)
    plt.ylabel('Mediana de Ingreso Real ($ Dic 2025)', fontsize=12)
    plt.legend(title='Tipo de Dato')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Formato Eje Y en Miles/Millones
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    
    plt.tight_layout()
    plt.savefig('comparacion_reales_imputados.png', dpi=150)
    plt.close()
    print("✅ Gráfico 'comparacion_reales_imputados.png' generado.")
    
    # 4. Exportar Imputaciones
    cols_export = ['ANO_ENCUESTA', VARS['REGION'], VARS['SEXO'], VARS['EDUCACION'], 
                   'P21_IMPUTADO_REAL', 'P21_IMPUTADO_NOMINAL']
    df_a_imputar[cols_export].head(50).to_csv('tabla_imputaciones_final.csv', index=False)
    print("✅ Archivo 'tabla_imputaciones_final.csv' generado con éxito.")

else:
    print("   [!] No se encontraron casos válidos para imputar.")

print("\n🎉 ¡PUNTO 4 COMPLETADO (CON GRÁFICO COMPARATIVO)!")