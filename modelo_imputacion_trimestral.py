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
from matplotlib.ticker import FuncFormatter

# --- CONFIGURACIÓN ---
RUTA_CARPETA = 'data/raw' 
AGLOMERADOS_ANALIZAR = [13, 32]
ANIO_ANALISIS = 2024  # Año elegido
TRIMESTRE = '4to'     # Trimestre elegido (el más reciente y completo)

# Variables EPH
VARS = {
    'INGRESO': 'P21',
    'EDAD': 'CH06',
    'SEXO': 'CH04',
    'EDUCACION': 'NIVEL_ED',
    'HORAS': 'PP3E_TOT',
    'CATEGORIA': 'CAT_OCUP',
    'REGION': 'AGLOMERADO',
    'ESTADO': 'ESTADO',
    'PONDERADOR': 'PONDIIO'
}

print("="*80)
print(f" 🤖 MODELO DE REGRESIÓN (TRIMESTRAL): {TRIMESTRE} Trimestre {ANIO_ANALISIS}")
print("="*80)

# -------------------------------------------------------------------------
# 1. CARGA DE UN SOLO TRIMESTRE
# -------------------------------------------------------------------------

# Buscamos específicamente el archivo del 4to trimestre de 2024
patron = os.path.join(RUTA_CARPETA, f'*{TRIMESTRE}*trim_{ANIO_ANALISIS}.txt')
archivos = glob.glob(patron)

# Si no encuentra con ese nombre, intentamos patrón corto (T424)
if not archivos:
    anio_corto = str(ANIO_ANALISIS)[2:]
    patron = os.path.join(RUTA_CARPETA, f'*T4{anio_corto}.txt')
    archivos = glob.glob(patron)

if not archivos:
    raise Exception(f"No se encontró el archivo para el T4 de {ANIO_ANALISIS}")

print(f"Cargando archivo: {archivos[0]}")
df = pd.read_csv(archivos[0], encoding='latin-1', sep=';', decimal=',', on_bad_lines='skip')
df.columns = df.columns.str.upper().str.strip()

# --- LIMPIEZA Y FILTROS ---
# Convertir a numérico
cols_num = [VARS['INGRESO'], VARS['EDAD'], VARS['HORAS'], VARS['EDUCACION'], 
            VARS['CATEGORIA'], VARS['REGION'], VARS['ESTADO'], VARS['PONDERADOR']]

for col in cols_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Filtros de Población (Ocupados, Aglomerados, Edad, Categoría válida)
df_model = df[
    (df[VARS['REGION']].isin(AGLOMERADOS_ANALIZAR)) &
    (df[VARS['ESTADO']] == 1) &
    (df[VARS['EDAD']] >= 14) &
    (df[VARS['HORAS']] > 0) &
    (df[VARS['EDUCACION']] < 9) &
    (df[VARS['CATEGORIA']].isin([1, 2, 3])) &
    (df[VARS['PONDERADOR']] > 0)
].copy()

# Feature Engineering
df_model['EDAD_SQ'] = df_model[VARS['EDAD']] ** 2

# Datos válidos para entrenar (Ingreso > 0)
df_train_valid = df_model[df_model[VARS['INGRESO']] > 0].copy()
df_train_valid['LOG_P21'] = np.log(df_train_valid[VARS['INGRESO']])

print(f"✅ Datos listos: {len(df_train_valid):,} registros para entrenar.")


# -------------------------------------------------------------------------
# 2. ENTRENAMIENTO (WLS con Logaritmo)
# -------------------------------------------------------------------------

X_train, X_test = train_test_split(
    df_train_valid, test_size=0.2, random_state=42, stratify=df_train_valid[VARS['REGION']]
)

formula = (
    "LOG_P21 ~ "
    f"{VARS['EDAD']} + EDAD_SQ + "
    f"C({VARS['SEXO']}) + "
    f"C({VARS['EDUCACION']}) + "
    f"np.log({VARS['HORAS']}) + "
    f"C({VARS['CATEGORIA']}) + "
    f"C({VARS['REGION']})"
)

print("\n⚙️ Entrenando modelo...")
model = smf.wls(formula, data=X_train, weights=X_train[VARS['PONDERADOR']]).fit()


# -------------------------------------------------------------------------
# 3. EVALUACIÓN (EN PESOS - Requisito Profe)
# -------------------------------------------------------------------------

# Predicción en Test
pred_log = model.predict(X_test)
pred_pesos = np.exp(pred_log)
y_real = X_test[VARS['INGRESO']]

# Métricas
r2 = r2_score(y_real, pred_pesos)
rmse = np.sqrt(mean_squared_error(y_real, pred_pesos))
mae = mean_absolute_error(y_real, pred_pesos)
error_relativo = (mae / y_real.mean()) * 100

print("\n" + "="*60)
print("📊 EVALUACIÓN DEL MODELO (EN PESOS DEL 2024)")
print("="*60)
print(f"   R² (En Pesos): {r2:.4f}")
print(f"   RMSE: ${rmse:,.0f}")
print(f"   EMA (Error Medio Absoluto): ${mae:,.0f}")
print(f"   Error Relativo (EMA / Media): {error_relativo:.1f}%")
print("-" * 60)

# Interpretación
coefs = model.params
print("\n📝 INTERPRETACIÓN:")
print(f"   • Retorno Universitario: +{np.exp(coefs.get(f'C({VARS['EDUCACION']})[T.6]', 0))*100-100:.1f}%")
print(f"   • Brecha Género: {np.exp(coefs.get(f'C({VARS['SEXO']})[T.2]', 0))*100-100:.1f}%")


# -------------------------------------------------------------------------
# 4. GRÁFICOS
# -------------------------------------------------------------------------

# A. Predicción vs Realidad
plt.figure(figsize=(8, 6))
plt.scatter(pred_pesos, y_real, alpha=0.3, color='blue', s=15)
# Línea de identidad hasta el percentil 99 para ver bien
max_val = np.percentile(y_real, 99)
plt.plot([0, max_val], [0, max_val], 'r--', lw=2)
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.xlabel("Ingreso Predicho ($)")
plt.ylabel("Ingreso Real ($)")
plt.title(f"Predicción vs Realidad (T4 2024)\nR²={r2:.2f}")
plt.grid(True, alpha=0.3)
# Formato miles
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${int(x/1000)}k'))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${int(x/1000)}k'))
plt.savefig('modelo_prediccion_trimestral.png', dpi=150)
plt.close()

# B. Comparación Real vs Imputado (Por Nivel Educativo)
df_a_imputar = df_model[
    ((df_model[VARS['INGRESO']].isna()) | (df_model[VARS['INGRESO']] <= 0))
].copy()

if not df_a_imputar.empty:
    df_a_imputar['P21_IMPUTADO'] = np.exp(model.predict(df_a_imputar))
    df_a_imputar['TIPO'] = 'Imputado'
    
    df_reales = df_train_valid.copy()
    df_reales['P21_IMPUTADO'] = df_reales[VARS['INGRESO']]
    df_reales['TIPO'] = 'Real'
    
    df_comp = pd.concat([df_reales, df_a_imputar])
    
    # Mapeo Educación
    map_edu = {1:'Primaria Inc', 2:'Primaria Comp', 3:'Secundaria Inc', 4:'Secundaria Comp', 5:'Univ Inc', 6:'Univ Comp', 7:'S/Inst'}
    df_comp['NIVEL_TXT'] = df_comp[VARS['EDUCACION']].map(map_edu)
    orden = ['S/Inst', 'Primaria Inc', 'Primaria Comp', 'Secundaria Inc', 'Secundaria Comp', 'Univ Inc', 'Univ Comp']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_comp, x='NIVEL_TXT', y='P21_IMPUTADO', hue='TIPO', order=orden, estimator=np.median, errorbar=None, palette={'Real':'#1f77b4', 'Imputado':'#ff7f0e'})
    plt.title('Ingreso Mediano Real vs Imputado (T4 2024)')
    plt.ylabel('Mediana Ingreso ($)')
    plt.xlabel('')
    plt.xticks(rotation=15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparacion_trimestral.png', dpi=150)
    plt.close()
    
    print("✅ Gráficos generados.")