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
COL_PONDERADOR = 'PONDII'  # 🔑 CORREGIDO: Ponderador de ingresos
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
# 1. CARGA Y PREPARACIÓN DE DATOS
# -------------------------------------------------------------------------
list_df = []

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
        df_anual['ANO_ENCUESTA'] = año
        list_df.append(df_anual)
        print(f"  ✅ Año {año}: {len(df_anual):,} registros")
    except Exception as e:
        print(f"  ⚠️ Error en año {año}: {e}")
        continue

df_total = pd.concat(list_df, ignore_index=True)
print(f"\n📊 Total consolidado: {len(df_total):,} registros")

# --- PREPROCESAMIENTO Y LIMPIEZA DE TIPOS ---
cols_a_numerico = [COL_INGRESO, COL_EDAD, COL_HORAS, COL_NIVEL_ED, COL_CAT_OCUP, 
                   COL_AGLOMERADO, 'ESTADO', COL_PONDERADOR]

for col in cols_a_numerico:
    if col in df_total.columns:
        df_total[col] = pd.to_numeric(df_total[col], errors='coerce')

# 🔑 VALIDACIÓN: Calcular tasa de no respuesta ANTES de filtrar
total_ocupados = df_total[
    (df_total[COL_AGLOMERADO].isin(aglomerados_a_analizar)) &
    (df_total['ESTADO'] == 1) &
    (df_total[COL_EDAD] >= 14)
]

n_total = len(total_ocupados)
n_sin_ingreso = total_ocupados[
    (total_ocupados[COL_INGRESO].isna()) | (total_ocupados[COL_INGRESO] <= 0)
].shape[0]

tasa_no_respuesta = (n_sin_ingreso / n_total) * 100

print(f"\n📈 TASA DE NO RESPUESTA:")
print(f"   Total ocupados: {n_total:,}")
print(f"   Sin ingreso declarado: {n_sin_ingreso:,}")
print(f"   Tasa: {tasa_no_respuesta:.2f}%")

# Filtros para el modelo
df_model = df_total[
    (df_total[COL_AGLOMERADO].isin(aglomerados_a_analizar)) &
    (df_total['ESTADO'] == 1) & 
    (df_total[COL_EDAD] >= 14) &
    (df_total[COL_HORAS] > 0) &
    (df_total[COL_NIVEL_ED] < 9) & 
    (df_total[COL_CAT_OCUP].isin([1, 2, 3])) &  # 🔑 CORREGIDO: Excluir CAT_OCUP=4
    (df_total[COL_PONDERADOR] > 0)  # 🔑 AGREGADO: Validar ponderador
].copy()

print(f"📊 Registros después de filtros: {len(df_model):,}")

# --- FEATURE ENGINEERING ---
df_model['EDAD_SQ'] = df_model[COL_EDAD] ** 2

# Filtrar casos CON ingreso para entrenamiento
df_train_valid = df_model[df_model[COL_INGRESO] > 0].copy()
df_train_valid['LOG_P21'] = np.log(df_train_valid[COL_INGRESO])

print(f"📊 Casos válidos para entrenamiento: {len(df_train_valid):,}")

# -------------------------------------------------------------------------
# 2. ENTRENAMIENTO DEL MODELO
# -------------------------------------------------------------------------

# 🔑 CORREGIDO: División con estratificación por aglomerado
X_train, X_test = train_test_split(
    df_train_valid, 
    test_size=0.2, 
    random_state=42,
    stratify=df_train_valid[COL_AGLOMERADO]  # 🔑 AGREGADO
)

print(f"\n📊 División Train/Test:")
print(f"   Train: {len(X_train):,} ({len(X_train)/len(df_train_valid)*100:.1f}%)")
print(f"   Test:  {len(X_test):,} ({len(X_test)/len(df_train_valid)*100:.1f}%)")

# Fórmula del modelo
formula = (
    "LOG_P21 ~ "
    "CH06 + EDAD_SQ + "
    "C(CH04) + "
    "C(NIVEL_ED) + "
    "np.log(PP3E_TOT) + "
    "C(CAT_OCUP) + "
    "C(AGLOMERADO) + "
    "C(ANO_ENCUESTA)"
)

print("\n🔧 Entrenando modelo WLS...")
model = smf.wls(formula, data=X_train, weights=X_train[COL_PONDERADOR]).fit()

# -------------------------------------------------------------------------
# 3. EVALUACIÓN COMPLETA
# -------------------------------------------------------------------------

# 🔑 AGREGADO: Evaluación en TRAIN y TEST
pred_train_log = model.predict(X_train)
pred_test_log = model.predict(X_test)

pred_train_nivel = np.exp(pred_train_log)
pred_test_nivel = np.exp(pred_test_log)

y_train = X_train[COL_INGRESO]
y_test = X_test[COL_INGRESO]

# Métricas
r2_train = r2_score(np.log(y_train), pred_train_log)
r2_test = r2_score(np.log(y_test), pred_test_log)
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train_nivel))
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test_nivel))

# 🔑 AGREGADO: R² Ajustado
n_train = len(X_train)
n_params = len(model.params)
r2_adj_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - n_params - 1)

print("\n" + "="*80)
print("📊 EVALUACIÓN DEL MODELO")
print("="*80)
print(f"{'Métrica':<30} {'Train':>15} {'Test':>15}")
print("-"*80)
print(f"{'R² (Varianza explicada)':<30} {r2_train:>15.4f} {r2_test:>15.4f}")
print(f"{'R² Ajustado':<30} {r2_adj_train:>15.4f} {'N/A':>15}")
print(f"{'RMSE (Error en $)':<30} ${rmse_train:>14,.0f} ${rmse_test:>14,.0f}")
print("="*80)

# 🔑 AGREGADO: Diagnóstico de sobreajuste
if r2_test < r2_train - 0.1:
    print("⚠️  ADVERTENCIA: Posible sobreajuste (R² test << R² train)")
else:
    print("✅ Modelo generaliza bien (sin sobreajuste significativo)")

# --- INTERPRETACIÓN DE COEFICIENTES (COMPLETA) ---
print("\n" + "="*80)
print("📝 COEFICIENTES DEL MODELO (Variables Significativas)")
print("="*80)

# Extraer coeficientes con p-value < 0.05
coef_df = pd.DataFrame({
    'Variable': model.params.index,
    'Coeficiente': model.params.values,
    'p-value': model.pvalues.values,
    'Significativo': model.pvalues.values < 0.05
})

coef_df['Efecto_%'] = (np.exp(coef_df['Coeficiente']) - 1) * 100

# Mostrar solo variables significativas
coef_signif = coef_df[coef_df['Significativo']].sort_values('p-value')

print(coef_signif[['Variable', 'Efecto_%', 'p-value']].to_string(index=False))
print("="*80)

# Interpretaciones clave
print("\n💡 INTERPRETACIONES CLAVE:")
print("-"*80)

# Sexo
if 'C(CH04)[T.2]' in model.params.index:
    efecto_mujer = (np.exp(model.params['C(CH04)[T.2]']) - 1) * 100
    print(f"• Brecha de género (Mujer vs Varón): {efecto_mujer:+.1f}%")

# Educación
if 'C(NIVEL_ED)[T.6]' in model.params.index:
    efecto_univ = (np.exp(model.params['C(NIVEL_ED)[T.6]']) - 1) * 100
    print(f"• Universitario completo (vs Primaria incompleta): {efecto_univ:+.1f}%")

# Horas
if 'np.log(PP3E_TOT)' in model.params.index:
    elasticidad = model.params['np.log(PP3E_TOT)']
    print(f"• Elasticidad horas trabajadas: {elasticidad:.3f}")
    print(f"  (10% más horas → {elasticidad*10:.1f}% más ingreso)")

# Aglomerado
if 'C(AGLOMERADO)[T.32]' in model.params.index:
    efecto_caba = (np.exp(model.params['C(AGLOMERADO)[T.32]']) - 1) * 100
    print(f"• Prima CABA vs Córdoba: {efecto_caba:+.1f}%")

# Categoría ocupacional
if 'C(CAT_OCUP)[T.3]' in model.params.index:
    efecto_cuentap = (np.exp(model.params['C(CAT_OCUP)[T.3]']) - 1) * 100
    print(f"• Cuenta propia vs Patrón: {efecto_cuentap:+.1f}%")

print("-"*80)

# --- GRÁFICO 1: VALORES PREDICHOS VS REALES ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train
axes[0].scatter(pred_train_log, np.log(y_train), alpha=0.1, color='blue', s=1)
axes[0].plot([8, 15], [8, 15], 'r--', lw=2)
axes[0].set_xlabel("Log Ingreso Predicho")
axes[0].set_ylabel("Log Ingreso Real")
axes[0].set_title(f"Train Set (R²={r2_train:.3f})")
axes[0].grid(True, alpha=0.3)

# Test
axes[1].scatter(pred_test_log, np.log(y_test), alpha=0.2, color='green', s=1)
axes[1].plot([8, 15], [8, 15], 'r--', lw=2)
axes[1].set_xlabel("Log Ingreso Predicho")
axes[1].set_ylabel("Log Ingreso Real")
axes[1].set_title(f"Test Set (R²={r2_test:.3f})")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('modelo_prediccion_vs_real-2.png', dpi=150)
plt.close()
print("\n✅ Gráfico 'modelo_prediccion_vs_real-2.png' generado.")

# --- GRÁFICO 2: RESIDUOS ---
residuos_train = model.resid
residuos_test = np.log(y_test) - pred_test_log

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(residuos_train, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].set_title("Distribución de Residuos - Train")
axes[0].set_xlabel("Error (Log)")
axes[0].set_ylabel("Frecuencia")
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuos_test, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].set_title("Distribución de Residuos - Test")
axes[1].set_xlabel("Error (Log)")
axes[1].set_ylabel("Frecuencia")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('modelo_residuos-2.png', dpi=150)
plt.close()
print("✅ Gráfico 'modelo_residuos-2.png' generado.")

# -------------------------------------------------------------------------
# 4. IMPUTACIÓN
# -------------------------------------------------------------------------

df_no_respondentes = df_model[
    ((df_model[COL_INGRESO].isna()) | (df_model[COL_INGRESO] <= 0))
].copy()

df_no_respondentes['EDAD_SQ'] = df_no_respondentes[COL_EDAD] ** 2

if not df_no_respondentes.empty:
    print(f"\n🔧 Imputando ingresos para {len(df_no_respondentes):,} casos...")
    print(f"   ({(len(df_no_respondentes)/n_total)*100:.2f}% del total de ocupados)")
    
    pred_imputacion_log = model.predict(df_no_respondentes)
    df_no_respondentes['P21_IMPUTADO'] = np.exp(pred_imputacion_log)
    
    # Estadísticas de imputación
    print(f"\n📊 ESTADÍSTICAS DE VALORES IMPUTADOS:")
    print(f"   Media:   ${df_no_respondentes['P21_IMPUTADO'].mean():,.0f}")
    print(f"   Mediana: ${df_no_respondentes['P21_IMPUTADO'].median():,.0f}")
    print(f"   Min:     ${df_no_respondentes['P21_IMPUTADO'].min():,.0f}")
    print(f"   Max:     ${df_no_respondentes['P21_IMPUTADO'].max():,.0f}")
    
    # Guardar muestra
    cols_export = ['ANO_ENCUESTA', 'AGLOMERADO', 'CH04', 'NIVEL_ED', 
                   'PP3E_TOT', 'CAT_OCUP', 'P21_IMPUTADO']
    df_export = df_no_respondentes[cols_export].copy()
    df_export.columns = ['Año', 'Aglomerado', 'Sexo', 'Nivel_Ed', 
                         'Horas', 'Cat_Ocup', 'Ingreso_Imputado']
    df_export.to_csv('ingresos_imputados-2.csv', index=False)
    print(f"✅ Archivo 'ingresos_imputados-2.csv' generado ({len(df_export):,} casos)")
    
    # Resumen del modelo
    with open('modelo_resumen-2.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESUMEN COMPLETO DEL MODELO DE IMPUTACIÓN\n")
        f.write("="*80 + "\n\n")
        f.write(model.summary().as_text())
        f.write("\n\n" + "="*80 + "\n")
        f.write("INTERPRETACIÓN DE COEFICIENTES\n")
        f.write("="*80 + "\n")
        f.write(coef_signif[['Variable', 'Efecto_%', 'p-value']].to_string(index=False))
    
    print("✅ Resumen estadístico guardado en 'modelo_resumen-2.txt'")

else:
    print("\n⚠️ No se encontraron casos para imputar")

print("\n" + "="*80)
print("✅ PROCESO FINALIZADO CON ÉXITO")
print("="*80)