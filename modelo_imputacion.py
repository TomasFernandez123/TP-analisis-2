import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib.ticker import FuncFormatter

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_CARPETA = 'data/raw'
TRIMESTRE = 'usu_individual_T225.txt'   # archivo del trimestre

VARS = {
    'INGRESO': 'P21',
    'EDAD': 'CH06',
    'SEXO': 'CH04',
    'EDUCACION': 'NIVEL_ED',
    'HORAS': 'PP3E_TOT',
    'CATEGORIA': 'CAT_OCUP',
    'REGION': 'AGLOMERADO',
    'ESTADO': 'ESTADO',
    'PONDERADOR': 'PONDIIO',
    'FORMALIDAD': 'PP07H',  
    'SECTOR': 'PP04A'
}

AGLOS = {32: "CABA", 13: "Córdoba"}

# ============================================================
# CARGA DEL TRIMESTRE
# ============================================================

df = pd.read_csv(
    os.path.join(RUTA_CARPETA, TRIMESTRE),
    encoding='latin-1',
    sep=';',
    decimal=',',
    on_bad_lines='skip'
)

df.columns = df.columns.str.upper().str.strip()

# Convertir a número
for col in VARS.values():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ============================================================
# FUNCION PARA MODELAR POR AGLOMERADO
# ============================================================

def estimar_modelo_por_aglomerado(df, aglo):

    print(f"\n==============================")
    print(f" MODELO PARA AGLOMERADO {aglo} ({AGLOS[aglo]})")
    print(f"==============================\n")

    # --- Filtrado ---
    df_model = df[
        (df[VARS['REGION']] == aglo) &
        (df[VARS['ESTADO']] == 1) &
        (df[VARS['EDAD']] >= 14) &
        (df[VARS['HORAS']] > 0) &
        (df[VARS['EDUCACION']] < 9) &
        (df[VARS['CATEGORIA']].isin([1, 2, 3])) &
        (df[VARS['FORMALIDAD']].isin([1, 2])) &
        (df[VARS['SECTOR']].isin([1, 2, 3]))
    ].copy()

    df_model["PESO"] = df_model[VARS['PONDERADOR']]
    df_model["EDAD_SQ"] = df_model[VARS['EDAD']] ** 2

    df_validos = df_model[df_model[VARS['INGRESO']] > 0].copy() 

    # --- Tratamiento de outliers SOLO para CABA ---
    if aglo == 32:  # CABA
        q99 = df_validos[VARS['INGRESO']].quantile(0.99)
        print(f"[Aglo {aglo}] Percentil 99 ingreso: {q99:,.0f}")
        
        # Winsorizar: aplastar la cola superior al p99
        df_validos[VARS['INGRESO']] = np.minimum(df_validos[VARS['INGRESO']], q99)

    df_validos["LOG_P21"] = np.log(df_validos[VARS['INGRESO']])

    print("Registros válidos:", len(df_validos))

    # ---------------------------------------
    # Entrenamiento
    # ---------------------------------------
    X_train, X_test = train_test_split(
        df_validos,
        test_size=0.2,
        random_state=42
    )

    formula = (
        "LOG_P21 ~ "
        f"{VARS['EDAD']} + EDAD_SQ + "
        f"C({VARS['SEXO']}) + "
        f"C({VARS['EDUCACION']}) + "
        f"np.log({VARS['HORAS']}) + "
        f"C({VARS['CATEGORIA']}) + "
        f"C({VARS['FORMALIDAD']}) + "
        f"C({VARS['SECTOR']})"
    )

    modelo = smf.wls(
        formula,
        data=X_train,
        weights=X_train["PESO"]
    ).fit()

    print("Modelo entrenado.")

    # ---------------------------------------
    # PREDICCIONES EN ESCALA ORIGINAL
    # ---------------------------------------

    pred_log = modelo.predict(X_test)
    pred_pesos = np.exp(pred_log)
    y_real = X_test[VARS['INGRESO']]

    mse = mean_squared_error(y_real, pred_pesos)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, pred_pesos)
    r2 = r2_score(y_real, pred_pesos)

    print("\n📊 RESULTADOS DEL MODELO")
    print(f"R²:   {r2:.4f}")
    print(f"MAE:  ${mae:,.0f}")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"MSE:  {mse:,.0f}")

    # ============================================================
    # GRAFICOS
    # ============================================================

    # 1. Real vs Predicho
    plt.figure(figsize=(8, 6))
    plt.scatter(pred_pesos, y_real, alpha=0.3)

    max_val = max(pred_pesos.max(), y_real.max())
    plt.plot([0, max_val], [0, max_val], 'r--')

    plt.title(f"Real vs Predicho – Aglomerado {aglo}")
    plt.xlabel("Ingreso Predicho ($)")
    plt.ylabel("Ingreso Real ($)")
    plt.grid(alpha=0.3)

    # --- Formato en pesos ---
    formatter = FuncFormatter(lambda x, _: f"${x:,.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(f"real_vs_predicho_{aglo}.png", dpi=150)
    plt.close()


    # 2. Residuos
    residuos = modelo.resid
    pred_fit = modelo.fittedvalues

    plt.figure(figsize=(8, 6))
    plt.scatter(pred_fit, residuos, alpha=0.3)
    plt.axhline(0, color="red")

    plt.title(f"Residuos vs Predichos – Aglomerado {aglo}")
    plt.xlabel("Valores Predichos")
    plt.ylabel("Residuos")

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"residuos_{aglo}.png", dpi=150)
    plt.close()


    # 3. QQ-plot
    fig = plt.figure(figsize=(8, 6))
    sm.qqplot(residuos, line="45", fit=True, ax=plt.gca())

    plt.title(f"QQ-Plot – Aglomerado {aglo}")

    plt.tight_layout()
    plt.savefig(f"qqplot_{aglo}.png", dpi=150)
    plt.close()


    return modelo


# ============================================================
# EJECUTAR MODELO PARA CADA AGLOMERADO
# ============================================================

for aglo in AGLOS.keys():
    estimar_modelo_por_aglomerado(df, aglo)

print("\n🎉 Modelos por aglomerado generados con éxito.")
