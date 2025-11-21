import pandas as pd
import glob
import os

# --- CONFIGURACIÓN ---
RUTA_CARPETA = 'data/raw' 
AGLOMERADOS_INTERES = [13, 32] # Córdoba y CABA

print("⏳ Procesando datos locales para generar resumen...")

list_df = []
# Procesamos todos los años
for año in range(2016, 2026):
    año_sufijo = str(año)[2:]
    search_patterns = [
        os.path.join(RUTA_CARPETA, f'*T?{año_sufijo}.txt'), 
        os.path.join(RUTA_CARPETA, f'*4to.trim_{año}.txt'), 
    ]
    files = sorted(list(set([f for p in search_patterns for f in glob.glob(p)])))
    
    if files:
        try:
            # Solo cargamos columnas necesarias para ahorrar memoria
            dfs = [pd.read_csv(f, encoding='latin-1', sep=';', decimal=',', 
                               usecols=['AGLOMERADO', 'ESTADO', 'PONDERA'],
                               on_bad_lines='skip') for f in files]
            df_anual = pd.concat(dfs, ignore_index=True)
            list_df.append(df_anual)
            print(f"  -> Año {año} procesado.")
        except: continue

df_total = pd.concat(list_df, ignore_index=True)

# Filtramos solo nuestros aglomerados
df_mapa = df_total[df_total['AGLOMERADO'].isin(AGLOMERADOS_INTERES)].copy()

# Calculamos la Tasa de Desocupación Promedio (TD)
# TD = Suma(PONDERA de Desocupados) / Suma(PONDERA de PEA) * 100
def calcular_td(df):
    pea = df[df['ESTADO'].isin([1, 2])]['PONDERA'].sum()
    desocupados = df[df['ESTADO'] == 2]['PONDERA'].sum()
    return (desocupados / pea * 100) if pea > 0 else 0

# Agrupamos y calculamos
resumen_mapa = df_mapa.groupby('AGLOMERADO').apply(calcular_td).reset_index(name='TASA_DESOCUPACION')

# Guardamos el archivo chiquito
resumen_mapa.to_csv('datos_para_mapa.csv', index=False)
print("\n✅ ¡LISTO! Se generó el archivo 'datos_para_mapa.csv'.")
print("👉 Sube ESTE archivo y 'aglomerados_eph.json' a Google Colab.")
print(resumen_mapa)