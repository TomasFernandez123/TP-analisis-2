import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ----------------------------------------------------------------------
# --- BLOQUE DE DEFINICIÓN DE VARIABLES Y FUNCIONES ESTATICAS ---
# ----------------------------------------------------------------------

# --- VARIABLES DE CONFIGURACIÓN ---
RUTA_CARPETA = 'data/raw' 
aglomerados_a_analizar = [13, 32]
años_a_analizar = range(2016, 2026) 

# Nombres de variables EPH
COL_AGLOMERADO = 'AGLOMERADO'
COL_ESTADO = 'ESTADO'          # 1: Ocupado (Para análisis de P21)
COL_INGRESO_NOMINAL = 'P21'    # Ingreso de la Ocupación Principal
COL_PONDERADOR_INGRESO = 'PONDII' # 🔑 CORREGIDO: PONDII para ingresos
COL_SEXO = 'CH04'              # 1: Varón, 2: Mujer
COL_EDAD = 'CH06'              # Edad de la persona
SEXO_MAP = {1: 'Varón', 2: 'Mujer'}

# Edad mínima para análisis laboral
EDAD_MIN = 14  # Población en edad de trabajar


# --- DATOS DE IPC (Corregido a la estimación 10079.43 para Dic-2025) ---
ipc_data = {
    'Año': [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    'IPC': [100.0, 125.0, 184.5, 284.4, 392.2, 590.1, 1145.9, 3584.8, 7687.3, 10079.43]
}
df_ipc = pd.DataFrame(ipc_data)
ipc_base_2025 = df_ipc[df_ipc['Año'] == 2025]['IPC'].iloc[0]
df_ipc['Factor_Deflactacion'] = df_ipc['IPC'] / ipc_base_2025


# --- FUNCIONES ESTADÍSTICAS PONDERADAS ---
# (Se necesita la función de mediana ponderada para el análisis)

def weighted_median(values, weights):
    """Calcula la mediana ponderada (Q2)."""
    df = pd.DataFrame({'value': values, 'weight': weights}).sort_values('value')
    df['cumulative_weight'] = df['weight'].cumsum()
    median_point = df['weight'].sum() / 2
    median_value = df[df['cumulative_weight'] >= median_point]['value'].iloc[0]
    return median_value


# ----------------------------------------------------------------------
# --- BLOQUE DE CÁLCULO PRINCIPAL ---
# ----------------------------------------------------------------------

resultados_ingresos_sexo = []

print("="*80)
print("INICIANDO ANÁLISIS MULTIVARIADO: INGRESO REAL POR SEXO")
print("="*80)

for año in años_a_analizar:
    
    # --- 1. CARGA DE DATOS ANUALES (Lógica de unificación trimestral) ---
    año_sufijo = str(año)[2:]
    search_patterns = [
        os.path.join(RUTA_CARPETA, f'*T?{año_sufijo}.txt'), 
        os.path.join(RUTA_CARPETA, f'*4to.trim_{año}.txt'), 
    ]
    all_quarterly_files = sorted(list(set([f for pattern in search_patterns for f in glob.glob(pattern)])))

    if not all_quarterly_files: continue

    try:
        list_df_quarterly = [
            pd.read_csv(file, encoding='latin-1', sep=';', decimal=',', on_bad_lines='skip') 
            for file in all_quarterly_files
        ]
        df_eph = pd.concat(list_df_quarterly, ignore_index=True)
        df_eph.columns = df_eph.columns.str.upper().str.strip()
    except Exception: continue
    
    # --- 2. FILTRADO: POBLACIÓN OCUPADA CON INGRESO Y SEXO VÁLIDO ---
    
    # Validar columnas necesarias
    columnas_necesarias = [COL_AGLOMERADO, COL_ESTADO, COL_INGRESO_NOMINAL, 
                          COL_SEXO, COL_EDAD, COL_PONDERADOR_INGRESO]
    
    if not all(col in df_eph.columns for col in columnas_necesarias):
        print(f"⚠️  Año {año}: Faltan columnas necesarias")
        continue
    
    # Filtramos por Aglomerado, Ocupado (ESTADO=1), Ingreso Válido (P21>0), Sexo Válido (1 o 2), Edad >= 14
    df_filtrado = df_eph[
        (df_eph[COL_AGLOMERADO].isin(aglomerados_a_analizar)) &
        (df_eph[COL_ESTADO] == 1) & 
        (df_eph[COL_INGRESO_NOMINAL] > 0) &
        (df_eph[COL_SEXO].isin([1, 2])) &
        (df_eph[COL_EDAD] >= EDAD_MIN) &  # 🔑 AGREGADO: Filtro de edad
        (df_eph[COL_PONDERADOR_INGRESO] > 0)  # 🔑 AGREGADO: Ponderador válido
    ].copy()
    
    if df_filtrado.empty: 
        print(f"⚠️  Año {año}: Sin datos válidos")
        continue
    
    # --- 3. CÁLCULO DE INGRESO REAL ---
    factor_deflactacion = df_ipc[df_ipc['Año'] == año]['Factor_Deflactacion'].iloc[0]
    df_filtrado['P21_REAL'] = df_filtrado[COL_INGRESO_NOMINAL] / factor_deflactacion
    
    # 🔑 AGREGADO: Ajuste trimestral del ponderador (consistente con análisis de tasas)
    df_filtrado['PONDII_ANUAL'] = df_filtrado[COL_PONDERADOR_INGRESO] / 4
    
    print(f"✅ Año {año}: {len(df_filtrado):,} registros válidos")
    
    # --- 4. AGRUPACIÓN Y CÁLCULO DE LA MEDIANA PONDERADA ---
    
    grouped_sexo = df_filtrado.groupby([COL_AGLOMERADO, COL_SEXO])
    
    for (aglo, sexo_cod), df_group in grouped_sexo:
        
        ingresos = df_group['P21_REAL'].values
        # 🔑 CORREGIDO: Usar PONDII_ANUAL (ponderador ajustado)
        ponderadores = df_group['PONDII_ANUAL'].values 
        
        if len(ingresos) > 0 and ponderadores.sum() > 0:
            mediana_ponderada = weighted_median(ingresos, ponderadores)
            
            n_ponderado = ponderadores.sum()
            
            resultados_ingresos_sexo.append({
                'Año': año,
                'Aglomerado': aglo,
                'Sexo': SEXO_MAP[sexo_cod],
                'Mediana_Real': mediana_ponderada,
                'N_Ponderado': int(n_ponderado)
            })


df_ingresos_sexo = pd.DataFrame(resultados_ingresos_sexo)


# ----------------------------------------------------------------------
# --- BLOQUE DE RESULTADOS Y GRÁFICO ---
# ----------------------------------------------------------------------

# --- TABLA DE SALIDA ---
print("\n" + "="*80)
print("MEDIANA DE INGRESO REAL POR SEXO (2016-2025)")
print("Base: Ocupados de 14+ años con ingreso declarado")
print("Valores en pesos constantes de diciembre 2025")
print("="*80)

# Formatear para mejor visualización
df_display = df_ingresos_sexo.copy()
df_display['Mediana_Format'] = df_display['Mediana_Real'].apply(lambda x: f"${x:,.0f}")
df_display['N_Format'] = df_display['N_Ponderado'].apply(lambda x: f"{x:,}")

print(df_display[['Año', 'Aglomerado', 'Sexo', 'Mediana_Format', 'N_Format']].to_string(index=False))
print("="*80)

# 🔑 AGREGADO: Análisis de brecha de género
print("\n📊 ANÁLISIS DE BRECHA DE GÉNERO:")
print("="*80)

for aglo in aglomerados_a_analizar:
    aglo_name = "CABA (32)" if aglo == 32 else "Gran Córdoba (13)"
    print(f"\n{aglo_name}:")
    
    df_aglo = df_ingresos_sexo[df_ingresos_sexo['Aglomerado'] == aglo]
    
    for año in sorted(df_aglo['Año'].unique()):
        df_año = df_aglo[df_aglo['Año'] == año]
        
        if len(df_año) == 2:  # Debe haber datos de ambos sexos
            mediana_varon = df_año[df_año['Sexo'] == 'Varón']['Mediana_Real'].values[0]
            mediana_mujer = df_año[df_año['Sexo'] == 'Mujer']['Mediana_Real'].values[0]
            
            if mediana_mujer > 0:
                brecha_porcentual = ((mediana_varon / mediana_mujer) - 1) * 100
                
                print(f"  {año}: Varón ${mediana_varon:>10,.0f} | Mujer ${mediana_mujer:>10,.0f} | "
                      f"Brecha: {brecha_porcentual:>+6.1f}%")

print("\n" + "="*80)


# --- GRÁFICO DE LÍNEAS (BRECHA DE GÉNERO) ---

if not df_ingresos_sexo.empty:

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colores: Varón (Azul), Mujer (Naranja). Estilo: CABA (Sólido), GC (Punteado)
    color_varon = '#1f77b4' 
    color_mujer = '#ff7f0e' 
    
    for aglo in aglomerados_a_analizar:
        df_aglo = df_ingresos_sexo[df_ingresos_sexo['Aglomerado'] == aglo]
        aglo_name = "CABA" if aglo == 32 else "GC"

        df_varon = df_aglo[df_aglo['Sexo'] == 'Varón']
        df_mujer = df_aglo[df_aglo['Sexo'] == 'Mujer']
        
        linestyle_aglo = '-' if aglo == 32 else '--'
        
        # Graficar Varones
        ax.plot(df_varon['Año'], df_varon['Mediana_Real'], 
                label=f'{aglo_name}: Varón', 
                color=color_varon, linewidth=2, linestyle=linestyle_aglo, marker='.')

        # Graficar Mujeres
        ax.plot(df_mujer['Año'], df_mujer['Mediana_Real'], 
                label=f'{aglo_name}: Mujer', 
                color=color_mujer, linewidth=2, linestyle=linestyle_aglo, marker='.')


    # Formato Final
    def format_y_tick_mil(value, pos):
        if value >= 1000000:
            return f'{value/1000000:.1f}M'
        elif value >= 1000:
            return f'{int(value/1000)}K'
        return f'{int(value)}'
        
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_tick_mil))
    
    ax.set_title('Mediana de Ingreso Real por Sexo (Brecha de Género)', fontsize=14)
    ax.set_xlabel('Año', fontsize=12)
    ax.set_ylabel('Mediana Ingreso Real (Pesos constantes dic 2025)', fontsize=12)
    
    ax.legend(title='Aglomerado y Sexo', loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(df_ingresos_sexo['Año'].unique())

    plt.tight_layout()
    plt.savefig('multivariado_ingresos_sexo.png')
    plt.close()
    
    print("\nEl gráfico multivariado de Ingresos por Sexo ha sido generado como 'multivariado_ingresos_sexo.png'.")