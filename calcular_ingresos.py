import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# --- 1. DATOS DE IPC (Índice de Precios al Consumidor) ---
# Usamos los valores proporcionados para deflactar a Pesos Constantes de Oct-2025.
ipc_data = {
    'Año': [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    'IPC': [100.0, 125.0, 184.5, 284.4, 392.2, 590.1, 1145.9, 3584.8, 7687.3, 9593.8]
}
df_ipc = pd.DataFrame(ipc_data)
ipc_base_2025 = df_ipc[df_ipc['Año'] == 2025]['IPC'].iloc[0]
df_ipc['Factor_Deflactacion'] = df_ipc['IPC'] / ipc_base_2025


# --- 2. FUNCIONES ESTADÍSTICAS PONDERADAS (Numpy y Pandas) ---
# Estas funciones están validadas para el curso (Unidades 2 y 3).

def weighted_mean(values, weights):
    """Calcula la media ponderada."""
    return np.average(values, weights=weights)

def weighted_median(values, weights):
    """Calcula la mediana ponderada."""
    df = pd.DataFrame({'value': values, 'weight': weights}).sort_values('value')
    df['cumulative_weight'] = df['weight'].cumsum()
    median_point = df['weight'].sum() / 2
    median_value = df[df['cumulative_weight'] >= median_point]['value'].iloc[0]
    return median_value

# --- 3. BUCLE PRINCIPAL DE PROCESAMIENTO ---

# La carpeta de origen de los microdatos
RUTA_CARPETA = 'data/raw' 

aglomerados_a_analizar = [13, 32]
años_a_analizar = range(2016, 2026) 
resultados_finales = []

# Nombres de variables a usar (basados en el EPH)
COL_AGLOMERADO = 'AGLOMERADO'
COL_INGRESO_NOMINAL = 'P21' # Ingreso de la Ocupación Principal
COL_PONDERADOR = 'PONDIIO'  # 🔑 CORREGIDO: Ponderador de ingresos individuales
COL_ESTADO = 'ESTADO'      # 🔑 AGREGADO: Para filtrar ocupados (1)
COL_EDAD = 'CH06'          # 🔑 AGREGADO: Para filtrar edad mínima

EDAD_MIN = 14  # 🔑 AGREGADO: Población en edad de trabajar


for año in años_a_analizar:
    
    # El año se representa con los últimos dos dígitos (ej: 16 para 2016, 25 para 2025)
    año_sufijo = str(año)[2:]
    
    # Definir patrón de búsqueda flexible:
    search_patterns = [
        # Patrón para la mayoría de los archivos (usu_individual_T?XX.txt)
        os.path.join(RUTA_CARPETA, f'*T?{año_sufijo}.txt'), 
        
        # Patrón para el archivo de 2020 T4 (EPH_usu_personas_4to.trim_2020.txt)
        # Esto solo lo encontrará para el año 2020, pero el patrón es seguro.
        os.path.join(RUTA_CARPETA, f'*4to.trim_{año}.txt'), 
    ]
    
    all_quarterly_files = []
    for pattern in search_patterns:
        all_quarterly_files.extend(glob.glob(pattern))

    # Eliminar duplicados
    all_quarterly_files = sorted(list(set(all_quarterly_files)))

    # 2. Cargar y concatenar todos los archivos encontrados para el año
    list_df_quarterly = []
    if all_quarterly_files:
        try:
            for file in all_quarterly_files:
                # ¡IMPORTANTE! Usar 'sep' (separador) y 'encoding' correctos para archivos .txt del INDEC.
                df_q = pd.read_csv(file, encoding='latin-1', sep=';', decimal=',', on_bad_lines='skip')
                list_df_quarterly.append(df_q)
            
            # CONCATENACIÓN ANUAL: Unimos los trimestres
            df_eph = pd.concat(list_df_quarterly, ignore_index=True)
            print(f"✅ Año {año}: Bases trimestrales cargadas y unificadas correctamente ({len(all_quarterly_files)} archivos).")
            
        except Exception as e:
            # Aquí te indicará si hay problemas con nombres de columna o separadores.
            print(f"❌ Error al cargar o concatenar archivos del año {año}: {e}")
            continue
    else:
        print(f"⚠️ Año {año}: No se encontraron archivos de microdatos en {RUTA_CARPETA}.")
        continue

    # --- FILTRADO Y CÁLCULO DE INGRESOS ---

    factor_deflactacion = df_ipc[df_ipc['Año'] == año]['Factor_Deflactacion'].iloc[0]
    
    # 🔑 CORREGIDO: Validar columnas necesarias primero
    columnas_necesarias = [COL_AGLOMERADO, COL_INGRESO_NOMINAL, COL_PONDERADOR, 
                          COL_ESTADO, COL_EDAD]
    
    if not all(col in df_eph.columns for col in columnas_necesarias):
        print(f"⚠️  Año {año}: Faltan columnas necesarias")
        continue
    
    # 🔑 CORREGIDO: Filtrado completo (aglomerado, ocupados, edad, ingreso válido, ponderador)
    df_filtrado = df_eph[
        (df_eph[COL_AGLOMERADO].isin(aglomerados_a_analizar)) &
        (df_eph[COL_ESTADO] == 1) &  # Solo ocupados
        (df_eph[COL_EDAD] >= EDAD_MIN) &  # Edad mínima
        (df_eph[COL_INGRESO_NOMINAL] > 0) &  # Ingreso válido
        (df_eph[COL_PONDERADOR] > 0)  # Ponderador válido
    ].copy()

    # c) Calcular el Ingreso Real (P21 / Factor de Deflactación)
    df_filtrado['P21_REAL'] = df_filtrado[COL_INGRESO_NOMINAL] / factor_deflactacion
    
    # 🔑 AGREGADO: Ajuste trimestral del ponderador (consistente con análisis de tasas)
    df_filtrado['PONDIIO_ANUAL'] = df_filtrado[COL_PONDERADOR] / 4
    
    # d) Iterar por aglomerado y calcular ponderados
    for aglo in aglomerados_a_analizar:
        df_aglo = df_filtrado[df_filtrado[COL_AGLOMERADO] == aglo]
        
        ingresos = df_aglo['P21_REAL'].values
        ponderadores = df_aglo['PONDIIO_ANUAL'].values  # 🔑 CORREGIDO: Usar ponderador anual
        
        if len(ingresos) > 0:
            media_real = weighted_mean(ingresos, ponderadores)
            mediana_real = weighted_median(ingresos, ponderadores)
            
            resultados_finales.append({
                'Año': año,
                'Aglomerado': aglo,
                'Media_Real_PONDIIO': media_real,  # 🔑 CORREGIDO: Nombre actualizado
                'Mediana_Real_PONDIIO': mediana_real
            })

# 4. CONSOLIDAR Y MOSTRAR RESULTADOS
df_resultados = pd.DataFrame(resultados_finales)

print("\n--- RESULTADOS DE INGRESO REAL PONDERADO CON PONDIIO (CORREGIDO) ---")
print("Base: Ocupados de 14+ años con ingreso declarado")
print(df_resultados.to_string(index=False))

# ---- 5. GRÁFICO DE BARRAS COMPARATIVO CORREGIDO (FIX) ---

# Preparamos el DataFrame para la visualización (separamos Aglomerados)
# IMPORTANTE: Asumimos que df_resultados ya contiene los datos correctos del cálculo anterior
df_13 = df_resultados[df_resultados['Aglomerado'] == 13]
df_32 = df_resultados[df_resultados['Aglomerado'] == 32]

# --- 1. GENERACIÓN DEL GRÁFICO (REVISADO) ---

# Definición del gráfico
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
r1 = np.arange(len(df_13['Año']))
r2 = [x + bar_width for x in r1]

# Barras para Gran Córdoba (13)
ax.bar(r1, df_13['Mediana_Real_PONDIIO'], color='#1f77b4', width=bar_width, 
       edgecolor='grey', label='Gran Córdoba (13)')

# Barras para CABA (32)
ax.bar(r2, df_32['Mediana_Real_PONDIIO'], color='#ff7f0e', width=bar_width, 
       edgecolor='grey', label='CABA (32)')

# Título y etiquetas
ax.set_xlabel('Año', fontsize=12)
ax.set_ylabel('Mediana ingreso real (P21) en pesos constantes (base oct 2025)', fontsize=10)
ax.set_title('Mediana ingreso real (P21) — Aglos 13 vs 32 (2016–2025)', fontsize=14)

# Configuración de Eje X
ax.set_xticks([r + bar_width/2 for r in range(len(df_13['Año']))])
ax.set_xticklabels(df_13['Año'])

# FIX: Función de formato más precisa para evitar etiquetas duplicadas o redondeos ambiguos
def format_y_tick_fixed(value, pos):
    if value >= 1000000:
        # Usamos .1f (un decimal) para mostrar 1.0M, 1.1M, etc., lo que resuelve la duplicación.
        return f'{value/1000000:.1f}M' 
    elif value >= 1000:
        return f'{int(value/1000)}K'
    return f'{int(value)}'
    
# Importar la herramienta de formato
from matplotlib.ticker import FuncFormatter
ax.yaxis.set_major_formatter(FuncFormatter(format_y_tick_fixed))

# Ajuste automático del límite Y para mejorar la visualización y evitar ticks duplicados
# Obtener el máximo valor y añadir un 10% de margen
y_max = df_resultados['Mediana_Real_PONDIIO'].max()
ax.set_ylim(0, y_max * 1.05)


# Leyenda y Grid
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Guardar la figura
plt.tight_layout()
plt.savefig('mediana_ingreso_real_corregida_fixed.png')
plt.close()

print("El gráfico ha sido regenerado y guardado como 'mediana_ingreso_real_corregida_fixed.png'.")
print("El problema de la etiqueta '1M' duplicada en el eje Y ha sido corregido con formato más preciso.")

# ----------------------------------------------------------------------
# --- BLOQUE DE CÓDIGO PARA GENERACIÓN DE DOS GRÁFICOS DE CUARTILES SEPARADOS ---
# ----------------------------------------------------------------------

# 1. FUNCIÓN PARA CUANTILES PONDERADOS (Ya definida, mantenemos aquí para referencia)
def weighted_quantile(values, weights, quantile):
    """Calcula un cuantil ponderado (Q1=0.25, Q2=0.5, Q3=0.75)."""
    df = pd.DataFrame({'value': values, 'weight': weights}).sort_values('value')
    df['cumulative_weight'] = df['weight'].cumsum()
    quantile_point = df['weight'].sum() * quantile
    quantile_value = df[df['cumulative_weight'] >= quantile_point]['value'].iloc[0]
    return quantile_value

# 2. DEFINICIÓN DE PARÁMETROS (Reutilización de variables globales)
aglomerados_a_analizar = [13, 32]
años_a_analizar = range(2016, 2026) 
COL_AGLOMERADO = 'AGLOMERADO'
COL_INGRESO_NOMINAL = 'P21' 
COL_PONDERADOR = 'PONDIIO'  # 🔑 CORREGIDO
COL_ESTADO = 'ESTADO'      # 🔑 AGREGADO
COL_EDAD = 'CH06'          # 🔑 AGREGADO
RUTA_CARPETA = 'data/raw' 

cuantiles_a_calcular = {
    'Q1': 0.25,
    'Q2': 0.50, # Mediana
    'Q3': 0.75
}

resultados_cuartiles = []

# 3. BUCLE DE PROCESAMIENTO ANUAL para Cuartiles (Mismo cálculo, solo repetido para auto-suficiencia)
for año in años_a_analizar: 
    
    # --- Lógica de Carga de Archivos ---
    año_sufijo = str(año)[2:]
    search_patterns = [os.path.join(RUTA_CARPETA, f'*T?{año_sufijo}.txt'), os.path.join(RUTA_CARPETA, f'*4to.trim_{año}.txt')]
    all_quarterly_files = sorted(list(set([f for pattern in search_patterns for f in glob.glob(pattern)])))

    if not all_quarterly_files: continue

    try:
        list_df_quarterly = [pd.read_csv(file, encoding='latin-1', sep=';', decimal=',', on_bad_lines='skip') for file in all_quarterly_files]
        df_eph = pd.concat(list_df_quarterly, ignore_index=True)
    except Exception: continue
    
    # --- FILTRADO Y CÁLCULO DE INGRESOS REALES (P21_REAL) ---
    factor_deflactacion = df_ipc[df_ipc['Año'] == año]['Factor_Deflactacion'].iloc[0]
    
    # 🔑 CORREGIDO: Filtrado completo
    df_filtrado = df_eph[
        (df_eph[COL_AGLOMERADO].isin(aglomerados_a_analizar)) &
        (df_eph[COL_ESTADO] == 1) &
        (df_eph[COL_EDAD] >= 14) &
        (df_eph[COL_INGRESO_NOMINAL] > 0) &
        (df_eph[COL_PONDERADOR] > 0)
    ].copy()
    
    df_filtrado['P21_REAL'] = df_filtrado[COL_INGRESO_NOMINAL] / factor_deflactacion
    df_filtrado['PONDIIO_ANUAL'] = df_filtrado[COL_PONDERADOR] / 4  # 🔑 AGREGADO

    
    # 4. CÁLCULO DE CUARTILES POR AGLOMERADO
    for aglo in aglomerados_a_analizar:
        df_aglo = df_filtrado[df_filtrado[COL_AGLOMERADO] == aglo]
        
        ingresos = df_aglo['P21_REAL'].values
        ponderadores = df_aglo['PONDIIO_ANUAL'].values  # 🔑 CORREGIDO

        if len(ingresos) > 0:
            row = {'Año': año, 'Aglomerado': aglo}
            for name, quantile in cuantiles_a_calcular.items():
                cuantil_value = weighted_quantile(ingresos, ponderadores, quantile)
                row[name] = cuantil_value
            
            resultados_cuartiles.append(row)

df_cuartiles = pd.DataFrame(resultados_cuartiles)


# -----------------------------------------------------------------------
# --- 5. GENERACIÓN DE LOS DOS GRÁFICOS SEPARADOS ---
# -----------------------------------------------------------------------

# Nombres y códigos de los aglomerados
aglomerado_map = {13: 'Gran Córdoba (13)', 32: 'CABA (32)'}
color_map = {13: '#1f77b4', 32: '#ff7f0e'} 
q_colors = {'Q3': 'green', 'Q2': 'black', 'Q1': 'red'}


# Función de formato Y (ya definida)
from matplotlib.ticker import FuncFormatter
def format_y_tick_quantiles(value, pos):
    if value >= 1000000:
        return f'{value/1000000:.1f}M'
    elif value >= 1000:
        return f'{int(value/1000)}K'
    return f'{int(value)}'


# Bucle para generar los 4 gráficos
for aglo_code, aglo_name in aglomerado_map.items():
    df_plot_aglo = df_cuartiles[df_cuartiles['Aglomerado'] == aglo_code].set_index('Año')
    y_max = df_cuartiles['Q3'].max() # Mantener la escala consistente

    # -----------------------------------------------
    # GRÁFICO SET 1: Q1, Q2 y Q3 como LÍNEAS SEPARADAS
    # -----------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plotear las tres líneas
    ax1.plot(df_plot_aglo.index, df_plot_aglo['Q3'], label='Tercer Cuartil (Q3)', color=q_colors['Q3'], linewidth=2)
    ax1.plot(df_plot_aglo.index, df_plot_aglo['Q2'], label='Mediana (Q2)', color=q_colors['Q2'], linewidth=3, linestyle='--')
    ax1.plot(df_plot_aglo.index, df_plot_aglo['Q1'], label='Primer Cuartil (Q1)', color=q_colors['Q1'], linewidth=2)
    
    # Formato
    ax1.set_title(f'Cuartiles de Ingreso (Líneas Separadas) - {aglo_name}', fontsize=14)
    ax1.set_ylabel('Ingreso Real (Pesos constantes oct 2025)', fontsize=12)
    ax1.set_xlabel('Año', fontsize=12)
    ax1.yaxis.set_major_formatter(FuncFormatter(format_y_tick_quantiles))
    ax1.set_ylim(0, y_max * 1.05)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(df_plot_aglo.index)

    # Guardar el archivo
    filename_lines = f'evolucion_cuartiles_lineas_{aglo_code}.png'
    plt.tight_layout()
    plt.savefig(filename_lines)
    plt.close()
    print(f"✅ Gráfico generado: {filename_lines} (Q1, Q2, Q3 en líneas)")


    # -----------------------------------------------
    # GRÁFICO SET 2: RANGO INTERCUARTÍLICO Y MEDIANA (BANDA)
    # -----------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # 1. Banda (Rango Intercuartílico Q1 a Q3)
    ax2.fill_between(df_plot_aglo.index, 
                     df_plot_aglo['Q1'], 
                     df_plot_aglo['Q3'], 
                     color=color_map[aglo_code], alpha=0.25, 
                     label='Rango Intercuartílico (Q3-Q1)')
    
    # 2. Línea (Mediana Q2)
    ax2.plot(df_plot_aglo.index, df_plot_aglo['Q2'], 
             color=color_map[aglo_code], linewidth=3, marker='o', 
             label='Mediana (Q2)')
    
    # Formato
    ax2.set_title(f'Dispersión y Mediana (Banda) - {aglo_name}', fontsize=14)
    ax2.set_ylabel('Ingreso Real (Pesos constantes oct 2025)', fontsize=12)
    ax2.set_xlabel('Año', fontsize=12)
    ax2.yaxis.set_major_formatter(FuncFormatter(format_y_tick_quantiles))
    ax2.set_ylim(0, y_max * 1.05)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(df_plot_aglo.index)

    # Guardar el archivo
    filename_band = f'evolucion_ric_banda_{aglo_code}.png'
    plt.tight_layout()
    plt.savefig(filename_band)
    plt.close()
    print(f"✅ Gráfico generado: {filename_band} (Banda y Mediana)")
    
print("\n--- ¡4 GRÁFICOS DE CUARTILES GENERADOS CON ÉXITO! ---")

# ----------------------------------------------------------------------
# --- NUEVO BLOQUE DE CÓDIGO: ANÁLISIS MULTIVARIADO DE INGRESOS ---
# (Mediana Ponderada de Ingreso Real por Nivel Educativo)
# ----------------------------------------------------------------------

# Mapeo de NIVEL_ED (según la documentación EPH)
map_nivel_ed_ingreso = {
    1: 'Primaria Incompleta',
    2: 'Primaria Completa',
    3: 'Secundaria Incompleta',
    4: 'Secundaria Completa',
    5: 'Sup. Univ. Incompleta',
    6: 'Sup. Univ. Completa',
    7: 'Sin instrucción',
    # Excluimos 9 (Ns/Nr)
}

resultados_ingresos_multivariado = []

# Iteramos sobre todos los años (2016-2025)
for año in años_a_analizar: 
    
    # --- Repetimos la lógica de carga de archivos ---
    año_sufijo = str(año)[2:]
    search_patterns = [
        os.path.join(RUTA_CARPETA, f'*T?{año_sufijo}.txt'), 
        os.path.join(RUTA_CARPETA, f'*4to.trim_{año}.txt'), 
    ]
    all_quarterly_files = []
    for pattern in search_patterns:
        all_quarterly_files.extend(glob.glob(pattern))
    all_quarterly_files = sorted(list(set(all_quarterly_files)))

    if not all_quarterly_files:
        continue

    try:
        list_df_quarterly = []
        for file in all_quarterly_files:
            df_q = pd.read_csv(file, encoding='latin-1', sep=';', decimal=',', on_bad_lines='skip') 
            list_df_quarterly.append(df_q)
        df_eph = pd.concat(list_df_quarterly, ignore_index=True)
        
    except Exception:
        continue
    
    # --- FILTRADO Y CÁLCULO DE INGRESOS REALES (P21_REAL) ---
    factor_deflactacion = df_ipc[df_ipc['Año'] == año]['Factor_Deflactacion'].iloc[0]
    
    # 🔑 CORREGIDO: Filtrado completo con edad y ponderador
    df_filtrado = df_eph[
        (df_eph[COL_AGLOMERADO].isin(aglomerados_a_analizar)) &
        (df_eph[COL_INGRESO_NOMINAL] > 0) &
        (df_eph['ESTADO'] == 1) &
        (df_eph['CH06'] >= 14) &  # 🔑 AGREGADO: Filtro de edad
        (df_eph['NIVEL_ED'].isin(map_nivel_ed_ingreso.keys())) &
        (df_eph[COL_PONDERADOR] > 0)  # 🔑 AGREGADO: Ponderador válido
    ].copy()
    
    df_filtrado['P21_REAL'] = df_filtrado[COL_INGRESO_NOMINAL] / factor_deflactacion
    df_filtrado['PONDIIO_ANUAL'] = df_filtrado[COL_PONDERADOR] / 4  # 🔑 AGREGADO

    
    # 3. CÁLCULO DE LA MEDIANA PONDERADA POR NIVEL EDUCATIVO
    
    # Agrupamos por Aglomerado y Nivel Educativo
    grouped_levels = df_filtrado.groupby([COL_AGLOMERADO, 'NIVEL_ED'])
    
    for (aglo, nivel_cod), df_group in grouped_levels:
        
        ingresos = df_group['P21_REAL'].values
        ponderadores = df_group['PONDIIO_ANUAL'].values  # 🔑 CORREGIDO: Usar ponderador anual
        
        if len(ingresos) > 0:
            mediana_ponderada = weighted_median(ingresos, ponderadores)
            
            resultados_ingresos_multivariado.append({
                'Año': año,
                'Aglomerado': aglo,
                'Nivel_Educativo_Cod': nivel_cod,
                'Nivel_Educativo': map_nivel_ed_ingreso[nivel_cod],
                'Mediana_Real_PONDIIO': mediana_ponderada  # 🔑 CORREGIDO: Nombre actualizado
            })


df_ingresos_multivariado = pd.DataFrame(resultados_ingresos_multivariado)

print("\n-----------------------------------------------------------------------")
print("--- ANÁLISIS MULTIVARIADO: MEDIANA DE INGRESO REAL POR NIVEL EDUCATIVO ---")
print("Base: Ocupados de 14+ años con ingreso declarado")
print(df_ingresos_multivariado[['Año', 'Aglomerado', 'Nivel_Educativo', 'Mediana_Real_PONDIIO']].to_string(index=False))
print("-----------------------------------------------------------------------")

# ----------------------------------------------------------------------
# --- BLOQUE DE CÓDIGO PARA GRÁFICO BÁSICO MULTIVARIADO DE INGRESOS ---
# ----------------------------------------------------------------------

# --- 1. PREPARACIÓN DE DATOS (Filtrando solo los extremos para simplificar el gráfico) ---

# Seleccionar los niveles a comparar: Superior Completa vs. Primaria Incompleta
niveles_a_plotear = ['Sup. Univ. Completa', 'Primaria Incompleta']
df_plot_niveles = df_ingresos_multivariado[
    df_ingresos_multivariado['Nivel_Educativo'].isin(niveles_a_plotear)
].copy()

# Separar por Aglomerado para el plotting
df_caba = df_plot_niveles[df_plot_niveles['Aglomerado'] == 32]
df_cordoba = df_plot_niveles[df_plot_niveles['Aglomerado'] == 13]


# --- 2. GENERACIÓN DEL GRÁFICO DE LÍNEAS (Matplotlib) ---

fig, ax = plt.subplots(figsize=(10, 6))

# Definición de colores y estilos para distinguir Nivel vs. Región
styles = {
    'Sup. Univ. Completa': {'color': 'green', 'linestyle': '-'},
    'Primaria Incompleta': {'color': 'red', 'linestyle': '-.'}
}


for nivel in niveles_a_plotear:
    style = styles[nivel]
    
    # Datos de CABA (Línea más gruesa)
    df_caba_nivel = df_caba[df_caba['Nivel_Educativo'] == nivel]
    ax.plot(df_caba_nivel['Año'], df_caba_nivel['Mediana_Real_PONDIIO'],
            label=f'CABA: {nivel}', 
            color=style['color'], linewidth=2.5, linestyle=style['linestyle'])
    
    # Datos de Gran Córdoba (Línea más delgada, mismo estilo y color de nivel)
    df_cordoba_nivel = df_cordoba[df_cordoba['Nivel_Educativo'] == nivel]
    ax.plot(df_cordoba_nivel['Año'], df_cordoba_nivel['Mediana_Real_PONDIIO'],
            label=f'GC: {nivel}', 
            color=style['color'], linewidth=1.0, linestyle=style['linestyle'])


# --- Formato Final ---

# Función de formato Y (ya definida anteriormente)
from matplotlib.ticker import FuncFormatter
def format_y_tick_mil(value, pos):
    if value >= 1000000:
        return f'{value/1000000:.1f}M'
    elif value >= 1000:
        return f'{int(value/1000)}K'
    return f'{int(value)}'
    
ax.yaxis.set_major_formatter(FuncFormatter(format_y_tick_mil))
ax.tick_params(axis='x', rotation=45)

ax.set_title('Retorno a la Educación: Mediana Real (2016-2025)', fontsize=14)
ax.set_xlabel('Año', fontsize=12)
ax.set_ylabel('Mediana Ingreso Real (Pesos constantes oct 2025)', fontsize=12)

# Colocamos la leyenda en un lugar legible
ax.legend(title='Aglomerado y Nivel', loc='upper right', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xticks(df_plot_niveles['Año'].unique()) # Asegura que todos los años se muestren

# Guardar la figura
plt.tight_layout()
plt.savefig('multivariado_ingresos_simples.png')
plt.close()

print("\nEl gráfico multivariado simple ha sido generado como 'multivariado_ingresos_simples.png'.")