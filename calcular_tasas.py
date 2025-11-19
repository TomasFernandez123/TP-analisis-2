import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- VARIABLES Y CONFIGURACIÓN ---
RUTA_CARPETA = 'data/raw' 
aglomerados_a_analizar = [13, 32]
años_a_analizar = range(2016, 2026) 

# Variables de la EPH:
COL_AGLOMERADO = 'AGLOMERADO'
COL_EDAD = 'CH06'           # Edad
COL_ESTADO = 'ESTADO'        # 1=Ocupado, 2=Desocupado, 3=Inactivo, 4=Menor 10 años
COL_PONDERA = 'PONDERA'      # Ponderador para tasas

# MÉTODO: Igual que el código R de referencia
METODO = 'SUMA_TRIMESTRES_PONDERADOS'

df_tasas_corregidas = []

print("="*80)
print("CÁLCULO DE TASAS - AGLOMERADOS 13 (CÓRDOBA) Y 32 (CABA)")
print(f"Método: {METODO}")
print("="*80)
print()
print("⚠️  IMPORTANTE: Este cálculo usa TODA la población")
print("   (incluyendo menores de 10 años, identificados como ESTADO==4)")
print("   Esto replica el método del TP de referencia")
print()
print("="*80)
print()

for año in años_a_analizar:
    
    # --- 1. CARGA Y UNIFICACIÓN DE DATOS ANUALES ---
    año_sufijo = str(año)[2:]
    search_patterns = [
        os.path.join(RUTA_CARPETA, f'*T?{año_sufijo}.txt'), 
        os.path.join(RUTA_CARPETA, f'*4to.trim_{año}.txt'), 
    ]
    all_quarterly_files = sorted(list(set([f for pattern in search_patterns for f in glob.glob(pattern)])))

    if not all_quarterly_files: 
        print(f"⚠️  Año {año}: No se encontraron archivos de microdatos.")
        continue

    try:
        list_df_quarterly = [
            pd.read_csv(file, encoding='latin-1', sep=';', decimal=',', on_bad_lines='skip') 
            for file in all_quarterly_files
        ]
        df_eph = pd.concat(list_df_quarterly, ignore_index=True)
        
        # Normalizar nombres de columnas
        df_eph.columns = df_eph.columns.str.upper().str.strip()
        
    except Exception as e:
        print(f"❌ Error al cargar datos para {año}: {e}")
        continue
    
    # --- 2. VALIDACIÓN DE COLUMNAS ---
    columnas_requeridas = [COL_AGLOMERADO, COL_ESTADO, COL_PONDERA, COL_EDAD]
    columnas_faltantes = [col for col in columnas_requeridas if col not in df_eph.columns]
    
    if columnas_faltantes:
        print(f"⚠️  Año {año}: Faltan columnas {columnas_faltantes}")
        print(f"   Columnas disponibles: {list(df_eph.columns[:20])}...")
        continue
    
    print(f"✅ Procesando año {año} ({len(df_eph):,} registros)")

    # --- 3. CÁLCULO POR AGLOMERADO ---
    for aglo in aglomerados_a_analizar:
        df_aglo = df_eph[df_eph[COL_AGLOMERADO] == aglo].copy()
        
        # CLAVE: Incluir TODOS los registros (como hace el código R)
        df_pea = df_aglo[df_aglo[COL_PONDERA] > 0].copy()
        
        if df_pea.empty: 
            print(f"   ⚠️  Aglomerado {aglo}: Sin datos")
            continue
        
        # 🔑 CLAVE: Dividir ponderador por 4 (método R)
        df_pea['PONDERA_ANUAL'] = df_pea[COL_PONDERA] / 4
        
        # --- 4. SUMAR TODOS LOS TRIMESTRES (como hace el código R) ---
        # PEA = Ocupados (1) + Desocupados (2)
        suma_pea = df_pea[df_pea[COL_ESTADO].isin([1, 2])]['PONDERA_ANUAL'].sum()
        suma_ocupados = df_pea[df_pea[COL_ESTADO] == 1]['PONDERA_ANUAL'].sum()
        suma_desocupados = df_pea[df_pea[COL_ESTADO] == 2]['PONDERA_ANUAL'].sum()
        
        # POBLACIÓN BASE: TODOS (incluye Inactivos=3 y Menores de 10=4)
        suma_total = df_pea['PONDERA_ANUAL'].sum()
        
        # --- 5. CÁLCULO DE TASAS (MÉTODO R) ---
        if suma_total > 0 and suma_pea > 0:
            # Tasa de Actividad: (PEA / Población Total) × 100
            tasa_actividad = (suma_pea / suma_total) * 100
            
            # Tasa de Empleo: (Ocupados / Población Total) × 100
            tasa_empleo = (suma_ocupados / suma_total) * 100
            
            # Tasa de Desocupación: (Desocupados / PEA) × 100
            tasa_desocupacion = (suma_desocupados / suma_pea) * 100
            
            df_tasas_corregidas.append({
                'Año': año,
                'Aglomerado': aglo,
                'Actividad': round(tasa_actividad, 2),
                'Empleo': round(tasa_empleo, 2),
                'Desocupacion': round(tasa_desocupacion, 2),
                'N_casos': len(df_pea),
                'Población_Total': int(suma_total)
            })
            
            aglo_nombre = "Córdoba" if aglo == 13 else "CABA"
            print(f"   ✓ {aglo_nombre}: TA={tasa_actividad:.1f}% | TE={tasa_empleo:.1f}% | TD={tasa_desocupacion:.1f}%")
        else:
            print(f"   ⚠️  Aglomerado {aglo}: Sumas ponderadas inválidas")
    
    print()

# --- 6. RESULTADOS FINALES ---
if df_tasas_corregidas:
    df_tasas_final = pd.DataFrame(df_tasas_corregidas)
    
    print("\n" + "="*80)
    print("TASAS DE ACTIVIDAD, EMPLEO Y DESOCUPACIÓN - SERIE ANUAL")
    print("Base: Población TOTAL (incluye menores de 10 años)")
    print("="*80)
    print(df_tasas_final[['Año', 'Aglomerado', 'Actividad', 'Empleo', 'Desocupacion']].to_string(index=False))
    print("="*80)
    
    # Agregar nombres de aglomerados para referencia
    print("\nCódigos de aglomerados:")
    print("  13 = Gran Córdoba")
    print("  32 = Ciudad Autónoma de Buenos Aires (CABA)")
    
    # Diagnóstico de resultados
    print("\n📊 DIAGNÓSTICO DE RESULTADOS:")
    
    ta_promedio = df_tasas_final['Actividad'].mean()
    te_promedio = df_tasas_final['Empleo'].mean()
    td_promedio = df_tasas_final['Desocupacion'].mean()
    
    print(f"   • Tasa Actividad promedio: {ta_promedio:.1f}%")
    print(f"   • Tasa Empleo promedio: {te_promedio:.1f}%")
    print(f"   • Tasa Desocupación promedio: {td_promedio:.1f}%")
    
    # Detectar valores extremos (solo desocupación, ya que actividad alta es normal en CABA/Córdoba)
    print("\n🔍 ANÁLISIS DE CRISIS:")
    if (df_tasas_final['Desocupacion'] > 12).any():
        años_crisis = df_tasas_final[df_tasas_final['Desocupacion'] > 12][['Año', 'Aglomerado', 'Desocupacion']]
        print(f"   ⚠️  Años con crisis laboral (desocupación > 12%):")
        for idx, row in años_crisis.iterrows():
            aglo_nombre = "Córdoba" if row['Aglomerado'] == 13 else "CABA"
            print(f"      - {row['Año']} {aglo_nombre}: {row['Desocupacion']:.1f}%")
    else:
        print("   ✅ No se detectaron crisis laborales severas en este período")
    
    # Comparación por aglomerado
    print("\n📈 COMPARACIÓN POR AGLOMERADO:")
    for aglo in aglomerados_a_analizar:
        df_aglo = df_tasas_final[df_tasas_final['Aglomerado'] == aglo]
        aglo_nombre = "Córdoba" if aglo == 13 else "CABA"
        print(f"\n   {aglo_nombre} (código {aglo}):")
        print(f"     - Actividad: {df_aglo['Actividad'].mean():.1f}% promedio")
        print(f"     - Empleo: {df_aglo['Empleo'].mean():.1f}% promedio")
        print(f"     - Desocupación: {df_aglo['Desocupacion'].mean():.1f}% promedio")
    
    print("\n" + "="*80)
    print("NOTA METODOLÓGICA:")
    print("="*80)
    print("• Tasas calculadas sobre población TOTAL (igual que el TP de referencia)")
    print("• Incluye menores de 10 años en el denominador (ESTADO==4)")
    print("• Fórmulas aplicadas:")
    print("  - Tasa Actividad = (PEA / Población Total) × 100")
    print("  - Tasa Empleo = (Ocupados / Población Total) × 100")
    print("  - Tasa Desocupación = (Desocupados / PEA) × 100")
    print("• Ponderador: PONDERA / 4 (promedio anual)")
    print("\n⚠️  NOTA: Este método difiere del estándar internacional (14+ años)")
    print("   pero replica exactamente la metodología del TP de referencia")
    print("="*80)
    
else:
    print("\n❌ No se pudieron calcular tasas")
    print("\nVerifica:")
    print("  1. Ruta de archivos: 'data/raw'")
    print("  2. Nombres de columnas: AGLOMERADO, ESTADO, CH06, PONDERA")
    print("  3. Códigos de aglomerado: 13 (Córdoba), 32 (CABA)")
    print("  4. Formato: separador ';' y decimal ','")

# --- NUEVO BLOQUE: GENERACIÓN DE LOS 3 GRÁFICOS DE EVOLUCIÓN DE TASAS ---

if 'df_tasas_final' in locals() and not df_tasas_final.empty:
    
    tasas_a_graficar = ['Actividad', 'Empleo', 'Desocupacion']
    aglomerado_map = {13: 'Gran Córdoba (13)', 32: 'CABA (32)'}
    
    print("\n" + "="*80)
    print("INICIANDO GENERACIÓN DE GRÁFICOS DE TASAS")
    print("="*80)

    for tasa in tasas_a_graficar:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Pivotear los datos para graficar líneas por aglomerado
        df_plot = df_tasas_final.pivot(index='Año', columns='Aglomerado', values=tasa)

        # Colores (CABA=Naranja, Córdoba=Azul)
        colors = {32: '#ff7f0e', 13: '#1f77b4'}
        
        for aglo_code, aglo_name in aglomerado_map.items():
            if aglo_code in df_plot.columns:
                ax.plot(df_plot.index, df_plot[aglo_code], 
                        label=aglo_name, 
                        color=colors[aglo_code], 
                        marker='o', linewidth=2)

        # Formato de Eje Y
        if tasa == 'Desocupacion':
            # Escala para Desocupación (0 a un máximo razonable para la TD)
            y_max = df_plot.max().max() * 1.2 if df_plot.max().max() > 0 else 15
            y_ticks = np.arange(0, y_max, 2.5)
            ax.set_ylim(bottom=0, top=y_max)
            ax.set_yticks(y_ticks)
        else:
            # Escala para Actividad y Empleo
            ax.set_ylim(bottom=0, top=65) # Maximo de 65% es apropiado para esta base
            ax.set_yticks(np.arange(35, 65, 5))
        
        # Etiquetas y Título
        ax.set_title(f'Evolución Anual de la Tasa de {tasa} (2016-2025)', fontsize=14)
        ax.set_xlabel('Año', fontsize=12)
        ax.set_ylabel(f'Tasa de {tasa} (%)', fontsize=12)
        ax.legend(loc='lower left')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(df_plot.index)
        
        # Guardar la figura
        filename = f'evolucion_tasa_{tasa.lower()}_corregida.png'
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"✅ Gráfico generado: {filename}")

    print("\nLos 3 gráficos de tasas corregidas están listos para ser incluidos en la sección de Evolución de Tasas Anuales.")
else:
    print("\n❌ No se generaron gráficos porque la tabla de tasas está vacía o no existe.")