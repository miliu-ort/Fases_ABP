import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import griddata
from skimage.graph import route_through_array
from scipy.ndimage import gaussian_filter1d, gaussian_filter,binary_closing, binary_opening, binary_fill_holes
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import maximum_filter
import sqlite3
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
# ==========================================
# CONFIGURACIÓN DEL ENTORNO
# ==========================================
st.set_page_config(page_title="Simulador Vial - Grupo 2", layout="wide")

if 'df' not in st.session_state: st.session_state.df = None
if 'ancho_via' not in st.session_state: st.session_state.ancho_via = 10.0
if 'presupuesto_tierra' not in st.session_state: st.session_state.presupuesto_tierra = 40000.0

# ==========================================
# MENÚ LATERAL
# ==========================================
st.sidebar.title("Etapas del Proyecto")
st.sidebar.write("Navegación:")
opcion = st.sidebar.radio(
    "",
    (
        "1. Ingesta de Datos",
        "2. Auditoría Espacial 3D",
        "3. Esqueleto Estructural (TIN)",
        "4. Superficie Sólida (MDE)",
        "5. Maqueta Topográfica (Bloque 3D)",
        "6. Parámetros de Diseño",
        "7. Diseño de Eje y Rasante",
        "8. Maqueta de Excavación 3D",
        "9. Base de Datos (Archivero)",
        "10. Emisión de Memoria (PDF)"
    )
)
st.sidebar.write("---")
st.sidebar.info("Ingeniería Vial Computacional - Grupo 2")

st.title("🚜 Simulador Vial y Movimiento de Tierras")

# ==========================================
# LÓGICA DE PANTALLAS
# ==========================================

# ------------------------------------------
# FASE 1
# ------------------------------------------
if opcion == "1. Ingesta de Datos":
    st.header("Fase 1: Leer la libreta topográfica")

    archivo_subido = st.file_uploader("Sube tu levantamiento (.txt o .csv)", type=["txt", "csv"])
    if archivo_subido is not None:
        try:
            df_raw = pd.read_csv(archivo_subido, sep=r'[,\s;]+', engine='python', header=None)
            df_temp = df_raw.iloc[:, [1, 2, 3]] if len(df_raw.columns) >= 4 else df_raw.iloc[:, [0, 1, 2]]
            df_temp.columns = ['X', 'Y', 'Z']
            for col in ['X', 'Y', 'Z']: df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
            df_temp = df_temp.dropna()

            if len(df_temp) > 0:
                st.session_state.df = df_temp
                st.success(f"✅ ¡Datos en memoria! {len(df_temp)} puntos topográficos listos para procesar.")
            else:
                st.error("❌ El archivo está vacío.")
        except Exception as e:
            st.error(f"Error de lectura: {e}")

    if st.session_state.df is not None:
        df_actual = st.session_state.df
        z_max, z_min = df_actual['Z'].max(), df_actual['Z'].min()
        st.write("")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cota Máxima (Z)", f"{z_max:,.3f} m")
        col2.metric("Cota Mínima (Z)", f"{z_min:,.3f} m")
        col3.metric("Desnivel Topográfico", f"{(z_max - z_min):,.3f} m")

        with st.expander("Ver tabla de coordenadas"):
            st.dataframe(df_actual, use_container_width=True)

elif st.session_state.df is None:
    st.warning("⚠️ Ve a la 'Fase 1: Ingesta de Datos' y carga un archivo primero.")

# ------------------------------------------
# FASE 2
# ------------------------------------------
elif opcion == "2. Auditoría Espacial 3D":
    st.header("Fase 2: Auditoría Espacial 3D")
    df = st.session_state.df
    fig = go.Figure(data=[go.Scatter3d(x=df['X'], y=df['Y'], z=df['Z'], mode='markers',
                                       marker=dict(size=2, color=df['Z'], colorscale='Viridis'))])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=700, paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# FASE 3
# ------------------------------------------
elif opcion == "3. Esqueleto Estructural (TIN)":
    st.header("🕸️ Fase 3: Triangulación")
    df = st.session_state.df
    longitud_maxima = st.slider("Longitud máxima de conexión (m):", 5.0, 150.0, 50.0)
    with st.spinner("Construyendo..."):
        pts_2d = df[['X', 'Y']].values
        tri = Delaunay(pts_2d)
        simplices = tri.simplices
        v1, v2, v3 = pts_2d[simplices[:, 0]], pts_2d[simplices[:, 1]], pts_2d[simplices[:, 2]]
        max_len = np.maximum(np.linalg.norm(v1 - v2, axis=1),
                             np.maximum(np.linalg.norm(v2 - v3, axis=1), np.linalg.norm(v3 - v1, axis=1)))
        simplices_filtrados = simplices[max_len <= longitud_maxima]
        x_tri, y_tri, z_tri = df['X'].values[simplices_filtrados], df['Y'].values[simplices_filtrados], df['Z'].values[
            simplices_filtrados]
        x_lines = np.c_[x_tri, x_tri[:, 0], np.full(len(x_tri), np.nan)].flatten()
        y_lines = np.c_[y_tri, y_tri[:, 0], np.full(len(y_tri), np.nan)].flatten()
        z_lines = np.c_[z_tri, z_tri[:, 0], np.full(len(z_tri), np.nan)].flatten()

        fig = go.Figure(
            data=[go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', line=dict(color='white', width=1))])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=700, paper_bgcolor='black')
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# FASE 4
# ------------------------------------------
elif opcion == "4. Superficie Sólida (MDE)":
    st.header("4. Superficie Sólida (MDE)")
    st.info(
        "💡 Curvas de nivel fijadas cada 5 metros. Utiliza el deslizador para eliminar las áreas vacías de los bordes.")
    df = st.session_state.df

    col1, col2 = st.columns([1, 3])
    with col1:
        resolucion = st.slider("Resolución (m):", 1, 10, 2)
        distancia_max = st.slider("Recorte de bordes (TIN max len):", 5.0, 150.0, 50.0)

    with col2:
        with st.spinner("Interpolando terreno..."):
            x_min, x_max = df['X'].min(), df['X'].max()
            y_min, y_max = df['Y'].min(), df['Y'].max()
            rango_x, rango_y = x_max - x_min, y_max - y_min

            grid_x, grid_y = np.mgrid[
                x_min:x_max:complex(0, rango_x / resolucion), y_min:y_max:complex(0, rango_y / resolucion)]
            puntos_2d = df[['X', 'Y']].values
            grid_z = griddata(puntos_2d, df['Z'].values, (grid_x, grid_y), method='linear')

            # Recorte usando la matemática espacial del TIN
            tri = Delaunay(puntos_2d)
            p = puntos_2d[tri.simplices]
            max_len = np.max([np.linalg.norm(p[:, 0] - p[:, 1], axis=1),
                              np.linalg.norm(p[:, 1] - p[:, 2], axis=1),
                              np.linalg.norm(p[:, 2] - p[:, 0], axis=1)], axis=0)

            simplex_indices = tri.find_simplex(np.c_[grid_x.ravel(), grid_y.ravel()])
            es_valido = np.zeros(len(simplex_indices), dtype=bool)
            adentro = simplex_indices != -1
            es_valido[adentro] = max_len[simplex_indices[adentro]] <= distancia_max

            grid_z[~es_valido.reshape(grid_x.shape)] = np.nan

            fig = go.Figure(data=[go.Surface(
                x=grid_x[:, 0], y=grid_y[0, :], z=grid_z.T, colorscale='Earth',
                contours=dict(z=dict(show=True, color="white", usecolormap=False, project_z=False, size=2))
            )])
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(
                aspectratio=dict(x=rango_x / max(rango_x, rango_y), y=rango_y / max(rango_x, rango_y), z=0.3)),
                              height=700)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# FASE 5
# ------------------------------------------
elif opcion == "5. Maqueta Topográfica (Bloque 3D)":
    st.header("5. Maqueta Topográfica (Bloque 3D)")
    st.info(
        "💡 Construimos la maqueta. Ajusta el deslizador de Recorte hasta que el bloque sea macizo pero sin líneas fantasmas.")
    df = st.session_state.df

    col1, col2, col3 = st.columns(3)
    with col1:
        resolucion = st.slider("Resolución (m):", 1, 10, 2)
    with col2:
        distancia_max = st.slider("Recorte de bordes (TIN max len):", 5.0, 150.0, 50.0)
    with col3:
        profundidad = st.slider("Profundidad base (m):", 5, 100, 10)

    with st.spinner("Esculpiendo bloque..."):
        x_min, x_max, y_min, y_max = df['X'].min(), df['X'].max(), df['Y'].min(), df['Y'].max()
        rango_x, rango_y = x_max - x_min, y_max - y_min
        cota_base = np.floor(df['Z'].min()) - profundidad

        grid_x, grid_y = np.mgrid[
            x_min:x_max:complex(0, rango_x / resolucion), y_min:y_max:complex(0, rango_y / resolucion)]
        puntos_2d = df[['X', 'Y']].values

        grid_z = griddata(puntos_2d, df['Z'].values, (grid_x, grid_y), method='linear')
        grid_z = np.where(np.isnan(grid_z), griddata(puntos_2d, df['Z'].values, (grid_x, grid_y), method='nearest'),
                          grid_z)

        # Usar Delaunay para recortar la maqueta
        tri = Delaunay(puntos_2d)
        p = puntos_2d[tri.simplices]
        max_len = np.max([np.linalg.norm(p[:, 0] - p[:, 1], axis=1),
                          np.linalg.norm(p[:, 1] - p[:, 2], axis=1),
                          np.linalg.norm(p[:, 2] - p[:, 0], axis=1)], axis=0)

        simplex_indices = tri.find_simplex(np.c_[grid_x.ravel(), grid_y.ravel()])
        es_valido = np.zeros(len(simplex_indices), dtype=bool)
        adentro = simplex_indices != -1
        es_valido[adentro] = max_len[simplex_indices[adentro]] <= distancia_max

        grid_z[~es_valido.reshape(grid_x.shape)] = cota_base

        fig = go.Figure()
        fig.add_trace(go.Surface(x=grid_x[:, 0], y=grid_y[0, :], z=grid_z.T, colorscale='Earth',
                                 contours=dict(z=dict(show=True, usecolormap=True, project_z=True, size=2))))
        fig.add_trace(
            go.Surface(x=grid_x[:, 0], y=grid_y[0, :], z=np.full_like(grid_z, cota_base).T, colorscale='Greys',
                       showscale=False, opacity=1.0, hoverinfo='skip'))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(
            aspectratio=dict(x=rango_x / max(rango_x, rango_y), y=rango_y / max(rango_x, rango_y), z=0.4),
            zaxis=dict(range=[cota_base - 2, np.nanmax(grid_z) + 5])), height=800)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# FASE 6
# ------------------------------------------
elif opcion == "6. Parámetros de Diseño":
    st.header("Fase 6: Geometría y Presupuesto")
    col1, col2 = st.columns(2)
    with col1:
        ancho_via = st.number_input("Ancho de vía (W en metros):", min_value=3.0,
                                    value=float(st.session_state.ancho_via), format="%.2f")
    with col2:
        presupuesto_tierra = st.number_input("Presupuesto de Tierra (Volumen Máximo M³):", min_value=100.0,
                                             value=float(st.session_state.presupuesto_tierra), format="%.2f")

    st.write("")
    if st.button("💾 Guardar Parámetros", use_container_width=True):
        st.session_state.ancho_via = ancho_via
        st.session_state.presupuesto_tierra = presupuesto_tierra
        st.success("✅ Guardado. Avanza a la Fase 7.")

# ------------------------------------------
# FASE 7  (CORREGIDA)
# ------------------------------------------
elif opcion == "7. Diseño de Eje y Rasante":
    st.header("2. Diseño de Rasante (Pendientes por Tramos)")
    df = st.session_state.df
    x_min, x_max = df['X'].min(), df['X'].max()
    y_min, y_max = df['Y'].min(), df['Y'].max()

    # FIX RAÍZ: el trazado ahora nace en el punto MÁS BAJO real del
    # levantamiento (Estaca 0+000) y termina en el punto MÁS ALTO real
    # (Meta). Antes se generaba una curva arbitraria dentro del rectángulo
    # x_min→x_max, que en terrenos no alineados con el eje X se salía del
    # área realmente levantada y producía saltos irreales de cota (el salto
    # brusco que se ve en la línea amarilla punteada de tus capturas).
    idx_bajo = df['Z'].idxmin()
    idx_alto = df['Z'].idxmax()
    p_inicio = df.loc[idx_bajo, ['X', 'Y']].values.astype(float)
    p_final = df.loc[idx_alto, ['X', 'Y']].values.astype(float)

    vector_dir = p_final - p_inicio
    longitud_linea = np.linalg.norm(vector_dir)
    dir_unit = vector_dir / longitud_linea
    perp_unit = np.array([-dir_unit[1], dir_unit[0]])

    st.info(f"📏 Longitud total trazada (punto más bajo → punto más alto): {longitud_linea:.2f} metros.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        p1 = st.number_input("K0+0 a K0+100 (%)", value=10.0, format="%.2f")
    with col2:
        p2 = st.number_input("K0+100 a K0+200 (%)", value=10.0, format="%.2f")
    with col3:
        p3 = st.number_input("K0+200 a K0+300 (%)", value=10.0, format="%.2f")
    with col4:
        p4 = st.number_input("K0+300 a K0+400 (%)", value=10.0, format="%.2f")

    col5, _, _, _ = st.columns(4)
    with col5:
        p5 = st.number_input("K0+400 a K0+450 (%)", value=10.0, format="%.2f")

    if st.button("🚜 Calcular Rasante Multitramo en 3D", use_container_width=True):
        st.progress(100, text="¡COMPLETADO!")
        st.session_state.pendientes = [p1, p2, p3, p4, p5]

        # FIX RUTA: en vez de una curva geométrica fija (línea recta + seno),
        # ahora se busca la RUTA MÁS FACTIBLE sobre el propio terreno: la que
        # menos se aleja de una pendiente cómoda de diseño, dando rodeos en
        # las zonas más empinadas para reducir el corte/relleno (como haría
        # un ingeniero vial real), en vez de subir en línea recta.
        with st.spinner("Buscando la ruta más viable sobre el terreno..."):
            rango_x, rango_y = x_max - x_min, y_max - y_min
            n_max_celdas = 350
            resolucion_ruta = max(max(rango_x, rango_y) / n_max_celdas, 1.0)
            nx = max(int(rango_x / resolucion_ruta), 10)
            ny = max(int(rango_y / resolucion_ruta), 10)

            grid_x, grid_y = np.mgrid[x_min:x_max:complex(0, nx), y_min:y_max:complex(0, ny)]
            puntos_2d = df[['X', 'Y']].values
            grid_z_ruta = griddata(puntos_2d, df['Z'].values, (grid_x, grid_y), method='linear')
#77777777777
            tri_ruta = Delaunay(puntos_2d)

            # SOLO casco convexo real del levantamiento (sin filtro de
            # densidad): ese filtro está pensado para la CALIDAD VISUAL del
            # render de la Fase 8, no para decidir qué es transitable. Al
            # copiarlo aquí, dejaba solo 11% del terreno disponible en islas
            # sueltas, forzando la ruta a la única franja conectada (el
            # borde). Con solo el casco convexo, toda el área realmente
            # levantada queda disponible para explorar rutas internas.
            simplex_idx_ruta = tri_ruta.find_simplex(np.c_[grid_x.ravel(), grid_y.ravel()])
            dentro = (simplex_idx_ruta != -1).reshape(grid_x.shape)
            #temporalddddddddddddddddddddddddddddddddddd
            # DIAGNÓSTICO TEMPORAL: visualizar qué tanto del terreno se
            # considera realmente "navegable" (dentro=True) vs inválido.
            st.write(f"🔍 Porcentaje de terreno considerado válido para la ruta: {100 * dentro.mean():.1f}%")
            fig_diag = go.Figure(data=go.Heatmap(z=dentro.astype(int), colorscale='Greys'))
            fig_diag.update_layout(title="Zona válida (blanco) vs inválida (negro) para la ruta", height=500)
            st.plotly_chart(fig_diag, use_container_width=True)
#777777777
            grid_z_relleno = np.where(np.isnan(grid_z_ruta), np.nanmin(grid_z_ruta), grid_z_ruta)
            dzdx, dzdy = np.gradient(grid_z_relleno, rango_x / max(nx - 1, 1), rango_y / max(ny - 1, 1))
            pendiente_local = np.sqrt(dzdx ** 2 + dzdy ** 2)
#777777777777
            GRADO_OBJETIVO = 0.035
            costo = 1.0 + 300.0 * np.clip(pendiente_local - GRADO_OBJETIVO, 0, None) ** 2.0

            # ALEJAR DEL BORDE: se calcula qué tan lejos está cada celda del
            # límite del terreno levantado (borde del TIN). Cuanto más cerca
            # del borde/cresta, más caro es pasar por ahí — así el algoritmo
            # prefiere ir por el interior del terreno en vez de pegarse al
            # filo, aunque el filo tenga menos pendiente.
            dist_borde_px = distance_transform_edt(dentro)
            dist_borde_m = dist_borde_px * resolucion_ruta
            PESO_BORDE = 5.0  # antes 60: ahora solo un empujón suave hacia el centro, no lo domina todo
            # RELATIVO AL ANCHO LOCAL (no global): se compara cada celda
            # contra el punto más "centrado" dentro de su propia vecindad
            # (radio ~ 40 celdas), no contra el ancho máximo de TODO el
            # terreno. Así, en zonas angostas donde el ancho máximo local es
            # pequeño, esa zona igual tiene su propio "centro relativo" bien
            # marcado, en vez de quedar toda pareja como pasaba antes al
            # normalizar contra el máximo global (que venía de otra parte
            # más ancha del terreno).
            radio_vecindad = 40
            max_local_borde = maximum_filter(dist_borde_m, size=radio_vecindad)
            max_local_borde = np.maximum(max_local_borde, 1e-6)
            factor_cercania = np.clip(1.0 - (dist_borde_m / max_local_borde), 0, 1)
            costo += PESO_BORDE * (np.exp(factor_cercania * 4) - 1)

            # Zona fuera del levantamiento: prácticamente intransitable (ya
            # no conviene, porque preferimos que se aleje del borde HACIA
            # ADENTRO, no que se salga del terreno).
            costo[~dentro] = 1e6

            costo = np.nan_to_num(costo, nan=1e6, posinf=1e6)
#777777777777777

            def celda_mas_cercana(punto):
                d2 = (grid_x - punto[0]) ** 2 + (grid_y - punto[1]) ** 2
                d2 = np.where(dentro, d2, np.inf)
                return np.unravel_index(np.argmin(d2), grid_z_ruta.shape)


            celda_inicio = celda_mas_cercana(p_inicio)
            celda_fin = celda_mas_cercana(p_final)

            indices_ruta, _ = route_through_array(costo, celda_inicio, celda_fin,
                                                  fully_connected=True, geometric=True)
            filas_r, cols_r = np.array(indices_ruta).T
            x_ruta, y_ruta = grid_x[filas_r, cols_r].astype(float), grid_y[filas_r, cols_r].astype(float)
#777777777777777
            if len(x_ruta) >= 5:
                x_suave = gaussian_filter1d(x_ruta, sigma=1.2)
                y_suave = gaussian_filter1d(y_ruta, sigma=1.2)

                # PROTECCIÓN: el suavizado puede "cortar camino" en curvas
                # cerradas y sacar la línea del área válida (terreno sólido).
                # Se verifica cada punto suavizado contra la máscara `dentro`;
                # si cae fuera, se usa el punto SIN suavizar en su lugar (ese
                # sí viene garantizado dentro, porque route_through_array
                # solo camina por celdas válidas).
                col_idx = np.clip(((x_suave - x_min) / max(rango_x, 1e-6) * (nx - 1)).astype(int), 0, nx - 1)
                fil_idx = np.clip(((y_suave - y_min) / max(rango_y, 1e-6) * (ny - 1)).astype(int), 0, ny - 1)
                fuera_de_mascara = ~dentro[col_idx, fil_idx]

                x_ruta = np.where(fuera_de_mascara, x_ruta, x_suave)
                y_ruta = np.where(fuera_de_mascara, y_ruta, y_suave)
                #7777777777
            x_ruta[0], y_ruta[0] = p_inicio
            x_ruta[-1], y_ruta[-1] = p_final

            d_ruta = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x_ruta), np.diff(y_ruta)))])
            long_ruta = d_ruta[-1] if d_ruta[-1] > 0 else longitud_linea
            t_destino = np.linspace(0, long_ruta, 500)
            x_eje = np.interp(t_destino, d_ruta, x_ruta)
            y_eje = np.interp(t_destino, d_ruta, y_ruta)

            z_terreno = griddata(puntos_2d, df['Z'].values, (x_eje, y_eje), method='linear')
            mask_nan = np.isnan(z_terreno)
            if mask_nan.any():
                z_terreno_nn = griddata(puntos_2d, df['Z'].values, (x_eje, y_eje), method='nearest')
                z_terreno[mask_nan] = z_terreno_nn[mask_nan]

        st.success(f"✅ Ruta calculada sobre el terreno: {long_ruta:.2f} m "
                   f"(línea recta: {longitud_linea:.2f} m).")

        z_diseno = np.zeros_like(x_eje)
        z_diseno[0] = z_terreno[0]

        dx_arr = np.sqrt(np.diff(x_eje) ** 2 + np.diff(y_eje) ** 2)
        dx_arr = np.insert(dx_arr, 0, dx_arr[0])

        dist = 0
        for i in range(1, len(x_eje)):
            dx = dx_arr[i]
            dist += dx
            idx_p = min(int(dist // 100), 4)
            z_diseno[i] = z_diseno[i - 1] + (st.session_state.pendientes[idx_p] / 100) * dx

        st.session_state.x_eje, st.session_state.y_eje = x_eje, y_eje
        st.session_state.z_terreno_eje, st.session_state.z_diseno_eje = z_terreno, z_diseno
        st.session_state.dx_arr = dx_arr

    if 'z_diseno_eje' in st.session_state:
        x_eje, y_eje = st.session_state.x_eje, st.session_state.y_eje
        z_terreno, z_diseno = st.session_state.z_terreno_eje, st.session_state.z_diseno_eje

        dist_acum = np.zeros(len(x_eje))
        dist_acum[1:] = np.cumsum(np.sqrt(np.diff(x_eje) ** 2 + np.diff(y_eje) ** 2))

        grid_x, grid_y = np.mgrid[x_min:x_max:complex(0, 50), y_min:y_max:complex(0, 50)]
        grid_z_contexto = griddata(df[['X', 'Y']].values, df['Z'].values, (grid_x, grid_y), method='linear')

        tri = Delaunay(df[['X', 'Y']].values)
        simplex_indices = tri.find_simplex(np.c_[grid_x.ravel(), grid_y.ravel()])
        grid_z_contexto[(simplex_indices == -1).reshape(grid_x.shape)] = np.nan

        fig7 = go.Figure()

        fig7.add_trace(go.Surface(x=grid_x[:, 0], y=grid_y[0, :], z=grid_z_contexto.T, colorscale='Greens', opacity=0.3,
                                  showscale=False))
        fig7.add_trace(
            go.Scatter3d(x=x_eje, y=y_eje, z=z_terreno, mode='lines', line=dict(color='yellow', width=3, dash='dot'),
                         name="Terreno Natural (Z)"))
        fig7.add_trace(go.Scatter3d(x=[x_eje[0]], y=[y_eje[0]], z=[z_terreno[0]], mode='markers',
                                    marker=dict(symbol='diamond', color='magenta', size=8), name="Estaca 0+000"))
        fig7.add_trace(go.Scatter3d(x=[x_eje[-1]], y=[y_eje[-1]], z=[z_terreno[-1]], mode='markers',
                                    marker=dict(symbol='x', color='red', size=12), name="Llegada Meta"))

        colores_tramos = ['red', 'lime', 'cyan', 'magenta', 'orange']
        for i in range(5):
            mask = (dist_acum >= i * 100) & (dist_acum <= (i + 1) * 100)
            idx = np.where(mask)[0]
            if len(idx) > 0:
                if idx[-1] < len(x_eje) - 1:
                    idx = np.append(idx, idx[-1] + 1)
                fig7.add_trace(go.Scatter3d(
                    x=x_eje[idx], y=y_eje[idx], z=z_diseno[idx],
                    mode='lines', line=dict(color=colores_tramos[i], width=8),
                    name=f"Tramo K0+{i * 100} ({st.session_state.pendientes[i]}%)"
                ))

        fig7.update_layout(template="plotly_dark", height=700, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig7, use_container_width=True)

# ------------------------------------------
# FASE 8  (CORREGIDA)
# ------------------------------------------
elif opcion == "8. Maqueta de Excavación 3D":
    st.header("Fase 8: Corte, Relleno y Renderizado Final")
    df = st.session_state.df

    if 'z_diseno_eje' not in st.session_state:
        st.warning("⚠️ Completa las Fases 6 y 7 primero.")
    else:
        # FIX: el "Recorte de bordes" ya no arranca fijo en 50 m. Se calibra
        # según la densidad real de puntos del levantamiento (mediana de la
        # distancia al vecino más cercano x4). Con un valor fijo, un archivo
        # con huecos más grandes que 50 m entre puntos fragmentaba la
        # superficie en pedazos sueltos (justo lo que salió en tu captura).
        _pts_xy = df[['X', 'Y']].values
        _tree_densidad = cKDTree(_pts_xy)
        _dist_vecino, _ = _tree_densidad.query(_pts_xy, k=2)
        _sugerido_max_len = float(np.clip(np.median(_dist_vecino[:, 1]) * 7, 5.0, 150.0))

        # Añadimos los mismos controles de la Fase 5 para mantener la consistencia
        st.info(f"⚙️ Parámetros de renderizado volumétrico (Recorte sugerido para este archivo: {_sugerido_max_len:.1f} m)")
        c1, c2, c3 = st.columns(3)
        with c1:
            resolucion = st.slider("Resolución (m):", 1, 10, 2, key="res8")
        with c2:
            distancia_max = st.slider("Recorte de bordes (TIN max len):", 5.0, 150.0, _sugerido_max_len, key="dist8")
        with c3:
            profundidad = st.slider("Profundidad base (m):", 5, 100, 10, key="prof8")

        if st.button("🚜 Renderizar Maqueta con Freno de Presupuesto", use_container_width=True):
            st.progress(100, text="¡COMPLETADO!")
            x_eje, y_eje = st.session_state.x_eje, st.session_state.y_eje
            z_terreno, z_diseno = st.session_state.z_terreno_eje, st.session_state.z_diseno_eje
            dx_arr = st.session_state.dx_arr
            W = st.session_state.ancho_via
            V_max = st.session_state.presupuesto_tierra

            vol_corte, vol_relleno = 0, 0
            idx_freno = len(x_eje) - 1
            distancia_lograda = 0

            # Cálculo volumétrico: metro a metro hasta agotar presupuesto
            for i in range(1, len(x_eje)):
                if np.isnan(z_terreno[i]): continue
                dz = z_terreno[i] - z_diseno[i]
                vol_tramo = abs(dz) * W * dx_arr[i]

                if dz > 0:
                    vol_corte += vol_tramo
                else:
                    vol_relleno += vol_tramo

                if (vol_corte + vol_relleno) >= V_max:
                    idx_freno = i
                    break
                distancia_lograda += dx_arr[i]

            st.session_state.resultados = {
                "longitud": distancia_lograda, "corte": vol_corte, "relleno": vol_relleno,
                "x_plot": x_eje[:idx_freno], "y_plot": y_eje[:idx_freno],
                "z_terreno_plot": z_terreno[:idx_freno], "z_diseno_plot": z_diseno[:idx_freno],
                "idx_freno": idx_freno
            }

        if 'resultados' in st.session_state:
            res = st.session_state.resultados

            # FIX: mensaje claro de "SÍ ALCANZA / NO ALCANZA" según qué tan lejos
            # llegó la excavación respecto al trazado completo.
            llego_completo = res["idx_freno"] >= (len(st.session_state.x_eje) - 2)
            if not llego_completo:
                st.error(
                    f"🛑 NO, EL PRESUPUESTO NO ES SUFICIENTE: Con {st.session_state.presupuesto_tierra:,.2f} m³, "
                    f"la excavación solo pudo avanzar hasta la abscisa K0+{res['longitud']:.2f} m "
                    f"(de {np.sum(st.session_state.dx_arr):.2f} m totales)."
                )
            else:
                st.success(
                    f"✅ SÍ, EL PRESUPUESTO ALCANZA: la vía se completó exitosamente hasta la Meta "
                    f"(K0+{res['longitud']:.2f} m)."
                )

            colm1, colm2, colm3 = st.columns(3)
            colm1.metric("Volumen de Corte", f"{res['corte']:.2f} m³")
            colm2.metric("Volumen de Relleno", f"{res['relleno']:.2f} m³")
            colm3.metric("Presupuesto Usado", f"{(res['corte'] + res['relleno']):,.0f} / {st.session_state.presupuesto_tierra:,.0f} m³")

            x_min, x_max, y_min, y_max = df['X'].min(), df['X'].max(), df['Y'].min(), df['Y'].max()
            rango_x, rango_y = x_max - x_min, y_max - y_min
            cota_base = np.floor(df['Z'].min()) - profundidad

            grid_x, grid_y = np.mgrid[
                x_min:x_max:complex(0, rango_x / resolucion), y_min:y_max:complex(0, rango_y / resolucion)]
            puntos_2d = df[['X', 'Y']].values

            grid_z_top = griddata(puntos_2d, df['Z'].values, (grid_x, grid_y), method='linear')
            grid_z_top = np.where(np.isnan(grid_z_top),
                                  griddata(puntos_2d, df['Z'].values, (grid_x, grid_y), method='nearest'), grid_z_top)

            # Filtro Matemático TIN
            tri = Delaunay(puntos_2d)
            p = puntos_2d[tri.simplices]
            max_len = np.max([np.linalg.norm(p[:, 0] - p[:, 1], axis=1),
                              np.linalg.norm(p[:, 1] - p[:, 2], axis=1),
                              np.linalg.norm(p[:, 2] - p[:, 0], axis=1)], axis=0)

            simplex_indices = tri.find_simplex(np.c_[grid_x.ravel(), grid_y.ravel()])
            es_valido = np.zeros(len(simplex_indices), dtype=bool)
            adentro = simplex_indices != -1
            es_valido[adentro] = max_len[simplex_indices[adentro]] <= distancia_max
            es_valido = es_valido.reshape(grid_x.shape)

            # LIMPIEZA FUERTE: un kernel más grande de cierre elimina las
            # rayas/huecos sueltos que aparecen en el interior del bloque
            # (triángulos delgados del TIN que rechazan celdas aisladas).
            # binary_fill_holes remata rellenando cualquier hueco que haya
            # quedado totalmente rodeado de terreno válido, dejando la
            # ladera como un bloque sólido continuo, sin rayas.
            es_valido = binary_closing(es_valido, structure=np.ones((15, 15)))
            es_valido = binary_fill_holes(es_valido)
            es_valido = binary_opening(es_valido, structure=np.ones((5, 5)))

            grid_z_top[~es_valido] = np.nan

            # FIX: en vez de recortar (NaN) la superficie donde pasa la vía —lo
            # que dejaba un hueco negro—, ahora la "hundimos" suavemente hacia
            # la cota de diseño (rasante), con un talud (pendiente de corte o
            # relleno) que se abre más donde el corte es más profundo, tal
            # como un corte real de carretera. Solo aplica sobre el tramo
            # realmente construido con el presupuesto.
            if len(res["x_plot"]) >= 2:
                tree_via = cKDTree(np.c_[res["x_plot"], res["y_plot"]])
                dist_via, idx_via = tree_via.query(np.c_[grid_x.ravel(), grid_y.ravel()])
                dist_via = dist_via.reshape(grid_x.shape)
                z_diseno_cercano = res["z_diseno_plot"][idx_via].reshape(grid_x.shape)

                mitad_via = st.session_state.ancho_via / 2

                # Talud angosto: el corte llega SIEMPRE hasta la cota real de
                # diseño (sin topes artificiales), así la carretera queda
                # visible al fondo de la excavación. La apertura del talud se
                # mantiene contenida para que no se vea como un cráter.
                talud_ratio = 0.5
                tope_talud_extra = 6.0
                ancho_talud = mitad_via + np.minimum(
                    np.abs(grid_z_top - z_diseno_cercano) * talud_ratio, tope_talud_extra
                )

                f = np.clip((dist_via - mitad_via) / np.maximum(ancho_talud - mitad_via, 0.5), 0, 1)
                f_suave = f * f * (3 - 2 * f)

                grid_z_top = np.where(
                    np.isnan(grid_z_top), grid_z_top,
                    z_diseno_cercano * (1 - f_suave) + grid_z_top * f_suave
                )

                # Suavizado "consciente de los huecos" (convolución
                # normalizada): promedia SOLO entre celdas válidas del TIN,
                # nunca contra relleno artificial. Esto deja el terreno sólido
                # y sin los dientes que salían antes en el borde del TIN.
                mascara_valida = ~np.isnan(grid_z_top)
                V = np.where(mascara_valida, grid_z_top, 0.0)
                W = mascara_valida.astype(float)
                VV = gaussian_filter(V, sigma=1.0)
                WW = gaussian_filter(W, sigma=1.0)
                grid_z_suave = np.divide(VV, WW, out=np.full_like(VV, np.nan), where=WW > 1e-6)
                grid_z_top = np.where(mascara_valida, grid_z_suave, np.nan)

                # Suavizado 2D final: redondea el borde del corte para que la
                # ladera "abrace" la vía en vez de mostrar un tajo brusco
                # (respeta los huecos NaN del recorte TIN).
                mascara_nan = np.isnan(grid_z_top)
                grid_z_relleno_tmp = np.where(mascara_nan, np.nanmin(grid_z_top), grid_z_top)
                grid_z_suave = gaussian_filter(grid_z_relleno_tmp, sigma=1.0)
                grid_z_top = np.where(mascara_nan, np.nan, grid_z_suave)

            # Base plana conectada correctamente
            grid_z_base = np.full_like(grid_z_top, cota_base)
            grid_z_base[np.isnan(grid_z_top)] = np.nan

            # Trazado completo (referencia), igual que en la Fase 7
            x_eje_full = st.session_state.x_eje
            y_eje_full = st.session_state.y_eje
            z_terreno_full = st.session_state.z_terreno_eje

            fig8 = go.Figure()

            # Terreno con sombreado tipo relieve (look más 3D / realista)
            fig8.add_trace(go.Surface(
                x=grid_x[:, 0], y=grid_y[0, :], z=grid_z_top.T, colorscale='Earth', name="Terreno",
                contours=dict(z=dict(show=True, usecolormap=True, project_z=True, size=2)),
                lighting=dict(ambient=0.55, diffuse=0.85, roughness=0.6, specular=0.25, fresnel=0.15),
                lightposition=dict(x=100, y=200, z=1000)
            ))
            fig8.add_trace(go.Surface(x=grid_x[:, 0], y=grid_y[0, :], z=grid_z_base.T, colorscale='Greys',
                                      showscale=False, opacity=1.0, hoverinfo='skip', name="Base"))

            # FIX: cinta de asfalto con el ancho REAL de la vía, apoyada justo
            # sobre la rasante hundida — así se ve como una carretera dentro
            # de la zanja, no como una línea flotando encima del terreno.
            xp, yp, zp = res["x_plot"], res["y_plot"], res["z_diseno_plot"]
            if len(xp) >= 2:
                dxp, dyp = np.gradient(xp), np.gradient(yp)
                norma = np.sqrt(dxp ** 2 + dyp ** 2)
                norma[norma == 0] = 1e-6
                perp_x, perp_y = -dyp / norma, dxp / norma
                mitad_via = st.session_state.ancho_via / 2

                x_asfalto = np.stack([xp - perp_x * mitad_via, xp + perp_x * mitad_via], axis=1)
                y_asfalto = np.stack([yp - perp_y * mitad_via, yp + perp_y * mitad_via], axis=1)
                z_asfalto = np.stack([zp, zp], axis=1) + 0.15  # leve offset anti z-fighting

                fig8.add_trace(go.Surface(
                    x=x_asfalto, y=y_asfalto, z=z_asfalto, name="Carretera (asfalto)",
                    colorscale=[[0, '#4a4a4a'], [1, '#4a4a4a']], showscale=False,
                    lighting=dict(ambient=0.8, diffuse=0.5, specular=0.6, roughness=0.35),
                    hoverinfo='skip'
                ))

            # FIX: agregar la línea punteada del terreno natural a lo largo de
            # TODO el trazado (para ver hasta dónde faltaba llegar).
            fig8.add_trace(go.Scatter3d(x=x_eje_full, y=y_eje_full, z=z_terreno_full, mode='lines',
                                        line=dict(color='yellow', width=3, dash='dot'),
                                        name="Terreno Natural (Z)"))

            # FIX: rombo de la Estaca 0+000
            fig8.add_trace(go.Scatter3d(x=[x_eje_full[0]], y=[y_eje_full[0]], z=[z_terreno_full[0]],
                                        mode='markers', marker=dict(symbol='diamond', color='magenta', size=8),
                                        name="Estaca 0+000"))

            # FIX: X roja en el punto de Meta (el destino objetivo, se haya
            # alcanzado o no con el presupuesto disponible).
            fig8.add_trace(go.Scatter3d(x=[x_eje_full[-1]], y=[y_eje_full[-1]], z=[z_terreno_full[-1]],
                                        mode='markers', marker=dict(symbol='x', color='red', size=12),
                                        name="Llegada Meta"))

            # Línea central de la vía (marca de pavimento), apoyada sobre el
            # asfalto en vez de flotar por encima del terreno
            fig8.add_trace(go.Scatter3d(x=xp, y=yp, z=zp + 0.25, mode='lines',
                                        line=dict(color='white', width=4, dash='dash'),
                                        name="Eje Construido"))

            fig8.update_layout(template="plotly_dark", height=800, margin=dict(l=0, r=0, b=0, t=0),
                               scene=dict(
                                   aspectratio=dict(x=rango_x / max(rango_x, rango_y),
                                                    y=rango_y / max(rango_x, rango_y), z=0.4),
                                   zaxis=dict(range=[cota_base - 2, np.nanmax(grid_z_top) + 5]),
                                   camera=dict(eye=dict(x=1.4, y=1.4, z=0.9))
                               ))

            st.plotly_chart(fig8, use_container_width=True)

# ------------------------------------------
# FASE 9
# ------------------------------------------
elif opcion == "9. Base de Datos (Archivero)":
    st.header("Fase 9: Archivero de Diseños (SQLite3)")
    st.info("Guarda el historial de tus cálculos volumétricos para compararlos.")

    conn = sqlite3.connect('registro_vial.db')
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS proyectos
           (
               id
               INTEGER
               PRIMARY
               KEY
               AUTOINCREMENT,
               nombre
               TEXT,
               ancho
               REAL,
               presupuesto
               REAL,
               longitud_lograda
               REAL,
               corte
               REAL,
               relleno
               REAL
           )''')
    conn.commit()

    nombre_ruta = st.text_input("Ingresa un nombre para guardar esta simulación:")
    if st.button("💾 Guardar iteración actual en la Base de Datos"):
        if 'resultados' in st.session_state and nombre_ruta:
            res = st.session_state.resultados
            c.execute(
                "INSERT INTO proyectos (nombre, ancho, presupuesto, longitud_lograda, corte, relleno) VALUES (?, ?, ?, ?, ?, ?)",
                (nombre_ruta, st.session_state.ancho_via, st.session_state.presupuesto_tierra, res["longitud"],
                 res["corte"], res["relleno"]))
            conn.commit()
            st.success(f"¡Iteración '{nombre_ruta}' guardada con éxito!")
        else:
            st.warning("Completa la Fase 8 y escribe un nombre.")

    df_db = pd.read_sql_query("SELECT * FROM proyectos", conn)
    if not df_db.empty:
        st.dataframe(df_db, use_container_width=True)
    conn.close()

# ------------------------------------------
# FASE 10
# ------------------------------------------
elif opcion == "10. Emisión de Memoria (PDF)":
    st.header("Fase 10: Memoria de Cálculo Legal (fpdf2)")
    st.info("Adjunta la captura fotográfica del modelo 3D (Botón de cámara en Plotly) y emite el PDF formal.")

    nombre_proyecto = st.text_input("Nombre del Proyecto:", value="memoria 19_06")
    imagen_3d = st.file_uploader("Sube la captura de pantalla de tu Maqueta (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if st.button("🖨️ Generar Memoria de Cálculo PDF") and 'resultados' in st.session_state and imagen_3d:
        res = st.session_state.resultados

        fig, ax = plt.subplots(figsize=(10, 4))
        distancias = np.insert(np.cumsum(st.session_state.dx_arr[:res["idx_freno"] - 1]), 0, 0)

        ax.plot(distancias, res["z_terreno_plot"], label="Terreno Natural", color="brown")
        ax.plot(distancias, res["z_diseno_plot"], label="Rasante Variable", color="darkblue", linestyle="--")
        ax.fill_between(distancias, res["z_terreno_plot"], res["z_diseno_plot"],
                        where=(res["z_terreno_plot"] > res["z_diseno_plot"]), color="lightcoral", label="Corte",
                        alpha=0.5)
        ax.fill_between(distancias, res["z_terreno_plot"], res["z_diseno_plot"],
                        where=(res["z_terreno_plot"] < res["z_diseno_plot"]), color="lightblue", label="Relleno",
                        alpha=0.5)
        ax.set_title(f"Perfil Longitudinal - Avance Logrado: {res['longitud']:.2f}m")
        ax.set_xlabel("Abscisa (m)")
        ax.set_ylabel("Elevación Z (m)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        fig.savefig("temp_plot.png", bbox_inches='tight')

        with open("temp_3d.png", "wb") as f:
            f.write(imagen_3d.getbuffer())

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "MEMORIA TÉCNICA: MOVIMIENTO DE TIERRAS", align="C", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", size=11)
        intro_text = (
            "1. INTRODUCCIÓN\n"
            "El presente informe detalla los cálculos volumétricos para la construcción del proyecto vial. "
            "Se procesó una nube de puntos para crear un Modelo Digital de Elevaciones (MDE), "
            "sobre el cual se intersectó la rasante de diseño para evaluar la viabilidad geológica y financiera."
        )
        pdf.multi_cell(0, 6, txt=intro_text)
        pdf.ln(5)

        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "2. PARÁMETROS Y RESULTADOS:", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, f"   - Ancho de Calzada de Diseño: {st.session_state.ancho_via} metros", ln=True)
        pdf.cell(0, 8, f"   - Tope Presupuestario (Volumen Máx): {st.session_state.presupuesto_tierra} m3", ln=True)
        pdf.cell(0, 8, f"   - Volumen Excavado (Corte): {res['corte']:.2f} m3", ln=True)
        pdf.cell(0, 8, f"   - Volumen Terraplén (Relleno): {res['relleno']:.2f} m3", ln=True)
        pdf.cell(0, 8, f"   - Longitud Vial Construida: {res['longitud']:.2f} metros", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", "B", 11)
        if res["idx_freno"] < (len(st.session_state.x_eje) - 2):
            msg = "3. CONCLUSIÓN OFICIAL: EL PRESUPUESTO NO ALCANZA. La obra debe detenerse prematuramente debido a la falta de capacidad volumétrica financiada para completar los cortes/rellenos requeridos."
            pdf.set_text_color(200, 0, 0)
        else:
            msg = "3. CONCLUSIÓN OFICIAL: EL PRESUPUESTO SÍ ALCANZA. Los cálculos demuestran que el volumen financiado es suficiente para cubrir las necesidades topográficas de todo el trazado."
            pdf.set_text_color(0, 150, 0)

        pdf.multi_cell(0, 6, txt=msg)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(10)

        pdf.image("temp_plot.png", w=170)
        pdf.add_page()
        pdf.cell(0, 10, "ANEXO 1: Renderizado Tridimensional de Excavación", ln=True)
        pdf.image("temp_3d.png", w=170)

        try:
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
        except (AttributeError, TypeError):
            pdf_bytes = bytes(pdf.output())

        os.remove("temp_plot.png")
        os.remove("temp_3d.png")

        st.success("📄 ¡Informe de Ingeniería Generado!")
        st.download_button("⬇️ Descargar Informe.pdf", data=pdf_bytes, file_name=f"{nombre_proyecto}.pdf",
                           mime="application/pdf")