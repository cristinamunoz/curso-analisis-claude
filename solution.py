"""Análisis de diversidad alfa del microbioma de suelo de Atacama.

Reproduce la primera pieza del análisis de Neilson et al. (2017):
¿la aridez del suelo reduce la diversidad microbiana?

Hipótesis (H1): a menor humedad relativa del suelo (AvgSoilRH),
menor diversidad alfa (índice de Shannon).

Pasos del script:
  1. Cargar la metadata y la tabla de abundancias de OTUs.
  2. Cruzar ambas por las muestras que están en las dos tablas.
  3. Calcular el índice de Shannon para cada muestra.
  4. Resumir Shannon por transecto (Baquedano vs. Yungay).
  5. Regresión lineal de Shannon frente a la humedad del suelo.
  6. Guardar tablas y figuras en la carpeta outputs/.

Uso:
    python solution.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # No abrir ventanas; solo guardar archivos.
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, linregress, spearmanr

# Semilla fija para que las permutaciones den siempre el mismo
# resultado (análisis reproducible).
SEMILLA = 42

# --- Rutas de archivos -------------------------------------------
RUTA_METADATA = os.path.join("data", "metadata.tsv")
RUTA_ABUNDANCIAS = os.path.join("data", "abundancias.tsv")
CARPETA_SALIDA = "outputs"

# Nombres exactos de las columnas que usamos de la metadata.
COL_HUMEDAD = "average-soil-relative-humidity"
COL_TRANSECTO = "transect-name"

# Colores daltónicos para los dos transectos (paleta Set2).
COLORES_TRANSECTO = {
    "Baquedano": "#66c2a5",  # transecto menos árido
    "Yungay": "#fc8d62",     # transecto más árido
}


def cargar_y_cruzar_datos():
    """Carga las dos tablas y las cruza por muestras comunes.

    Devuelve un DataFrame con una fila por muestra y estas
    columnas: el índice de Shannon (se añade después), el
    transecto y la humedad del suelo. Solo incluye las muestras
    que aparecen en las dos tablas.
    """
    # Metadata: una fila por muestra; usamos sample-id como índice.
    metadata = pd.read_csv(RUTA_METADATA, sep="\t", index_col=0)

    # Abundancias: una fila por OTU, una columna por muestra.
    # La transponemos para tener muestras en filas y OTUs en
    # columnas, que es más cómodo para calcular por muestra.
    abundancias = pd.read_csv(RUTA_ABUNDANCIAS, sep="\t", index_col=0)
    abundancias = abundancias.transpose()

    # Nos quedamos solo con las muestras presentes en ambas tablas.
    muestras_comunes = metadata.index.intersection(abundancias.index)
    metadata = metadata.loc[muestras_comunes]
    abundancias = abundancias.loc[muestras_comunes]

    print(f"Muestras en metadata:      {len(metadata)}")
    print(f"Muestras con abundancias:  {abundancias.shape[0]}")
    print(f"Muestras cruzadas (comun): {len(muestras_comunes)}")
    print()
    print("Muestras por transecto:")
    print(metadata[COL_TRANSECTO].value_counts().to_string())
    print()

    return metadata, abundancias


def shannon_por_muestra(abundancias):
    """Calcula el índice de Shannon para cada muestra.

    El índice de Shannon (H) resume la diversidad de una comunidad:
    combina cuántos tipos de microbio (OTUs) hay y cuán equilibradas
    son sus abundancias. Un valor alto = comunidad diversa y
    equilibrada; un valor bajo = pocos tipos o uno dominante.

    Fórmula: H = -suma(p_i * ln(p_i)), donde p_i es la proporción
    del OTU i dentro de la muestra. Usamos scipy.stats.entropy, que
    convierte los conteos a proporciones automáticamente.
    """
    valores = abundancias.apply(
        lambda fila: entropy(fila.values, base=np.e), axis=1
    )
    valores.name = "shannon"
    return valores


def construir_tabla_diversidad(metadata, shannon):
    """Une el índice de Shannon con transecto y humedad del suelo."""
    tabla = pd.DataFrame(
        {
            "shannon": shannon,
            "transecto": metadata[COL_TRANSECTO],
            "humedad": metadata[COL_HUMEDAD],
        }
    )
    # Descartamos muestras sin dato de humedad (no sirven para la
    # regresión).
    tabla = tabla.dropna(subset=["humedad"])
    return tabla


def resumir_por_transecto(tabla, ruta_salida):
    """Media, mediana y rango de Shannon por transecto."""
    resumen = tabla.groupby("transecto")["shannon"].agg(
        n="count",
        media="mean",
        mediana="median",
        minimo="min",
        maximo="max",
    )
    resumen = resumen.round(3)

    print("Resumen del índice de Shannon por transecto:")
    print(resumen.to_string())
    print()

    resumen.to_csv(ruta_salida, sep="\t")
    print(f"Tabla guardada en: {ruta_salida}")
    print()
    return resumen


def figura_boxplot(tabla, ruta_png, ruta_pdf):
    """Boxplot de Shannon por transecto (Baquedano vs. Yungay)."""
    orden = ["Baquedano", "Yungay"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=tabla,
        x="transecto",
        y="shannon",
        order=orden,
        hue="transecto",
        palette=COLORES_TRANSECTO,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=tabla,
        x="transecto",
        y="shannon",
        order=orden,
        color="black",
        size=4,
        alpha=0.5,
        ax=ax,
    )
    ax.set_title(
        "Diversidad alfa (Shannon) por transecto\n"
        "Baquedano (menos árido) vs. Yungay (más árido)"
    )
    ax.set_xlabel("Transecto")
    ax.set_ylabel("Índice de Shannon (H, log natural)")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=300)
    fig.savefig(ruta_pdf)
    plt.close(fig)
    print(f"Figura guardada en: {ruta_png}")
    print(f"Figura guardada en: {ruta_pdf}")
    print()


def regresion_lineal(tabla, ruta_salida):
    """Regresión lineal de Shannon frente a la humedad del suelo.

    Devuelve el resultado de scipy.stats.linregress para poder
    dibujar la recta después.
    """
    x = tabla["humedad"].values
    y = tabla["shannon"].values
    n = len(x)

    # Regresión lineal por mínimos cuadrados (lo que pediste).
    reg = linregress(x, y)
    r2 = reg.rvalue ** 2

    # Correlación de Spearman: el método exacto del paper de
    # Neilson (no asume que la relación sea una línea recta).
    rho, p_spearman = spearmanr(x, y)

    print("Regresión lineal: Shannon ~ humedad del suelo (AvgSoilRH)")
    print(f"  n muestras     = {n}")
    print(f"  pendiente      = {reg.slope:.4f}")
    print(f"  intercepto     = {reg.intercept:.4f}")
    print(f"  R2 (lineal)    = {r2:.4f}")
    print(f"  p-valor        = {reg.pvalue:.4g}")
    print()
    print("Referencia (método del paper): correlación de Spearman")
    print(f"  rho de Spearman = {rho:.4f}")
    print(f"  p-valor         = {p_spearman:.4g}")
    print()

    # Aviso de la regla de CLAUDE.md: R2 muy bajo puede indicar un
    # problema con los datos o que no hay relación lineal.
    if r2 < 0.05:
        print("AVISO: R2 < 0.05. La humedad explica muy poca")
        print("variación de la diversidad de forma lineal. Conviene")
        print("revisar si la relación no es lineal o si hay ruido en")
        print("los datos antes de sacar conclusiones.")
        print()

    resultado = pd.DataFrame(
        {
            "metrica": [
                "n",
                "pendiente",
                "intercepto",
                "R2_lineal",
                "p_valor_lineal",
                "rho_spearman",
                "p_valor_spearman",
            ],
            "valor": [
                n,
                reg.slope,
                reg.intercept,
                r2,
                reg.pvalue,
                rho,
                p_spearman,
            ],
        }
    )
    resultado.to_csv(ruta_salida, sep="\t", index=False)
    print(f"Tabla guardada en: {ruta_salida}")
    print()
    return reg, r2, rho, p_spearman


def guardar_tabla_diversidad_alfa(
    metadata, shannon, reg, r2, rho, p_spearman, ruta_salida
):
    """Tabla por muestra del Shannon + estadísticos de la regresión.

    Guarda una fila por muestra (Shannon, transecto, AvgSoilRH) y, en
    líneas de cabecera comentadas al inicio (empiezan por #), los
    resultados de la regresión de Shannon frente a AvgSoilRH. Así el
    archivo reúne los datos por muestra y el resumen estadístico, y
    sigue siendo fácil de leer (con pandas: comment='#').
    """
    tabla = pd.DataFrame(
        {
            "sample_id": shannon.index,
            "transecto": metadata.loc[shannon.index, COL_TRANSECTO].values,
            "AvgSoilRH": metadata.loc[shannon.index, COL_HUMEDAD].values,
            "shannon": shannon.round(4).values,
        }
    )
    n_regresion = int(
        metadata.loc[shannon.index, COL_HUMEDAD].notna().sum()
    )

    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        archivo.write(
            "# Diversidad alfa (indice de Shannon) por muestra\n"
        )
        archivo.write(
            "# Estadisticos de la regresion Shannon ~ AvgSoilRH:\n"
        )
        archivo.write(f"# n = {n_regresion}\n")
        archivo.write(f"# pendiente = {reg.slope:.4f}\n")
        archivo.write(f"# intercepto = {reg.intercept:.4f}\n")
        archivo.write(f"# R2_lineal = {r2:.4f}\n")
        archivo.write(f"# p_valor_lineal = {reg.pvalue:.4g}\n")
        archivo.write(f"# rho_spearman = {rho:.4f}\n")
        archivo.write(f"# p_valor_spearman = {p_spearman:.4g}\n")
        tabla.to_csv(archivo, sep="\t", index=False)

    print(f"Tabla guardada en: {ruta_salida}")
    print()


def figura_regresion(tabla, reg, r2, ruta_png, ruta_pdf):
    """Scatter Shannon vs. humedad con recta y banda de confianza."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Recta de regresión con banda de confianza al 95 % (seaborn
    # la calcula por remuestreo). scatter=False para dibujar los
    # puntos aparte y poder colorearlos por transecto.
    sns.regplot(
        data=tabla,
        x="humedad",
        y="shannon",
        scatter=False,
        color="gray",
        line_kws={"linewidth": 2},
        ax=ax,
    )
    sns.scatterplot(
        data=tabla,
        x="humedad",
        y="shannon",
        hue="transecto",
        palette=COLORES_TRANSECTO,
        s=50,
        ax=ax,
    )

    # Mostrar R2 y p-valor dentro del gráfico.
    texto = f"R² = {r2:.3f}\np = {reg.pvalue:.3g}\nn = {len(tabla)}"
    ax.text(
        0.03,
        0.97,
        texto,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    ax.set_title(
        "Diversidad alfa (Shannon) frente a la humedad del suelo"
    )
    ax.set_xlabel("Humedad relativa del suelo, AvgSoilRH (%)")
    ax.set_ylabel("Índice de Shannon (H, log natural)")
    ax.legend(title="Transecto")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=300)
    fig.savefig(ruta_pdf)
    plt.close(fig)
    print(f"Figura guardada en: {ruta_png}")
    print(f"Figura guardada en: {ruta_pdf}")
    print()


def figura_regresion_viridis(tabla, reg, r2, ruta_png, ruta_pdf):
    """Igual que figura_regresion, pero coloreando los puntos por el
    gradiente continuo de humedad con la paleta viridis.

    Sigue la regla de la skill: viridis para variables continuas.
    Aquí el color de cada punto codifica su humedad del suelo.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Recta de regresión con banda de confianza al 95 %.
    sns.regplot(
        data=tabla,
        x="humedad",
        y="shannon",
        scatter=False,
        color="gray",
        line_kws={"linewidth": 2},
        ax=ax,
    )
    # Puntos coloreados por humedad (gradiente continuo, viridis).
    dispersion = ax.scatter(
        tabla["humedad"],
        tabla["shannon"],
        c=tabla["humedad"],
        cmap="viridis",
        s=50,
        edgecolor="black",
        linewidth=0.3,
    )
    barra = fig.colorbar(dispersion, ax=ax)
    barra.set_label("Humedad relativa del suelo, AvgSoilRH (%)")

    texto = f"R² = {r2:.3f}\np = {reg.pvalue:.3g}\nn = {len(tabla)}"
    ax.text(
        0.03,
        0.97,
        texto,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    ax.set_title(
        "Diversidad alfa (Shannon) frente a la humedad del suelo"
    )
    ax.set_xlabel("Humedad relativa del suelo, AvgSoilRH (%)")
    ax.set_ylabel("Índice de Shannon (H, log natural)")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=300)
    fig.savefig(ruta_pdf)
    plt.close(fig)
    print(f"Figura guardada en: {ruta_png}")
    print(f"Figura guardada en: {ruta_pdf}")
    print()


# =================================================================
# DIVERSIDAD BETA: composición entre muestras (H2 y H3)
# =================================================================


def filtrar_otus_frecuentes(abundancias, min_prevalencia=0.10):
    """Descarta los OTUs raros (presentes en pocas muestras).

    Muchos OTUs aparecen en solo 1 o 2 muestras y aportan sobre todo
    ruido. Nos quedamos con los OTUs presentes (conteo > 0) en al
    menos una fracción `min_prevalencia` de las muestras. Esto suele
    reforzar la señal ambiental sin cambiar la biología de fondo.

    `abundancias` tiene muestras en filas y OTUs en columnas.
    """
    n_muestras = abundancias.shape[0]
    presencia = (abundancias > 0).sum(axis=0)  # en cuántas muestras
    umbral = min_prevalencia * n_muestras
    frecuentes = presencia[presencia >= umbral].index
    filtradas = abundancias[frecuentes]

    print(
        f"Filtro de OTUs: {abundancias.shape[1]} -> "
        f"{filtradas.shape[1]} OTUs "
        f"(presentes en >= {min_prevalencia:.0%} de las muestras, "
        f"es decir >= {int(np.ceil(umbral))} de {n_muestras})."
    )
    print()
    return filtradas


def matriz_bray_curtis(abundancias):
    """Calcula las distancias de Bray-Curtis entre todas las muestras.

    Primero convierte los conteos de cada muestra a proporciones
    (abundancia relativa), para que las muestras con más lecturas no
    pesen artificialmente más. La distancia de Bray-Curtis va de 0
    (dos muestras con la misma composición) a 1 (sin OTUs en común).

    Devuelve un DataFrame cuadrado (muestras x muestras).
    """
    # Abundancia relativa: cada fila (muestra) suma 1.
    totales = abundancias.sum(axis=1)
    relativas = abundancias.div(totales, axis=0)

    distancias = squareform(pdist(relativas.values, metric="braycurtis"))
    return pd.DataFrame(
        distancias, index=abundancias.index, columns=abundancias.index
    )


def pcoa(distancias):
    """Ordenación PCoA (Análisis de Coordenadas Principales).

    Resume la matriz de distancias en unos pocos ejes, colocando las
    muestras en un espacio donde las distancias reflejan lo parecidas
    o distintas que son sus comunidades. Es el equivalente para
    distancias de un análisis de componentes principales.

    Devuelve:
      - coordenadas: DataFrame (muestras x ejes PC1, PC2, ...).
      - varianza_pct: % de varianza que explica cada eje.
    """
    dist = distancias.values
    n = dist.shape[0]

    # Doble centrado de Gower sobre -1/2 * (distancia al cuadrado).
    a = -0.5 * dist ** 2
    centrado = np.eye(n) - np.ones((n, n)) / n
    gower = centrado @ a @ centrado

    # Autovalores/autovectores (la matriz es simétrica).
    autovalores, autovectores = np.linalg.eigh(gower)
    orden = np.argsort(autovalores)[::-1]
    autovalores = autovalores[orden]
    autovectores = autovectores[:, orden]

    # Solo los ejes con autovalor positivo tienen sentido geométrico.
    positivos = autovalores > 0
    coords = autovectores[:, positivos] * np.sqrt(autovalores[positivos])

    varianza_pct = (
        autovalores[positivos] / autovalores[positivos].sum() * 100
    )

    nombres = [f"PC{i + 1}" for i in range(coords.shape[1])]
    coordenadas = pd.DataFrame(
        coords, index=distancias.index, columns=nombres
    )
    return coordenadas, varianza_pct


def guardar_varianza_pcoa(varianza_pct, ruta_salida, n_ejes=5):
    """Guarda el % de varianza explicada por los primeros ejes."""
    n_ejes = min(n_ejes, len(varianza_pct))
    tabla = pd.DataFrame(
        {
            "eje": [f"PC{i + 1}" for i in range(n_ejes)],
            "varianza_explicada_pct": np.round(
                varianza_pct[:n_ejes], 2
            ),
        }
    )
    print("Varianza explicada por la PCoA (primeros ejes):")
    print(tabla.to_string(index=False))
    print()
    tabla.to_csv(ruta_salida, sep="\t", index=False)
    print(f"Tabla guardada en: {ruta_salida}")
    print()
    return tabla


def permanova(distancias, predictor, permutaciones=999, semilla=SEMILLA):
    """PERMANOVA / db-RDA univariada para un predictor.

    Mide qué parte de la variación en composición (distancias
    Bray-Curtis) explica una variable ambiental. Funciona igual para
    una variable continua (p. ej. humedad) o de grupo (transecto,
    codificado como 0/1).

    Método estándar basado en distancias (McArdle & Anderson, 2001):
    se reparte la variación total en la parte explicada por el
    predictor y el resto, y se compara con 999 reordenamientos al
    azar para obtener el p-valor.

    Devuelve un diccionario con F (pseudo-F), R2 y p-valor.
    """
    dist = distancias.values
    n = dist.shape[0]

    # Matriz de Gower (variación total en composición).
    a = -0.5 * dist ** 2
    centrado = np.eye(n) - np.ones((n, n)) / n
    gower = centrado @ a @ centrado
    sc_total = np.trace(gower)

    # Matriz de diseño: intercepto + predictor.
    x = np.column_stack([np.ones(n), np.asarray(predictor, dtype=float)])

    def pseudo_f(g):
        hat = x @ np.linalg.inv(x.T @ x) @ x.T
        sc_modelo = np.trace(hat @ g @ hat)
        sc_error = np.trace((np.eye(n) - hat) @ g @ (np.eye(n) - hat))
        gl_modelo = x.shape[1] - 1
        gl_error = n - x.shape[1]
        f = (sc_modelo / gl_modelo) / (sc_error / gl_error)
        r2 = sc_modelo / sc_total
        return f, r2

    f_obs, r2_obs = pseudo_f(gower)

    # Prueba por permutaciones: reordenamos las muestras al azar.
    rng = np.random.default_rng(semilla)
    conteo = 0
    for _ in range(permutaciones):
        perm = rng.permutation(n)
        g_perm = gower[np.ix_(perm, perm)]
        f_perm, _ = pseudo_f(g_perm)
        if f_perm >= f_obs:
            conteo += 1
    p_valor = (conteo + 1) / (permutaciones + 1)

    return {"F": f_obs, "R2": r2_obs, "p_valor": p_valor}


def permanova_variables_ambientales(distancias, metadata, ruta_salida):
    """PERMANOVA univariada para cada variable ambiental (H3).

    Usa los mismos casos completos (muestras con las tres variables)
    para que los R² sean comparables entre variables.
    """
    variables = {
        "AvgSoilRH": COL_HUMEDAD,
        "temperatura": "average-soil-temperature",
        "elevacion": "elevation",
    }
    columnas = list(variables.values())
    completos = metadata.dropna(subset=columnas)
    ids = completos.index

    dist_sub = distancias.loc[ids, ids]

    filas = []
    for nombre, columna in variables.items():
        resultado = permanova(dist_sub, completos[columna].values)
        filas.append(
            {
                "variable": nombre,
                "F": round(resultado["F"], 3),
                "R2": round(resultado["R2"], 3),
                "p_valor": round(resultado["p_valor"], 4),
            }
        )

    tabla = pd.DataFrame(filas)
    # Ordenar de mayor a menor R² (regla de CLAUDE.md).
    tabla = tabla.sort_values("R2", ascending=False).reset_index(
        drop=True
    )

    print(f"PERMANOVA univariada (n = {len(ids)}, 999 permutaciones):")
    print(tabla.to_string(index=False))
    print()
    tabla.to_csv(ruta_salida, sep="\t", index=False)
    print(f"Tabla guardada en: {ruta_salida}")
    print()
    return tabla


def permanova_transecto(distancias, metadata):
    """PERMANOVA de Baquedano vs. Yungay para apoyar H2."""
    transecto = metadata[COL_TRANSECTO]
    # Codificar el grupo como 0/1.
    codigo = (transecto == "Yungay").astype(int).values
    resultado = permanova(distancias, codigo)
    print("PERMANOVA por transecto (Baquedano vs. Yungay):")
    print(f"  F       = {resultado['F']:.3f}")
    print(f"  R2      = {resultado['R2']:.3f}")
    print(f"  p-valor = {resultado['p_valor']:.4f}")
    print()
    return resultado


def figura_pcoa(coordenadas, varianza_pct, metadata, ruta_png, ruta_pdf):
    """Figura de la ordenación PCoA.

    - Color de cada punto = humedad del suelo (gradiente viridis).
    - Forma del punto = transecto (círculo Baquedano, triángulo
      Yungay), para ver la separación entre sitios (H2).
    """
    datos = coordenadas.join(
        metadata[[COL_TRANSECTO, COL_HUMEDAD]]
    )
    marcadores = {"Baquedano": "o", "Yungay": "^"}

    fig, ax = plt.subplots(figsize=(8, 6))
    norma = plt.Normalize(
        vmin=datos[COL_HUMEDAD].min(), vmax=datos[COL_HUMEDAD].max()
    )

    for transecto, marca in marcadores.items():
        sub = datos[datos[COL_TRANSECTO] == transecto]
        con_dato = sub[sub[COL_HUMEDAD].notna()]
        sin_dato = sub[sub[COL_HUMEDAD].isna()]
        ax.scatter(
            con_dato["PC1"],
            con_dato["PC2"],
            c=con_dato[COL_HUMEDAD],
            cmap="viridis",
            norm=norma,
            marker=marca,
            s=70,
            edgecolor="black",
            linewidth=0.4,
        )
        # Muestras sin dato de humedad: en gris, sin perderlas.
        if not sin_dato.empty:
            ax.scatter(
                sin_dato["PC1"],
                sin_dato["PC2"],
                color="lightgray",
                marker=marca,
                s=70,
                edgecolor="black",
                linewidth=0.4,
            )

    # Barra de color para la humedad.
    mapa = plt.cm.ScalarMappable(cmap="viridis", norm=norma)
    mapa.set_array([])
    barra = fig.colorbar(mapa, ax=ax)
    barra.set_label("Humedad relativa del suelo, AvgSoilRH (%)")

    # Leyenda de formas (transecto), en gris para no competir con el
    # color de la humedad.
    handles = [
        Line2D(
            [0],
            [0],
            marker=marca,
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=9,
            label=transecto,
        )
        for transecto, marca in marcadores.items()
    ]
    ax.legend(handles=handles, title="Transecto", loc="best")

    ax.set_title(
        "PCoA de composición microbiana (distancias Bray-Curtis)"
    )
    ax.set_xlabel(f"PC1 ({varianza_pct[0]:.1f} % de la varianza)")
    ax.set_ylabel(f"PC2 ({varianza_pct[1]:.1f} % de la varianza)")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=300)
    fig.savefig(ruta_pdf)
    plt.close(fig)
    print(f"Figura guardada en: {ruta_png}")
    print(f"Figura guardada en: {ruta_pdf}")
    print()


def analisis_beta(metadata, abundancias):
    """Ejecuta todo el análisis de diversidad beta (H2 y H3)."""
    print("=" * 60)
    print("DIVERSIDAD BETA: composición entre muestras")
    print("=" * 60)
    print()

    # Quitar OTUs raros para reforzar la señal ambiental.
    abundancias = filtrar_otus_frecuentes(abundancias, min_prevalencia=0.10)
    distancias = matriz_bray_curtis(abundancias)
    coordenadas, varianza_pct = pcoa(distancias)

    guardar_varianza_pcoa(
        varianza_pct,
        os.path.join(CARPETA_SALIDA, "h2_varianza_explicada_pcoa.tsv"),
    )
    figura_pcoa(
        coordenadas,
        varianza_pct,
        metadata,
        os.path.join(CARPETA_SALIDA, "fig2_pcoa_braycurtis.png"),
        os.path.join(CARPETA_SALIDA, "fig2_pcoa_braycurtis.pdf"),
    )
    permanova_variables_ambientales(
        distancias,
        metadata,
        os.path.join(
            CARPETA_SALIDA, "h3_permanova_variables_ambientales.tsv"
        ),
    )
    permanova_transecto(distancias, metadata)


def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    metadata, abundancias = cargar_y_cruzar_datos()
    shannon = shannon_por_muestra(abundancias)
    tabla = construir_tabla_diversidad(metadata, shannon)

    resumir_por_transecto(
        tabla,
        os.path.join(
            CARPETA_SALIDA, "h1_resumen_shannon_por_transecto.tsv"
        ),
    )
    figura_boxplot(
        tabla,
        os.path.join(CARPETA_SALIDA, "fig1_shannon_por_transecto.png"),
        os.path.join(CARPETA_SALIDA, "fig1_shannon_por_transecto.pdf"),
    )
    reg, r2, rho, p_spearman = regresion_lineal(
        tabla,
        os.path.join(
            CARPETA_SALIDA, "h1_regresion_shannon_vs_humedad.tsv"
        ),
    )
    guardar_tabla_diversidad_alfa(
        metadata,
        shannon,
        reg,
        r2,
        rho,
        p_spearman,
        os.path.join(CARPETA_SALIDA, "h1_diversidad_alfa.tsv"),
    )
    figura_regresion(
        tabla,
        reg,
        r2,
        os.path.join(CARPETA_SALIDA, "fig1_shannon_vs_avgsoilrh.png"),
        os.path.join(CARPETA_SALIDA, "fig1_shannon_vs_avgsoilrh.pdf"),
    )
    figura_regresion_viridis(
        tabla,
        reg,
        r2,
        os.path.join(
            CARPETA_SALIDA, "fig1_shannon_vs_avgsoilrh_viridis.png"
        ),
        os.path.join(
            CARPETA_SALIDA, "fig1_shannon_vs_avgsoilrh_viridis.pdf"
        ),
    )

    # --- Diversidad beta (H2 y H3) ---------------------------------
    print()
    analisis_beta(metadata, abundancias)

    print("Análisis terminado. Revisa la carpeta outputs/.")


if __name__ == "__main__":
    main()
