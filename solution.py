"""Diversidad y composicion microbiana vs. aridez del suelo
(desierto de Atacama).

Pregunta biologica: a medida que el suelo se vuelve mas arido,
como cambian la diversidad y la composicion de la comunidad
microbiana?

Hipotesis H1: a menor humedad relativa del suelo (AvgSoilRH),
menor diversidad alfa (Shannon).

Hipotesis H2: los sitios mas aridos (Yungay) se separan de los
menos aridos (Baquedano) en la ordenacion PCoA a lo largo del
primer eje.

Hipotesis H3: AvgSoilRH explica mas varianza composicional que
temperatura o elevacion.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

RUTA_METADATA = "data/metadata.tsv"
RUTA_ABUNDANCIAS = "data/abundancias.tsv"
CARPETA_SALIDA = "outputs"

COLUMNA_HUMEDAD = "average-soil-relative-humidity"
COLUMNA_TRANSECTO = "transect-name"
COLUMNA_TEMPERATURA = "average-soil-temperature"
COLUMNA_ELEVACION = "elevation"

N_PERMUTACIONES = 999
SEMILLA_PERMUTACIONES = 0


def cargar_datos():
    """Carga metadata y abundancias, y cruza por sample-id comun."""
    metadata = pd.read_csv(RUTA_METADATA, sep="\t")
    abundancias = pd.read_csv(RUTA_ABUNDANCIAS, sep="\t")
    abundancias = abundancias.set_index("#OTU ID")

    ids_metadata = set(metadata["sample-id"])
    ids_abundancias = set(abundancias.columns)
    ids_comunes = sorted(ids_metadata & ids_abundancias)

    print(
        f"Muestras en metadata: {len(ids_metadata)} | "
        f"muestras en abundancias: {len(ids_abundancias)} | "
        f"muestras en comun (usadas en el analisis): "
        f"{len(ids_comunes)}"
    )

    abundancias = abundancias[ids_comunes]
    metadata = metadata[
        metadata["sample-id"].isin(ids_comunes)
    ].set_index("sample-id")
    metadata = metadata.loc[ids_comunes]

    return metadata, abundancias


def calcular_shannon(abundancias):
    """Calcula el indice de Shannon (base e) para cada muestra.

    abundancias: OTUs en filas, muestras en columnas.
    """
    conteos = abundancias.to_numpy(dtype=float)
    total_por_muestra = conteos.sum(axis=0)
    proporciones = conteos / total_por_muestra

    with np.errstate(divide="ignore", invalid="ignore"):
        aporte = np.where(
            proporciones > 0,
            proporciones * np.log(proporciones),
            0.0,
        )
    shannon = -aporte.sum(axis=0)
    return pd.Series(
        shannon, index=abundancias.columns, name="shannon"
    )


def resumen_por_transecto(datos):
    """Imprime media, mediana y rango de Shannon por transecto."""
    resumen = datos.groupby(COLUMNA_TRANSECTO)["shannon"].agg(
        media="mean",
        mediana="median",
        minimo="min",
        maximo="max",
        n="count",
    )
    print("\nResumen de diversidad de Shannon por transecto:")
    print(resumen.round(3))
    return resumen


def graficar_boxplot_transecto(datos):
    """Boxplot de Shannon por transecto (Baquedano vs. Yungay)."""
    transectos = sorted(datos[COLUMNA_TRANSECTO].unique())
    colores = plt.get_cmap("Set2").colors

    fig, ax = plt.subplots(figsize=(8, 6))
    grupos = [
        datos.loc[
            datos[COLUMNA_TRANSECTO] == transecto, "shannon"
        ]
        for transecto in transectos
    ]
    cajas = ax.boxplot(
        grupos, tick_labels=transectos, patch_artist=True
    )
    for caja, color in zip(cajas["boxes"], colores):
        caja.set_facecolor(color)

    ax.set_xlabel("Transecto")
    ax.set_ylabel("Indice de diversidad de Shannon")
    ax.set_title(
        "Diversidad de Shannon por transecto\n"
        "(Baquedano: mas humedo | Yungay: mas arido)"
    )
    fig.tight_layout()

    fig.savefig(
        f"{CARPETA_SALIDA}/fig_shannon_por_transecto.png",
        dpi=300,
    )
    fig.savefig(
        f"{CARPETA_SALIDA}/fig_shannon_por_transecto.pdf"
    )
    plt.close(fig)


def regresion_shannon_vs_humedad(datos):
    """Regresion Shannon ~ humedad relativa del suelo.

    Devuelve un diccionario con rho de Spearman, R2, p-valor y n
    de la regresion lineal.
    """
    completos = datos.dropna(subset=[COLUMNA_HUMEDAD, "shannon"])
    excluidas = len(datos) - len(completos)
    if excluidas:
        print(
            f"\nAVISO: se excluyeron {excluidas} muestra(s) sin "
            f"dato de {COLUMNA_HUMEDAD} (valor faltante)."
        )

    x = completos[COLUMNA_HUMEDAD].to_numpy(dtype=float)
    y = completos["shannon"].to_numpy(dtype=float)
    n = len(x)

    rho, p_spearman = stats.spearmanr(x, y)
    reg = stats.linregress(x, y)
    r2 = reg.rvalue ** 2

    resultado = {
        "n": n,
        "spearman_rho": rho,
        "spearman_p_valor": p_spearman,
        "pearson_r2": r2,
        "regresion_p_valor": reg.pvalue,
        "pendiente": reg.slope,
        "intercepto": reg.intercept,
    }

    print("\nRegresion Shannon vs. humedad relativa del suelo:")
    for clave, valor in resultado.items():
        print(f"  {clave}: {valor}")

    if r2 < 0.05:
        print(
            "\nAVISO: R2 < 0.05. La relacion lineal es muy debil "
            "-- revisar si esto tiene sentido biologico antes de "
            "continuar."
        )

    return resultado, reg


def graficar_regresion(datos, reg):
    """Scatter Shannon vs. humedad con recta e IC 95%."""
    completos = datos.dropna(subset=[COLUMNA_HUMEDAD, "shannon"])
    x = completos[COLUMNA_HUMEDAD].to_numpy(dtype=float)
    y = completos["shannon"].to_numpy(dtype=float)
    n = len(x)

    x_linea = np.linspace(x.min(), x.max(), 200)
    y_linea = reg.intercept + reg.slope * x_linea

    residuales = y - (reg.intercept + reg.slope * x)
    gl = n - 2
    error_estandar_residual = np.sqrt(
        np.sum(residuales ** 2) / gl
    )
    x_media = x.mean()
    ssx = np.sum((x - x_media) ** 2)
    error_estandar_media = error_estandar_residual * np.sqrt(
        1 / n + (x_linea - x_media) ** 2 / ssx
    )
    valor_t = stats.t.ppf(0.975, gl)
    banda = valor_t * error_estandar_media

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, color="#4c72b0", alpha=0.8, label="Muestras")
    ax.plot(
        x_linea, y_linea, color="#c44e52", label="Regresion lineal"
    )
    ax.fill_between(
        x_linea,
        y_linea - banda,
        y_linea + banda,
        color="#c44e52",
        alpha=0.2,
        label="IC 95%",
    )

    r2 = reg.rvalue ** 2
    texto = (
        f"R2 = {r2:.3f}\n"
        f"p = {reg.pvalue:.3g}\n"
        f"n = {n}"
    )
    ax.text(
        0.05,
        0.95,
        texto,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.8
        ),
    )

    ax.set_xlabel("Humedad relativa promedio del suelo (%)")
    ax.set_ylabel("Indice de diversidad de Shannon")
    ax.set_title(
        "Diversidad de Shannon vs. humedad relativa del suelo"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()

    fig.savefig(
        f"{CARPETA_SALIDA}/fig_shannon_vs_avgsoilrh.png",
        dpi=300,
    )
    fig.savefig(
        f"{CARPETA_SALIDA}/fig_shannon_vs_avgsoilrh.pdf"
    )
    plt.close(fig)


def guardar_tabla_resultado(resultado):
    """Guarda los resultados de la regresion en un TSV."""
    tabla = pd.DataFrame([resultado])
    ruta = (
        f"{CARPETA_SALIDA}/"
        "h1_correlacion_shannon_vs_humedad.tsv"
    )
    tabla.to_csv(ruta, sep="\t", index=False)
    print(f"\nTabla de resultados guardada en: {ruta}")


def calcular_bray_curtis(abundancias):
    """Matriz de distancias Bray-Curtis entre muestras.

    abundancias: OTUs en filas, muestras en columnas.
    """
    conteos = abundancias.T.to_numpy(dtype=float)
    distancias = squareform(pdist(conteos, metric="braycurtis"))
    return pd.DataFrame(
        distancias,
        index=abundancias.columns,
        columns=abundancias.columns,
    )


def _matriz_gower(distancias):
    """Centrado de Gower de la matriz de distancias al cuadrado.

    Es la base tanto del PCoA como del PERMANOVA (McArdle &
    Anderson, 2001): ambos parten de la misma matriz G.
    """
    d2 = distancias.to_numpy() ** 2
    n = d2.shape[0]
    centrado = np.eye(n) - np.ones((n, n)) / n
    return centrado @ d2 @ centrado * -0.5


def calcular_pcoa(distancias):
    """PCoA por descomposicion espectral de la matriz de Gower.

    Devuelve las coordenadas de PC1/PC2 por muestra y el
    porcentaje de varianza que explica cada eje (sobre la suma
    de autovalores positivos, convencion habitual cuando la
    distancia no es estrictamente euclidiana).
    """
    g = _matriz_gower(distancias)
    autovalores, autovectores = np.linalg.eigh(g)

    orden = np.argsort(autovalores)[::-1]
    autovalores = autovalores[orden]
    autovectores = autovectores[:, orden]

    autovalores_pos = np.clip(autovalores, 0, None)
    coordenadas = autovectores * np.sqrt(autovalores_pos)
    varianza_total = autovalores_pos.sum()
    porcentaje = autovalores_pos / varianza_total * 100

    tabla_coordenadas = pd.DataFrame(
        coordenadas[:, :2],
        index=distancias.index,
        columns=["PC1", "PC2"],
    )
    return tabla_coordenadas, porcentaje[:2]


def graficar_pcoa(coordenadas, humedad, porcentaje):
    """PCoA coloreado por gradiente continuo de humedad (viridis)."""
    humedad = humedad.reindex(coordenadas.index)
    con_dato = humedad.notna()

    fig, ax = plt.subplots(figsize=(8, 6))
    dispersión = ax.scatter(
        coordenadas.loc[con_dato, "PC1"],
        coordenadas.loc[con_dato, "PC2"],
        c=humedad[con_dato],
        cmap="viridis",
        s=70,
        edgecolor="black",
        linewidth=0.3,
    )
    if (~con_dato).any():
        ax.scatter(
            coordenadas.loc[~con_dato, "PC1"],
            coordenadas.loc[~con_dato, "PC2"],
            color="lightgray",
            edgecolor="black",
            linewidth=0.3,
            s=70,
            label="Sin dato de humedad",
        )
        ax.legend(loc="best")

    barra_color = fig.colorbar(dispersión, ax=ax)
    barra_color.set_label("Humedad relativa promedio del suelo (%)")

    ax.set_xlabel(f"PC1 ({porcentaje[0]:.1f}% de varianza)")
    ax.set_ylabel(f"PC2 ({porcentaje[1]:.1f}% de varianza)")
    ax.set_title(
        "PCoA (Bray-Curtis) de la composicion microbiana del suelo"
    )
    fig.tight_layout()

    fig.savefig(
        f"{CARPETA_SALIDA}/fig2_pcoa_braycurtis.png", dpi=300
    )
    fig.savefig(f"{CARPETA_SALIDA}/fig2_pcoa_braycurtis.pdf")
    plt.close(fig)


def guardar_tabla_pcoa(porcentaje):
    """Guarda el porcentaje de varianza explicada por PC1/PC2."""
    tabla = pd.DataFrame(
        {
            "eje": ["PC1", "PC2"],
            "porcentaje_varianza": porcentaje,
        }
    )
    ruta = f"{CARPETA_SALIDA}/h2_varianza_explicada_pcoa.tsv"
    tabla.to_csv(ruta, sep="\t", index=False)
    print(f"\nTabla de varianza del PCoA guardada en: {ruta}")


def permanova_variable(distancias, variable, nombre, columna):
    """PERMANOVA univariada de composicion ~ una variable continua.

    Metodo de McArdle & Anderson (2001): parte de la misma
    matriz de Gower que el PCoA y evalua la significancia del F
    observado por permutacion de la variable (999 veces, por
    defecto). Las muestras sin dato en `variable` se excluyen
    solo para esta variable.
    """
    valido = variable.notna()
    ids = variable.index[valido]
    excluidas = (~valido).sum()
    if excluidas:
        print(
            f"AVISO: {nombre} -- se excluyeron {excluidas} "
            f"muestra(s) sin dato de {columna}."
        )

    sub_distancias = distancias.loc[ids, ids]
    x = variable.loc[ids].to_numpy(dtype=float)
    n = len(ids)

    g = _matriz_gower(sub_distancias)
    ss_total = np.trace(g)

    def f_y_r2(x_actual):
        diseño = np.column_stack([np.ones(n), x_actual])
        proyeccion = diseño @ np.linalg.pinv(
            diseño.T @ diseño
        ) @ diseño.T
        ss_modelo = np.trace(proyeccion @ g)
        ss_residual = ss_total - ss_modelo
        gl_residual = n - 2
        f_obs = (ss_modelo / 1) / (ss_residual / gl_residual)
        r2 = ss_modelo / ss_total
        return f_obs, r2

    f_obs, r2_obs = f_y_r2(x)

    generador = np.random.default_rng(SEMILLA_PERMUTACIONES)
    extremos = 0
    for _ in range(N_PERMUTACIONES):
        x_permutado = generador.permutation(x)
        f_permutado, _ = f_y_r2(x_permutado)
        if f_permutado >= f_obs:
            extremos += 1
    p_valor = (extremos + 1) / (N_PERMUTACIONES + 1)

    return {
        "variable": nombre,
        "columna_original": columna,
        "F": f_obs,
        "R2": r2_obs,
        "p_valor": p_valor,
        "n_permutaciones": N_PERMUTACIONES,
        "n": n,
    }


def modelar_composicion_vs_ambiente(datos, distancias):
    """PERMANOVA de composicion ~ humedad, temperatura, elevacion.

    Reporta F, R2, p-valor y n por variable, ordenado de mayor a
    menor R2 (regla del curso).
    """
    variables = [
        ("humedad_relativa", COLUMNA_HUMEDAD),
        ("temperatura", COLUMNA_TEMPERATURA),
        ("elevacion", COLUMNA_ELEVACION),
    ]
    resultados = [
        permanova_variable(
            distancias, datos[columna], nombre, columna
        )
        for nombre, columna in variables
    ]
    tabla = pd.DataFrame(resultados).sort_values(
        "R2", ascending=False
    )

    print(
        "\nPERMANOVA univariada -- composicion ~ variable "
        "ambiental (999 permutaciones):"
    )
    print(tabla.round(4).to_string(index=False))

    for _, fila in tabla.iterrows():
        if fila["n"] < 10:
            print(
                f"\nAVISO: {fila['variable']} tiene n={fila['n']}"
                " (< 10) -- las permutaciones pueden ser "
                "insuficientes."
            )

    ruta = (
        f"{CARPETA_SALIDA}/"
        "h3_permanova_variables_ambientales.tsv"
    )
    tabla.to_csv(ruta, sep="\t", index=False)
    print(f"\nTabla de PERMANOVA guardada en: {ruta}")
    return tabla


def main():
    metadata, abundancias = cargar_datos()
    shannon = calcular_shannon(abundancias)

    datos = metadata.copy()
    datos["shannon"] = shannon

    resumen_por_transecto(datos)
    graficar_boxplot_transecto(datos)

    resultado, reg = regresion_shannon_vs_humedad(datos)
    graficar_regresion(datos, reg)
    guardar_tabla_resultado(resultado)

    distancias = calcular_bray_curtis(abundancias)
    coordenadas, porcentaje = calcular_pcoa(distancias)
    graficar_pcoa(coordenadas, datos[COLUMNA_HUMEDAD], porcentaje)
    guardar_tabla_pcoa(porcentaje)

    modelar_composicion_vs_ambiente(datos, distancias)


if __name__ == "__main__":
    main()
