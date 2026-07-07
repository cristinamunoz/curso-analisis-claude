"""
Analisis H1: diversidad alfa (Shannon) vs humedad relativa
del suelo (AvgSoilRH), siguiendo a Neilson et al. (2017).

Hipotesis: a menor humedad relativa del suelo, menor
diversidad alfa (Shannon).
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

RUTA_ABUNDANCIAS = "data/abundancias.tsv"
RUTA_METADATA = "data/metadata.tsv"
CARPETA_SALIDA = "outputs"

COLOR_TRANSECTO = {
    "Baquedano": "#66c2a5",  # Set2
    "Yungay": "#fc8d62",     # Set2
}


def calcular_shannon(columna_abundancias):
    """Calcula el indice de Shannon de una muestra (una columna
    de conteos de OTUs)."""
    total = columna_abundancias.sum()
    proporciones = columna_abundancias[columna_abundancias > 0]
    proporciones = proporciones / total
    return -(proporciones * np.log(proporciones)).sum()


def cargar_datos():
    """Carga abundancias y metadata, y cruza por sample-id
    usando solo las muestras presentes en ambos archivos."""
    abundancias = pd.read_csv(RUTA_ABUNDANCIAS, sep="\t")
    metadata = pd.read_csv(RUTA_METADATA, sep="\t")

    columnas_muestra = abundancias.columns[1:]
    muestras_comunes = sorted(
        set(columnas_muestra) & set(metadata["sample-id"])
    )

    shannon_por_muestra = {
        muestra: calcular_shannon(abundancias[muestra])
        for muestra in muestras_comunes
    }
    tabla_shannon = pd.DataFrame({
        "sample-id": list(shannon_por_muestra.keys()),
        "shannon": list(shannon_por_muestra.values()),
    })

    datos = tabla_shannon.merge(metadata, on="sample-id", how="left")
    datos = datos.rename(
        columns={"average-soil-relative-humidity": "AvgSoilRH"}
    )
    return datos[["sample-id", "transect-name", "shannon", "AvgSoilRH"]]


def resumen_por_transecto(datos):
    """Resumen estadistico de Shannon por transecto."""
    resumen = datos.groupby("transect-name")["shannon"].agg(
        mean="mean", median="median", min="min", max="max",
        count="count",
    ).reset_index()
    return resumen


def graficar_boxplot(datos, resumen):
    """Boxplot de Shannon por transecto (Baquedano vs Yungay)."""
    transectos = resumen["transect-name"].tolist()
    grupos = [
        datos.loc[datos["transect-name"] == t, "shannon"]
        for t in transectos
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    cajas = ax.boxplot(
        grupos, tick_labels=transectos, patch_artist=True
    )
    for caja, transecto in zip(cajas["boxes"], transectos):
        caja.set_facecolor(COLOR_TRANSECTO[transecto])

    ax.set_title(
        "Diversidad de Shannon por transecto (desierto de Atacama)"
    )
    ax.set_xlabel("Transecto")
    ax.set_ylabel("Indice de Shannon")
    fig.tight_layout()

    fig.savefig(f"{CARPETA_SALIDA}/fig1_shannon_por_transecto.png",
                dpi=300)
    fig.savefig(f"{CARPETA_SALIDA}/fig1_shannon_por_transecto.pdf")
    plt.close(fig)


def correlacion_spearman(datos):
    """Correlacion de Spearman entre Shannon y AvgSoilRH,
    descartando muestras sin dato de humedad."""
    datos_validos = datos.dropna(subset=["AvgSoilRH"])
    rho, p_valor = stats.spearmanr(
        datos_validos["AvgSoilRH"], datos_validos["shannon"]
    )
    n = len(datos_validos)
    r2 = rho ** 2

    tabla = pd.DataFrame({
        "variable_x": ["average-soil-relative-humidity"],
        "variable_y": ["shannon"],
        "spearman_rho": [rho],
        "r2": [r2],
        "p_valor": [p_valor],
        "n": [n],
    })
    return datos_validos, tabla


def graficar_dispersion_con_regresion(datos_validos, tabla):
    """Scatter Shannon vs AvgSoilRH con recta de regresion OLS,
    banda de confianza 95% y anotacion de R2 y p-valor."""
    x = datos_validos["AvgSoilRH"].to_numpy()
    y = datos_validos["shannon"].to_numpy()
    n = len(x)

    pendiente, intercepto = np.polyfit(x, y, 1)
    y_pred = pendiente * x + intercepto
    residuos = y - y_pred
    error_estandar = np.sqrt(np.sum(residuos ** 2) / (n - 2))
    x_media = x.mean()
    suma_cuadrados_x = np.sum((x - x_media) ** 2)

    x_linea = np.linspace(x.min(), x.max(), 100)
    y_linea = pendiente * x_linea + intercepto
    error_pred = error_estandar * np.sqrt(
        1 / n + (x_linea - x_media) ** 2 / suma_cuadrados_x
    )
    t_critico = stats.t.ppf(0.975, df=n - 2)
    banda_superior = y_linea + t_critico * error_pred
    banda_inferior = y_linea - t_critico * error_pred

    rho = tabla.loc[0, "spearman_rho"]
    r2 = tabla.loc[0, "r2"]
    p_valor = tabla.loc[0, "p_valor"]

    fig, ax = plt.subplots(figsize=(8, 6))
    colores = [
        COLOR_TRANSECTO[t]
        for t in datos_validos["transect-name"]
    ]
    ax.scatter(x, y, c=colores, edgecolor="black", zorder=3)
    ax.plot(x_linea, y_linea, color="black", zorder=2)
    ax.fill_between(
        x_linea, banda_inferior, banda_superior,
        color="gray", alpha=0.3, zorder=1,
        label="Banda de confianza 95%",
    )

    texto = (
        f"Spearman rho = {rho:.2f}\n"
        f"R2 = {r2:.2f}\n"
        f"p-valor = {p_valor:.4f}\n"
        f"n = {n}"
    )
    ax.text(
        0.05, 0.95, texto, transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white",
              "alpha": 0.8},
    )

    ax.set_title(
        "Diversidad de Shannon vs. humedad relativa del suelo"
    )
    ax.set_xlabel("Humedad relativa promedio del suelo (%)")
    ax.set_ylabel("Indice de Shannon")

    manejadores = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=color, markeredgecolor="black",
            label=transecto,
        )
        for transecto, color in COLOR_TRANSECTO.items()
    ]
    ax.legend(handles=manejadores, title="Transecto")
    fig.tight_layout()

    fig.savefig(f"{CARPETA_SALIDA}/fig1_shannon_vs_avgsoilrh.png",
                dpi=300)
    fig.savefig(f"{CARPETA_SALIDA}/fig1_shannon_vs_avgsoilrh.pdf")
    plt.close(fig)


def cargar_abundancias_comunes():
    """Carga la tabla de abundancias restringida a las muestras
    presentes tambien en metadata.tsv."""
    abundancias = pd.read_csv(RUTA_ABUNDANCIAS, sep="\t")
    metadata = pd.read_csv(RUTA_METADATA, sep="\t")

    columnas_muestra = abundancias.columns[1:]
    muestras_comunes = sorted(
        set(columnas_muestra) & set(metadata["sample-id"])
    )
    abundancias_comunes = abundancias[muestras_comunes]
    return abundancias_comunes, muestras_comunes, metadata


def calcular_distancias_bray_curtis(abundancias_comunes):
    """Calcula la matriz cuadrada de distancias Bray-Curtis entre
    muestras a partir de la tabla de abundancias."""
    matriz_muestras = abundancias_comunes.T.to_numpy()
    return squareform(pdist(matriz_muestras, metric="braycurtis"))


def calcular_pcoa(distancias, muestras_comunes):
    """Calcula una ordenacion PCoA (escalamiento multidimensional
    clasico) a partir de una matriz de distancias Bray-Curtis."""
    n = distancias.shape[0]
    matriz_centrado = np.eye(n) - np.ones((n, n)) / n
    matriz_b = -0.5 * matriz_centrado @ (distancias ** 2) @ matriz_centrado

    autovalores, autovectores = np.linalg.eigh(matriz_b)
    orden = np.argsort(autovalores)[::-1]
    autovalores = autovalores[orden]
    autovectores = autovectores[:, orden]

    varianza_total = autovalores[autovalores > 0].sum()
    porcentaje_varianza = 100 * autovalores / varianza_total

    coordenadas = autovectores[:, :2] * np.sqrt(
        np.clip(autovalores[:2], 0, None)
    )
    tabla_coordenadas = pd.DataFrame({
        "sample-id": muestras_comunes,
        "PC1": coordenadas[:, 0],
        "PC2": coordenadas[:, 1],
    })
    tabla_varianza = pd.DataFrame({
        "eje": ["PC1", "PC2"],
        "porcentaje_varianza": porcentaje_varianza[:2],
    })
    return tabla_coordenadas, tabla_varianza


def graficar_pcoa(tabla_coordenadas, tabla_varianza, metadata):
    """Grafica PC1 vs PC2 coloreando por gradiente continuo de
    AvgSoilRH (paleta plasma)."""
    datos = tabla_coordenadas.merge(metadata, on="sample-id", how="left")
    datos = datos.rename(
        columns={"average-soil-relative-humidity": "AvgSoilRH"}
    )
    datos_validos = datos.dropna(subset=["AvgSoilRH"])

    var_pc1 = tabla_varianza.loc[
        tabla_varianza["eje"] == "PC1", "porcentaje_varianza"
    ].iloc[0]
    var_pc2 = tabla_varianza.loc[
        tabla_varianza["eje"] == "PC2", "porcentaje_varianza"
    ].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    dispersion = ax.scatter(
        datos_validos["PC1"], datos_validos["PC2"],
        c=datos_validos["AvgSoilRH"], cmap="plasma",
        edgecolor="black",
    )
    barra_color = fig.colorbar(dispersion, ax=ax)
    barra_color.set_label("Humedad relativa promedio del suelo (%)")

    ax.set_title("PCoA (Bray-Curtis) de composicion microbiana")
    ax.set_xlabel(f"PC1 ({var_pc1:.1f}% varianza explicada)")
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}% varianza explicada)")
    fig.tight_layout()

    fig.savefig(f"{CARPETA_SALIDA}/fig2_pcoa_braycurtis.png", dpi=300)
    fig.savefig(f"{CARPETA_SALIDA}/fig2_pcoa_braycurtis.pdf")
    plt.close(fig)

    return datos


def evaluar_separacion_transectos(datos_validos):
    """Compara PC1 entre transectos para evaluar H2 (Yungay vs
    Baquedano se separan a lo largo del primer eje)."""
    baquedano = datos_validos.loc[
        datos_validos["transect-name"] == "Baquedano", "PC1"
    ]
    yungay = datos_validos.loc[
        datos_validos["transect-name"] == "Yungay", "PC1"
    ]
    estadistico, p_valor = stats.mannwhitneyu(
        baquedano, yungay, alternative="two-sided"
    )
    tabla = pd.DataFrame({
        "transect-name": ["Baquedano", "Yungay"],
        "PC1_mean": [baquedano.mean(), yungay.mean()],
        "PC1_median": [baquedano.median(), yungay.median()],
        "n": [len(baquedano), len(yungay)],
    })
    return tabla, p_valor


VARIABLES_AMBIENTALES = {
    "humedad_relativa": "average-soil-relative-humidity",
    "temperatura": "average-soil-temperature",
    "elevacion": "elevation",
}


def calcular_pseudo_f(matriz_gower, variable):
    """Calcula el pseudo-F de un PERMANOVA univariado: que tanta
    varianza de la matriz de Gower explica una variable continua."""
    n = matriz_gower.shape[0]
    diseno = np.column_stack([np.ones(n), variable])
    proyeccion = diseno @ np.linalg.pinv(diseno.T @ diseno) @ diseno.T

    ss_total = np.trace(matriz_gower)
    ss_explicada = np.trace(matriz_gower @ proyeccion)
    ss_residual = ss_total - ss_explicada

    grados_libertad_variable = 1
    grados_libertad_residual = n - grados_libertad_variable - 1
    f = (
        (ss_explicada / grados_libertad_variable)
        / (ss_residual / grados_libertad_residual)
    )
    r2 = ss_explicada / ss_total
    return f, r2


def permanova_univariado(distancias, muestras_comunes, metadata,
                          columna, n_permutaciones=999, semilla=0):
    """PERMANOVA univariado de una variable ambiental continua
    sobre una matriz de distancias Bray-Curtis, con p-valor
    obtenido por permutacion."""
    metadata_indexada = metadata.set_index("sample-id")
    valores = metadata_indexada.loc[muestras_comunes, columna]
    validos = valores.notna().to_numpy()
    indices_validos = np.where(validos)[0]
    n = len(indices_validos)

    distancias_validas = distancias[np.ix_(indices_validos,
                                            indices_validos)]
    matriz_centrado = np.eye(n) - np.ones((n, n)) / n
    matriz_gower = -0.5 * matriz_centrado @ (
        distancias_validas ** 2
    ) @ matriz_centrado

    variable = valores.to_numpy()[indices_validos].astype(float)
    f_observado, r2 = calcular_pseudo_f(matriz_gower, variable)

    generador = np.random.default_rng(semilla)
    contador = 0
    for _ in range(n_permutaciones):
        variable_permutada = generador.permutation(variable)
        f_permutado, _ = calcular_pseudo_f(
            matriz_gower, variable_permutada
        )
        if f_permutado >= f_observado:
            contador += 1
    p_valor = (contador + 1) / (n_permutaciones + 1)

    if n < 10:
        print(
            f"AVISO: n={n} para '{columna}' es menor a 10; las "
            "permutaciones pueden ser insuficientes."
        )

    return f_observado, r2, p_valor, n


def tabla_permanova(distancias, muestras_comunes, metadata):
    """Corre un PERMANOVA univariado para cada variable ambiental
    y ordena el resultado de mayor a menor R2."""
    filas = []
    for nombre, columna in VARIABLES_AMBIENTALES.items():
        f_obs, r2, p_valor, n = permanova_univariado(
            distancias, muestras_comunes, metadata, columna
        )
        filas.append({
            "variable": nombre,
            "columna_original": columna,
            "F": f_obs,
            "R2": r2,
            "p_valor": p_valor,
            "n_permutaciones": 999,
            "n": n,
        })
    tabla = pd.DataFrame(filas)
    tabla = tabla.sort_values("R2", ascending=False).reset_index(
        drop=True
    )
    return tabla


def main():
    datos = cargar_datos()

    resumen = resumen_por_transecto(datos)
    print("Resumen de Shannon por transecto:")
    print(resumen.to_string(index=False))
    print()

    graficar_boxplot(datos, resumen)

    datos_validos, tabla_correlacion = correlacion_spearman(datos)
    print("Correlacion de Spearman (Shannon vs AvgSoilRH):")
    print(tabla_correlacion.to_string(index=False))
    print()

    if tabla_correlacion.loc[0, "r2"] < 0.05:
        print(
            "AVISO: R2 menor a 0.05. Revisar datos o analisis "
            "antes de continuar."
        )

    graficar_dispersion_con_regresion(datos_validos, tabla_correlacion)

    resumen.to_csv(
        f"{CARPETA_SALIDA}/h1_resumen_shannon_por_transecto.tsv",
        sep="\t", index=False,
    )
    tabla_correlacion.to_csv(
        f"{CARPETA_SALIDA}/h1_correlacion_shannon_vs_humedad.tsv",
        sep="\t", index=False,
    )
    print("Figuras y tablas de H1 guardadas en outputs/.")
    print()

    abundancias_comunes, muestras_comunes, metadata = (
        cargar_abundancias_comunes()
    )
    distancias = calcular_distancias_bray_curtis(abundancias_comunes)
    tabla_coordenadas, tabla_varianza = calcular_pcoa(
        distancias, muestras_comunes
    )
    print("Varianza explicada por eje (PCoA Bray-Curtis):")
    print(tabla_varianza.to_string(index=False))
    print()

    datos_pcoa = graficar_pcoa(tabla_coordenadas, tabla_varianza, metadata)

    tabla_separacion, p_valor_separacion = evaluar_separacion_transectos(
        datos_pcoa
    )
    print("PC1 por transecto (para evaluar H2):")
    print(tabla_separacion.to_string(index=False))
    print(f"Mann-Whitney U, p-valor = {p_valor_separacion:.4g}")
    print()

    tabla_varianza.to_csv(
        f"{CARPETA_SALIDA}/h2_varianza_explicada_pcoa.tsv",
        sep="\t", index=False,
    )
    print("Figuras y tablas de H2 guardadas en outputs/.")
    print()

    tabla_h3 = tabla_permanova(distancias, muestras_comunes, metadata)
    print("PERMANOVA univariado por variable ambiental:")
    print(tabla_h3.to_string(index=False))
    print()

    tabla_h3.to_csv(
        f"{CARPETA_SALIDA}/h3_permanova_variables_ambientales.tsv",
        sep="\t", index=False,
    )
    print("Tabla de H3 guardada en outputs/.")


if __name__ == "__main__":
    main()
