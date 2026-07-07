"""
Analisis H1: relacion entre diversidad alfa (Shannon) y humedad
relativa del suelo (AvgSoilRH) en el desierto de Atacama.

Hipotesis: a menor humedad relativa del suelo, menor diversidad
alfa (Shannon).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import squareform, pdist

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_DIR_H2 = "outputs2"
os.makedirs(OUTPUT_DIR_H2, exist_ok=True)

PALETTE_TRANSECTOS = {
    "Baquedano": "#1b9e77",
    "Yungay": "#d95f02",
}


def cargar_datos():
    """Carga abundancias y metadata, y las cruza por sample-id."""
    abundancias = pd.read_csv(
        "data/abundancias.tsv", sep="\t", index_col=0
    )
    metadata = pd.read_csv(
        "data/metadata.tsv", sep="\t", index_col="sample-id"
    )

    # Solo 54 de las 75 muestras en metadata tienen secuenciacion.
    # Se usa la interseccion de IDs entre ambos archivos.
    muestras_comunes = abundancias.columns.intersection(
        metadata.index
    )
    abundancias = abundancias[muestras_comunes]
    metadata = metadata.loc[muestras_comunes]

    return abundancias, metadata


def calcular_shannon(abundancias):
    """Calcula el indice de diversidad de Shannon por muestra."""
    shannon_por_muestra = {}
    for muestra in abundancias.columns:
        conteos = abundancias[muestra].values
        conteos = conteos[conteos > 0]
        proporciones = conteos / conteos.sum()
        shannon = -np.sum(proporciones * np.log(proporciones))
        shannon_por_muestra[muestra] = shannon
    return pd.Series(shannon_por_muestra, name="shannon")


def resumen_por_transecto(datos):
    """Imprime media, mediana y rango de Shannon por transecto."""
    resumen = datos.groupby("transect-name")["shannon"].agg(
        media="mean",
        mediana="median",
        minimo="min",
        maximo="max",
        n="count",
    )
    print("\nResumen de diversidad de Shannon por transecto:")
    print(resumen.round(3))
    return resumen


def boxplot_por_transecto(datos):
    """Genera y guarda el boxplot de Shannon por transecto."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=datos,
        x="transect-name",
        y="shannon",
        hue="transect-name",
        palette=PALETTE_TRANSECTOS,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=datos,
        x="transect-name",
        y="shannon",
        color="black",
        alpha=0.5,
        ax=ax,
    )
    ax.set_title(
        "Diversidad de Shannon por transecto\n"
        "Desierto de Atacama"
    )
    ax.set_xlabel("Transecto")
    ax.set_ylabel("Indice de diversidad de Shannon")
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "fig1_shannon_por_transecto.png"),
        dpi=300,
    )
    fig.savefig(
        os.path.join(OUTPUT_DIR, "fig1_shannon_por_transecto.pdf")
    )
    plt.close(fig)


def correlacion_shannon_humedad(datos):
    """Calcula correlacion de Spearman y regresion lineal."""
    columna_humedad = "average-soil-relative-humidity"
    n_faltantes = datos[columna_humedad].isna().sum()
    if n_faltantes > 0:
        print(
            f"\n  AVISO: {n_faltantes} muestra(s) sin dato de "
            "humedad relativa del suelo - se excluyen de la "
            "correlacion y la regresion."
        )
    datos_validos = datos.dropna(subset=[columna_humedad])

    x = datos_validos[columna_humedad]
    y = datos_validos["shannon"]

    rho, p_valor_spearman = stats.spearmanr(x, y)

    pendiente, intercepto, r_valor, p_valor_reg, error_est = (
        stats.linregress(x, y)
    )
    r2 = r_valor ** 2
    n = len(datos_validos)

    print("\nCorrelacion Shannon vs. humedad relativa del suelo:")
    print(f"  Spearman rho = {rho:.3f} (p = {p_valor_spearman:.4f})")
    print(f"  R2 (regresion lineal) = {r2:.3f}")
    print(f"  p-valor (regresion) = {p_valor_reg:.4f}")
    print(f"  n = {n}")

    if r2 < 0.05:
        print(
            "\n  AVISO: R2 menor a 0.05 - la relacion lineal es muy "
            "debil. Revisar antes de continuar."
        )

    resultado = pd.DataFrame(
        {
            "spearman_rho": [rho],
            "spearman_p_valor": [p_valor_spearman],
            "r2_regresion": [r2],
            "p_valor_regresion": [p_valor_reg],
            "pendiente": [pendiente],
            "n": [n],
        }
    )
    resultado.to_csv(
        os.path.join(
            OUTPUT_DIR, "h1_correlacion_shannon_vs_humedad.tsv"
        ),
        sep="\t",
        index=False,
    )
    return rho, p_valor_spearman, r2, p_valor_reg, n


def scatter_shannon_humedad(datos, r2, p_valor):
    """Genera el scatter Shannon vs. humedad con banda de IC 95%."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.regplot(
        data=datos,
        x="average-soil-relative-humidity",
        y="shannon",
        ci=95,
        scatter_kws={"alpha": 0.6, "color": "#7570b3"},
        line_kws={"color": "#d95f02"},
        ax=ax,
    )

    ax.set_title(
        "Diversidad de Shannon vs. humedad relativa del suelo\n"
        "Desierto de Atacama"
    )
    ax.set_xlabel("Humedad relativa promedio del suelo (%)")
    ax.set_ylabel("Indice de diversidad de Shannon")

    texto = f"R2 = {r2:.3f}\np = {p_valor:.4f}"
    ax.text(
        0.05,
        0.05,
        texto,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.8
        ),
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            OUTPUT_DIR, "fig1_shannon_vs_avgsoilrh.png"
        ),
        dpi=300,
    )
    fig.savefig(
        os.path.join(OUTPUT_DIR, "fig1_shannon_vs_avgsoilrh.pdf")
    )
    plt.close(fig)


def calcular_pcoa(abundancias):
    """
    Calcula PCoA clasico a partir de distancias Bray-Curtis.

    Usa el metodo de Gower: centra la matriz de distancias al
    cuadrado y extrae los eigenvalores/eigenvectores. Es el mismo
    procedimiento que usan librerias como scikit-bio, sin depender
    de esa libreria.
    """
    muestras = abundancias.columns
    matriz_conteos = abundancias.T.values

    distancias = squareform(
        pdist(matriz_conteos, metric="braycurtis")
    )

    n = distancias.shape[0]
    matriz_a = -0.5 * distancias ** 2
    centrador = np.eye(n) - np.ones((n, n)) / n
    matriz_g = centrador @ matriz_a @ centrador

    eigenvalores, eigenvectores = np.linalg.eigh(matriz_g)

    # np.linalg.eigh ordena de menor a mayor; se invierte el orden.
    orden = np.argsort(eigenvalores)[::-1]
    eigenvalores = eigenvalores[orden]
    eigenvectores = eigenvectores[:, orden]

    varianza_total = eigenvalores[eigenvalores > 0].sum()
    porcentaje_varianza = (
        eigenvalores / varianza_total * 100
    )

    coordenadas = eigenvectores * np.sqrt(
        np.clip(eigenvalores, a_min=0, a_max=None)
    )

    coordenadas_df = pd.DataFrame(
        coordenadas[:, :2],
        index=muestras,
        columns=["PC1", "PC2"],
    )

    return coordenadas_df, porcentaje_varianza[:2]


def guardar_varianza_pcoa(porcentaje_varianza):
    """Guarda el % de varianza explicado por PC1 y PC2."""
    tabla = pd.DataFrame(
        {
            "eje": ["PC1", "PC2"],
            "porcentaje_varianza": porcentaje_varianza,
        }
    )
    print("\nVarianza explicada por la ordenacion PCoA:")
    print(tabla.round(2).to_string(index=False))
    tabla.to_csv(
        os.path.join(
            OUTPUT_DIR_H2, "h2_varianza_explicada_pcoa.tsv"
        ),
        sep="\t",
        index=False,
    )
    return tabla


def graficar_pcoa(coordenadas, metadata, porcentaje_varianza):
    """
    Genera el scatter de PCoA (PC1 vs PC2).

    Color: gradiente continuo de humedad relativa del suelo
    (paleta viridis). Forma: transecto (Baquedano vs Yungay), para
    evaluar visualmente la hipotesis H2.
    """
    datos = coordenadas.join(metadata)
    columna_humedad = "average-soil-relative-humidity"
    con_humedad = datos[datos[columna_humedad].notna()]
    sin_humedad = datos[datos[columna_humedad].isna()]

    fig, ax = plt.subplots(figsize=(8, 6))

    marcadores = {"Baquedano": "o", "Yungay": "^"}

    dispersión = ax.scatter(
        con_humedad["PC1"],
        con_humedad["PC2"],
        c=con_humedad[columna_humedad],
        cmap="viridis",
        s=90,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    if len(sin_humedad) > 0:
        ax.scatter(
            sin_humedad["PC1"],
            sin_humedad["PC2"],
            facecolor="#d3d3d3",
            hatch="///",
            s=90,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            label="Sin dato de humedad",
        )

    for transecto, marcador in marcadores.items():
        subconjunto = datos[datos["transect-name"] == transecto]
        ax.scatter(
            subconjunto["PC1"],
            subconjunto["PC2"],
            facecolor="none",
            edgecolor="black",
            marker=marcador,
            s=140,
            linewidth=1.2,
            label=transecto,
        )

    barra_color = fig.colorbar(dispersión, ax=ax)
    barra_color.set_label(
        "Humedad relativa promedio del suelo (%)"
    )

    ax.set_xlabel(f"PC1 ({porcentaje_varianza[0]:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({porcentaje_varianza[1]:.1f}% varianza)")
    ax.set_title(
        "PCoA (Bray-Curtis) de la composicion microbiana\n"
        "Desierto de Atacama"
    )
    ax.legend(title="Transecto / dato faltante", loc="best")

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR_H2, "fig2_pcoa_braycurtis.png"),
        dpi=300,
    )
    fig.savefig(
        os.path.join(OUTPUT_DIR_H2, "fig2_pcoa_braycurtis.pdf")
    )
    plt.close(fig)


def main_h2():
    abundancias, metadata = cargar_datos()

    coordenadas, porcentaje_varianza = calcular_pcoa(abundancias)
    guardar_varianza_pcoa(porcentaje_varianza)
    graficar_pcoa(coordenadas, metadata, porcentaje_varianza)

    print(
        "\nListo. Figuras y tabla de H2 guardadas en la carpeta "
        "'outputs2/'."
    )


def main():
    abundancias, metadata = cargar_datos()
    shannon = calcular_shannon(abundancias)

    datos = metadata.copy()
    datos["shannon"] = shannon

    resumen = resumen_por_transecto(datos)
    resumen.to_csv(
        os.path.join(
            OUTPUT_DIR, "h1_resumen_shannon_por_transecto.tsv"
        ),
        sep="\t",
    )

    boxplot_por_transecto(datos)

    rho, p_spearman, r2, p_reg, n = correlacion_shannon_humedad(
        datos
    )
    scatter_shannon_humedad(datos, r2, p_reg)

    print(
        "\nListo. Figuras y tablas guardadas en la carpeta "
        "'outputs/'."
    )


if __name__ == "__main__":
    main()
    main_h2()
