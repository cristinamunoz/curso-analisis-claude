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

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
