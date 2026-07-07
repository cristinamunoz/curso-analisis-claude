"""Diversidad alfa (Shannon) vs. humedad relativa del suelo.

Reproduce H1 de Neilson et al. (2017): a menor humedad relativa
del suelo (AvgSoilRH), menor diversidad alfa (Shannon).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

RUTA_ABUNDANCIAS = "data/abundancias.tsv"
RUTA_METADATA = "data/metadata.tsv"
CARPETA_SALIDA = "outputs"

COLOR_TRANSECTO = {
    "Baquedano": "#66c2a5",
    "Yungay": "#fc8d62",
}


def calcular_shannon(abundancias):
    """Calcula el indice de Shannon (H') para cada muestra.

    H' = -suma(p_i * ln(p_i)), con p_i = abundancia_i / total.
    """
    proporciones = abundancias.div(abundancias.sum(axis=1), axis=0)
    log_proporciones = np.log(proporciones.where(proporciones > 0))
    shannon = -(proporciones * log_proporciones).sum(axis=1)
    return shannon


def main():
    abundancias = pd.read_csv(
        RUTA_ABUNDANCIAS, sep="\t", index_col=0
    ).T
    metadata = pd.read_csv(RUTA_METADATA, sep="\t")

    ids_comunes = metadata["sample-id"].isin(abundancias.index)
    metadata = metadata.loc[ids_comunes].copy()
    abundancias = abundancias.loc[metadata["sample-id"]]

    metadata["shannon"] = calcular_shannon(abundancias).values

    datos_completos = metadata[
        ["sample-id", "transect-name",
         "average-soil-relative-humidity", "shannon"]
    ].dropna(subset=["shannon"])

    resumen = (
        datos_completos.groupby("transect-name")["shannon"]
        .agg(mean="mean", median="median", min="min", max="max",
             count="count")
        .reset_index()
    )
    print("\nResumen de Shannon por transecto:")
    print(resumen.to_string(index=False))

    datos = datos_completos.dropna(
        subset=["average-soil-relative-humidity"]
    )

    resumen.to_csv(
        f"{CARPETA_SALIDA}/h1_resumen_shannon_por_transecto.tsv",
        sep="\t", index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=datos_completos, x="transect-name", y="shannon",
        hue="transect-name", palette=COLOR_TRANSECTO,
        legend=False, ax=ax,
    )
    sns.stripplot(
        data=datos_completos, x="transect-name", y="shannon",
        color="black", alpha=0.5, size=4, ax=ax,
    )
    ax.set_title("Diversidad de Shannon por transecto")
    ax.set_xlabel("Transecto")
    ax.set_ylabel("Indice de diversidad de Shannon (H')")
    fig.tight_layout()
    fig.savefig(
        f"{CARPETA_SALIDA}/fig1_shannon_por_transecto.png", dpi=300
    )
    fig.savefig(f"{CARPETA_SALIDA}/fig1_shannon_por_transecto.pdf")
    plt.close(fig)

    rho, p_rho = stats.spearmanr(
        datos["average-soil-relative-humidity"], datos["shannon"]
    )
    pendiente, intercepto, r_valor, p_lineal, error_std = stats.linregress(
        datos["average-soil-relative-humidity"], datos["shannon"]
    )
    r2 = r_valor ** 2
    n = len(datos)

    correlacion = pd.DataFrame([{
        "variable_x": "average-soil-relative-humidity",
        "variable_y": "shannon",
        "spearman_rho": rho,
        "p_valor": p_rho,
        "n": n,
    }])
    correlacion.to_csv(
        f"{CARPETA_SALIDA}/h1_correlacion_shannon_vs_humedad.tsv",
        sep="\t", index=False,
    )
    print("\nCorrelacion Spearman (Shannon vs. AvgSoilRH):")
    print(correlacion.to_string(index=False))
    print(f"R2 (regresion lineal): {r2:.4f}")
    print(f"p-valor (regresion lineal): {p_lineal:.4g}")

    if r2 < 0.05:
        print(
            "\nAVISO: R2 < 0.05. Revisar si el filtrado o la "
            "normalizacion son correctos antes de continuar."
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        data=datos, x="average-soil-relative-humidity", y="shannon",
        ci=95, scatter=False, color="#4d4d4d", ax=ax,
    )
    sns.scatterplot(
        data=datos, x="average-soil-relative-humidity", y="shannon",
        hue="transect-name", palette=COLOR_TRANSECTO, s=60, ax=ax,
    )
    ax.set_title(
        "Diversidad de Shannon vs. humedad relativa del suelo"
    )
    ax.set_xlabel("Humedad relativa promedio del suelo (AvgSoilRH, %)")
    ax.set_ylabel("Indice de diversidad de Shannon (H')")
    ax.legend(title="Transecto")

    texto_stats = (
        f"Spearman rho = {rho:.3f}\n"
        f"R2 = {r2:.3f}\n"
        f"p-valor = {p_lineal:.3g}\n"
        f"n = {n}"
    )
    ax.text(
        0.02, 0.02, texto_stats, transform=ax.transAxes,
        fontsize=9, va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    postit = (
        "Post-it estadistico:\n"
        "- Shannon (H') mide diversidad: combina cuantas especies\n"
        "  (OTUs) hay y que tan parejas son sus abundancias.\n"
        "- Spearman rho mide si, al ordenar las muestras, la\n"
        "  humedad y la diversidad suben o bajan juntas (va de\n"
        "  -1 a 1; no asume relacion lineal).\n"
        "- R2 (de la regresion lineal) indica que fraccion de la\n"
        "  variacion en Shannon se explica por AvgSoilRH (0 a 1).\n"
        "- El p-valor estima la probabilidad de ver esta relacion\n"
        "  si en realidad no existiera ninguna (se suele usar\n"
        "  0.05 como umbral de referencia)."
    )
    fig.text(
        0.5, -0.02, postit, ha="center", va="top", fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(
        f"{CARPETA_SALIDA}/fig1_shannon_vs_avgsoilrh.png",
        dpi=300, bbox_inches="tight",
    )
    fig.savefig(
        f"{CARPETA_SALIDA}/fig1_shannon_vs_avgsoilrh.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\nFiguras y tablas guardadas en '{CARPETA_SALIDA}/'.")


if __name__ == "__main__":
    main()
