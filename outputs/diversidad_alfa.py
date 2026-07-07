"""Analisis general de diversidad alfa del dataset.

Calcula varias metricas de diversidad alfa (riqueza, Shannon,
Simpson, equidad de Pielou) por muestra, las resume por transecto
y las correlaciona con la humedad relativa del suelo (AvgSoilRH).
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

METRICAS = {
    "riqueza": "Riqueza observada (S)",
    "shannon": "Shannon (H')",
    "simpson": "Simpson (1-D)",
    "pielou": "Equidad de Pielou (J)",
}


def calcular_metricas_alfa(abundancias):
    """Calcula riqueza, Shannon, Simpson y equidad de Pielou.

    - Riqueza (S): numero de OTUs con abundancia > 0.
    - Shannon (H'): -suma(p_i * ln(p_i)).
    - Simpson (1-D): 1 - suma(p_i^2); probabilidad de que dos
      individuos al azar sean de especies distintas.
    - Pielou (J): H' / ln(S); que tan parejas son las abundancias.
    """
    proporciones = abundancias.div(abundancias.sum(axis=1), axis=0)

    riqueza = (abundancias > 0).sum(axis=1)

    log_proporciones = np.log(proporciones.where(proporciones > 0))
    shannon = -(proporciones * log_proporciones).sum(axis=1)

    simpson = 1 - (proporciones ** 2).sum(axis=1)

    pielou = shannon / np.log(riqueza)

    return pd.DataFrame({
        "riqueza": riqueza,
        "shannon": shannon,
        "simpson": simpson,
        "pielou": pielou,
    })


def main():
    abundancias = pd.read_csv(
        RUTA_ABUNDANCIAS, sep="\t", index_col=0
    ).T
    metadata = pd.read_csv(RUTA_METADATA, sep="\t")

    ids_comunes = metadata["sample-id"].isin(abundancias.index)
    metadata = metadata.loc[ids_comunes].copy()
    abundancias = abundancias.loc[metadata["sample-id"]]

    metricas = calcular_metricas_alfa(abundancias)
    metricas.index = metadata["sample-id"].values
    metadata = metadata.join(metricas, on="sample-id")

    columnas_base = ["sample-id", "transect-name",
                      "average-soil-relative-humidity"]
    datos_completos = metadata[
        columnas_base + list(METRICAS)
    ].dropna(subset=list(METRICAS))

    # --- resumen por transecto ---
    filas_resumen = []
    for metrica in METRICAS:
        resumen = (
            datos_completos.groupby("transect-name")[metrica]
            .agg(mean="mean", median="median", min="min",
                 max="max", count="count")
            .reset_index()
        )
        resumen.insert(0, "metrica", metrica)
        filas_resumen.append(resumen)
    resumen_total = pd.concat(filas_resumen, ignore_index=True)
    print("\nResumen de metricas de diversidad alfa por transecto:")
    print(resumen_total.to_string(index=False))
    resumen_total.to_csv(
        f"{CARPETA_SALIDA}/diversidad_alfa_resumen_por_transecto.tsv",
        sep="\t", index=False,
    )

    # --- boxplots 2x2 ---
    fig, ejes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, (metrica, titulo) in zip(ejes.flat, METRICAS.items()):
        sns.boxplot(
            data=datos_completos, x="transect-name", y=metrica,
            hue="transect-name", palette=COLOR_TRANSECTO,
            legend=False, ax=ax,
        )
        sns.stripplot(
            data=datos_completos, x="transect-name", y=metrica,
            color="black", alpha=0.5, size=3, ax=ax,
        )
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Metricas de diversidad alfa por transecto")

    postit = (
        "Post-it: que es la diversidad alfa?\n"
        "Es la diversidad DENTRO de una sola muestra o sitio (a\n"
        "diferencia de la diversidad beta, que compara la\n"
        "composicion ENTRE muestras, y la gamma, que es la\n"
        "diversidad total de la region). Combina dos ingredientes:\n"
        "  1) Riqueza (S): cuantas especies/OTUs distintos hay.\n"
        "  2) Equidad: que tan repartidas estan las abundancias\n"
        "     entre esas especies (una muestra con 50 especies\n"
        "     pero 1 dominante al 90% es 'menos diversa' que una\n"
        "     con 50 especies repartidas parejo).\n"
        "Shannon y Simpson combinan riqueza + equidad en un solo\n"
        "numero; Pielou aisla solo la equidad (0=una domina todo,\n"
        "1=todas las especies igual de abundantes)."
    )
    fig.text(
        0.5, -0.05, postit, ha="center", va="top", fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(
        f"{CARPETA_SALIDA}/fig_diversidad_alfa_por_transecto.png",
        dpi=300, bbox_inches="tight",
    )
    fig.savefig(
        f"{CARPETA_SALIDA}/fig_diversidad_alfa_por_transecto.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # --- correlacion con AvgSoilRH ---
    datos_humedad = datos_completos.dropna(
        subset=["average-soil-relative-humidity"]
    )
    filas_corr = []
    for metrica in METRICAS:
        rho, p_rho = stats.spearmanr(
            datos_humedad["average-soil-relative-humidity"],
            datos_humedad[metrica],
        )
        _, _, r_valor, p_lineal, _ = stats.linregress(
            datos_humedad["average-soil-relative-humidity"],
            datos_humedad[metrica],
        )
        filas_corr.append({
            "metrica": metrica,
            "variable_x": "average-soil-relative-humidity",
            "spearman_rho": rho,
            "p_valor_spearman": p_rho,
            "r2_lineal": r_valor ** 2,
            "p_valor_lineal": p_lineal,
            "n": len(datos_humedad),
        })
    tabla_corr = pd.DataFrame(filas_corr)
    print("\nCorrelacion de cada metrica con AvgSoilRH:")
    print(tabla_corr.to_string(index=False))
    tabla_corr.to_csv(
        f"{CARPETA_SALIDA}/diversidad_alfa_correlaciones_vs_humedad.tsv",
        sep="\t", index=False,
    )

    metricas_bajo_umbral = tabla_corr[tabla_corr["r2_lineal"] < 0.05]
    if not metricas_bajo_umbral.empty:
        print(
            "\nAVISO: las siguientes metricas tienen R2 < 0.05 "
            "frente a AvgSoilRH -- revisar antes de continuar:"
        )
        print(metricas_bajo_umbral["metrica"].to_string(index=False))

    print(f"\nArchivos guardados en '{CARPETA_SALIDA}/'.")


if __name__ == "__main__":
    main()
