"""Carga y filtra datos de abundancia y metadata para el analisis
de aridez y microbioma (Neilson et al. 2017)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

RUTA_ABUNDANCIAS = "data/abundancias.tsv"
RUTA_METADATA = "data/metadata.tsv"
CARPETA_SALIDA = "outputs"


def cargar_datos():
    """Lee las tablas de abundancias y metadata desde disco."""
    abundancias = pd.read_csv(
        RUTA_ABUNDANCIAS, sep="\t", index_col=0
    )
    metadata = pd.read_csv(
        RUTA_METADATA, sep="\t", index_col="sample-id"
    )
    return abundancias, metadata


def filtrar_muestras_en_comun(abundancias, metadata):
    """Deja solo las muestras presentes en ambas tablas.

    Solo 54 de las 75 muestras de metadata tienen datos de
    secuenciacion, asi que no se puede asumir correspondencia
    1 a 1 entre ambos archivos.
    """
    muestras_comunes = abundancias.columns.intersection(
        metadata.index
    )
    abundancias_filtradas = abundancias[muestras_comunes]
    metadata_filtrada = metadata.loc[muestras_comunes]
    return abundancias_filtradas, metadata_filtrada


def calcular_riqueza(abundancias):
    """Numero de OTUs con al menos una lectura, por muestra."""
    return (abundancias > 0).sum(axis=0)


def calcular_shannon(abundancias):
    """Indice de Shannon (diversidad), por muestra."""
    proporciones = abundancias / abundancias.sum(axis=0)
    log_proporciones = np.log(proporciones.where(proporciones > 0))
    return -(proporciones * log_proporciones).sum(axis=0)


def calcular_simpson(abundancias):
    """Indice de Simpson (1 - dominancia), por muestra."""
    proporciones = abundancias / abundancias.sum(axis=0)
    return 1 - (proporciones ** 2).sum(axis=0)


def calcular_diversidad_alfa(abundancias, metadata):
    """Junta riqueza, Shannon y Simpson con la metadata."""
    diversidad = pd.DataFrame({
        "richness": calcular_riqueza(abundancias),
        "shannon": calcular_shannon(abundancias),
        "simpson": calcular_simpson(abundancias),
    })
    return diversidad.join(metadata)


def resumen_por_transecto(diversidad):
    """Media, mediana y rango de cada indice por transecto."""
    metricas = ["richness", "shannon", "simpson"]
    resumen = diversidad.groupby("transect-name")[metricas].agg(
        ["mean", "median", "min", "max"]
    )
    return resumen


def graficar_boxplot_por_transecto(diversidad):
    """Boxplot de riqueza, Shannon y Simpson por transecto."""
    metricas = ["richness", "shannon", "simpson"]
    etiquetas = {
        "richness": "Richness (número de OTUs)",
        "shannon": "Shannon diversity index",
        "simpson": "Simpson diversity index",
    }
    fig, ejes = plt.subplots(1, 3, figsize=(8, 6))
    for eje, metrica in zip(ejes, metricas):
        sns.boxplot(
            data=diversidad,
            x="transect-name",
            y=metrica,
            hue="transect-name",
            palette="Set2",
            legend=False,
            ax=eje,
        )
        eje.set_xlabel("Transect")
        eje.set_ylabel(etiquetas[metrica])
        eje.set_title(etiquetas[metrica])
    fig.suptitle("Alpha diversity by transect")
    fig.tight_layout()
    fig.savefig(
        f"{CARPETA_SALIDA}/fig_alpha_diversity_boxplot_by_transect.png",
        dpi=300,
    )
    fig.savefig(
        f"{CARPETA_SALIDA}/fig_alpha_diversity_boxplot_by_transect.pdf"
    )
    plt.close(fig)


def graficar_scatter_vs_humedad(diversidad):
    """Scatter de riqueza, Shannon y Simpson vs AvgSoilRH."""
    metricas = ["richness", "shannon", "simpson"]
    etiquetas = {
        "richness": "Richness (número de OTUs)",
        "shannon": "Shannon diversity index",
        "simpson": "Simpson diversity index",
    }
    datos = diversidad.dropna(
        subset=["average-soil-relative-humidity"]
    )
    x = datos["average-soil-relative-humidity"]

    fig, ejes = plt.subplots(1, 3, figsize=(8, 6))
    for eje, metrica in zip(ejes, metricas):
        y = datos[metrica]
        eje.scatter(x, y, c=x, cmap="viridis")
        sns.regplot(
            x=x,
            y=y,
            ax=eje,
            scatter=False,
            line_kws={"color": "black"},
        )
        pendiente, interseccion, r_pearson, p_lineal, _ = (
            stats.linregress(x, y)
        )
        rho, p_spearman = stats.spearmanr(x, y)
        r2 = r_pearson ** 2
        n = len(x)
        texto = (
            f"R²={r2:.2f}, p={p_lineal:.3f}\n"
            f"Spearman ρ={rho:.2f}, p={p_spearman:.3f}\n"
            f"n={n}"
        )
        eje.text(
            0.05, 0.95, texto,
            transform=eje.transAxes,
            va="top", fontsize=8,
        )
        eje.set_xlabel("Average soil relative humidity (%)")
        eje.set_ylabel(etiquetas[metrica])
        eje.set_title(etiquetas[metrica])
        if r2 < 0.05:
            print(
                f"Aviso: R2 muy bajo ({r2:.3f}) para {metrica} "
                "vs AvgSoilRH. Revisar antes de continuar."
            )
    fig.suptitle("Alpha diversity vs average soil relative humidity")
    fig.tight_layout()
    fig.savefig(
        f"{CARPETA_SALIDA}/fig_alpha_diversity_vs_avgsoilrh.png",
        dpi=300,
    )
    fig.savefig(
        f"{CARPETA_SALIDA}/fig_alpha_diversity_vs_avgsoilrh.pdf"
    )
    plt.close(fig)


def main():
    abundancias, metadata = cargar_datos()
    abundancias_f, metadata_f = filtrar_muestras_en_comun(
        abundancias, metadata
    )
    print(f"OTUs: {abundancias_f.shape[0]}")
    print(f"Muestras en comun: {abundancias_f.shape[1]}")
    print(metadata_f["transect-name"].value_counts())

    diversidad = calcular_diversidad_alfa(abundancias_f, metadata_f)
    print("\nResumen de diversidad alfa por transecto:")
    print(resumen_por_transecto(diversidad))

    graficar_boxplot_por_transecto(diversidad)
    graficar_scatter_vs_humedad(diversidad)
    print(f"\nFiguras guardadas en {CARPETA_SALIDA}/")


if __name__ == "__main__":
    main()
