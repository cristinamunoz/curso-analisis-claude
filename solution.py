"""Carga y filtra datos de abundancia y metadata para el analisis
de aridez y microbioma (Neilson et al. 2017)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from skbio.diversity import beta_diversity
from skbio.stats.distance import permanova
from skbio.stats.ordination import pcoa

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


def asteriscos_por_p_valor(p_valor):
    """Convierte un p-valor en su notacion de asteriscos habitual."""
    if p_valor < 0.001:
        return "***"
    if p_valor < 0.01:
        return "**"
    if p_valor < 0.05:
        return "*"
    return "ns"


def graficar_boxplot_por_transecto(diversidad):
    """Boxplot de riqueza, Shannon y Simpson por transecto, con
    prueba de Mann-Whitney U (Baquedano vs. Yungay) y asteriscos
    de significancia sobre cada panel."""
    metricas = ["richness", "shannon", "simpson"]
    etiquetas = {
        "richness": "Richness (número de OTUs)",
        "shannon": "Shannon diversity index",
        "simpson": "Simpson diversity index",
    }
    baquedano = diversidad[
        diversidad["transect-name"] == "Baquedano"
    ]
    yungay = diversidad[diversidad["transect-name"] == "Yungay"]

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
        _, p_valor = stats.mannwhitneyu(
            baquedano[metrica], yungay[metrica]
        )
        etiqueta_significancia = asteriscos_por_p_valor(p_valor)
        y_maximo = diversidad[metrica].max()
        y_linea = y_maximo * 1.05
        eje.plot([0, 0, 1, 1], [y_linea, y_linea * 1.02,
                                 y_linea * 1.02, y_linea],
                 color="black", linewidth=1)
        eje.text(
            0.5, y_linea * 1.02, etiqueta_significancia,
            ha="center", va="bottom",
        )
        eje.set_ylim(top=y_linea * 1.1)
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


def calcular_distancia_bray_curtis(abundancias):
    """Matriz de distancias Bray-Curtis entre muestras."""
    return beta_diversity(
        "braycurtis",
        abundancias.T.values,
        ids=abundancias.columns,
    )


def calcular_pcoa(distancias):
    """Ordenacion PCoA a partir de una matriz de distancias."""
    return pcoa(distancias)


def graficar_pcoa(resultado_pcoa, metadata):
    """PCoA coloreado por transecto (Set2) y por AvgSoilRH (viridis)."""
    coordenadas = resultado_pcoa.samples.loc[metadata.index]
    varianza = resultado_pcoa.proportion_explained
    etiqueta_x = f"PC1 ({varianza.iloc[0] * 100:.1f}% varianza)"
    etiqueta_y = f"PC2 ({varianza.iloc[1] * 100:.1f}% varianza)"

    fig, (eje1, eje2) = plt.subplots(1, 2, figsize=(8, 6))

    sns.scatterplot(
        x=coordenadas["PC1"],
        y=coordenadas["PC2"],
        hue=metadata["transect-name"],
        palette="Set2",
        ax=eje1,
    )
    eje1.set_xlabel(etiqueta_x)
    eje1.set_ylabel(etiqueta_y)
    eje1.set_title("PCoA coloreado por transecto")
    eje1.legend(title="Transect")

    dispersion = eje2.scatter(
        coordenadas["PC1"],
        coordenadas["PC2"],
        c=metadata["average-soil-relative-humidity"],
        cmap="viridis",
    )
    eje2.set_xlabel(etiqueta_x)
    eje2.set_ylabel(etiqueta_y)
    eje2.set_title("PCoA coloreado por AvgSoilRH")
    fig.colorbar(
        dispersion, ax=eje2, label="Average soil relative humidity (%)"
    )

    fig.suptitle("PCoA (Bray-Curtis) de la composicion comunitaria")
    fig.tight_layout()
    fig.savefig(
        f"{CARPETA_SALIDA}/fig_pcoa_braycurtis_baquedano_vs_yungay.png",
        dpi=300,
    )
    fig.savefig(
        f"{CARPETA_SALIDA}/fig_pcoa_braycurtis_baquedano_vs_yungay.pdf"
    )
    plt.close(fig)


def probar_permanova_transecto(distancias, metadata):
    """PERMANOVA (999 permutaciones) de composicion ~ transecto.

    Agrega R2 (varianza explicada por el transecto), que skbio no
    incluye en el resultado por defecto, a partir de pseudo-F,
    numero de grupos y tamano de muestra.
    """
    resultado = permanova(
        distancias,
        grouping=metadata["transect-name"],
        permutations=999,
    )
    f = resultado["test statistic"]
    n = resultado["sample size"]
    a = resultado["number of groups"]
    r2 = (f * (a - 1)) / (f * (a - 1) + (n - a))
    resultado["R2"] = r2
    return resultado


def permanova_variable_continua(distancias, valores, permutaciones=999):
    """PERMANOVA (metodo McArdle & Anderson) para un predictor
    continuo, ya que scikit-bio solo acepta grupos categoricos.

    Reporta F, R2 y p-valor comparando el pseudo-F observado
    contra pseudo-F de permutaciones aleatorias de las muestras.
    """
    ids_comunes = valores.dropna().index
    submatriz = distancias.filter(ids_comunes)
    x = valores.loc[ids_comunes].values
    n = len(ids_comunes)

    matriz_d = submatriz.data
    centrado = -0.5 * matriz_d ** 2
    centrador = np.eye(n) - np.ones((n, n)) / n
    g = centrador @ centrado @ centrador

    disenio = np.column_stack([np.ones(n), x])
    h = disenio @ np.linalg.pinv(disenio.T @ disenio) @ disenio.T

    ss_total = np.trace(g)
    df_modelo = 1
    df_residual = n - 2

    def pseudo_f(g_actual):
        ss_modelo = np.trace(h @ g_actual @ h)
        ss_residual = ss_total - ss_modelo
        return (ss_modelo / df_modelo) / (ss_residual / df_residual)

    f_observado = pseudo_f(g)
    r2 = (f_observado * df_modelo) / (
        f_observado * df_modelo + df_residual
    )

    rng = np.random.default_rng(0)
    f_permutados = np.empty(permutaciones)
    for i in range(permutaciones):
        orden = rng.permutation(n)
        f_permutados[i] = pseudo_f(g[np.ix_(orden, orden)])

    p_valor = (
        np.sum(f_permutados >= f_observado) + 1
    ) / (permutaciones + 1)

    return {
        "F": f_observado,
        "R2": r2,
        "p_valor": p_valor,
        "n_permutaciones": permutaciones,
        "n": n,
    }


def probar_permanova_variables_ambientales(distancias, metadata):
    """PERMANOVA univariada de humedad, temperatura y elevacion,
    ordenada de mayor a menor R2 (regla del skill de composicion)."""
    variables = {
        "humedad_relativa": "average-soil-relative-humidity",
        "temperatura": "average-soil-temperature",
        "elevacion": "elevation",
    }
    filas = []
    for nombre, columna in variables.items():
        resultado = permanova_variable_continua(
            distancias, metadata[columna]
        )
        filas.append({
            "variable": nombre,
            "columna_original": columna,
            "F": resultado["F"],
            "R2": resultado["R2"],
            "p_valor": resultado["p_valor"],
            "n_permutaciones": resultado["n_permutaciones"],
            "n": resultado["n"],
        })
    tabla = pd.DataFrame(filas).sort_values(
        "R2", ascending=False
    ).reset_index(drop=True)
    return tabla


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

    if (metadata_f["transect-name"].value_counts() < 10).any():
        print(
            "Aviso: hay un grupo con menos de 10 muestras; las "
            "permutaciones del PERMANOVA pueden ser insuficientes."
        )

    distancias = calcular_distancia_bray_curtis(abundancias_f)
    resultado_pcoa = calcular_pcoa(distancias)
    graficar_pcoa(resultado_pcoa, metadata_f)

    resultado_permanova = probar_permanova_transecto(
        distancias, metadata_f
    )
    print("\nPERMANOVA (Bray-Curtis) ~ transect-name:")
    print(resultado_permanova)

    tabla_h3 = probar_permanova_variables_ambientales(
        distancias, metadata_f
    )
    print("\nPERMANOVA univariada (humedad, temperatura, elevacion):")
    print(tabla_h3)
    tabla_h3.to_csv(
        f"{CARPETA_SALIDA}/h3_permanova_variables_ambientales.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()
