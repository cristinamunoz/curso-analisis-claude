"""Analisis de diversidad y composicion microbiana - Atacama.

Pregunta guia: a medida que el suelo del desierto de Atacama se
vuelve mas arido, como cambian la diversidad y la composicion de
la comunidad microbiana?

Este script reproduce, hipotesis por hipotesis, el analisis de
Neilson et al. (2017). Por ahora contiene H1 (diversidad alfa).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa

OUTPUT_DIR = "outputs"
DATA_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def cargar_datos():
    """Carga abundancias y metadata, cruzando por sample-id.

    Solo 54 de las 75 muestras de metadata tienen datos de
    abundancia, asi que se usa la interseccion de IDs.
    """
    ruta_abund = os.path.join(DATA_DIR, "abundancias.tsv")
    ruta_meta = os.path.join(DATA_DIR, "metadata.tsv")

    abundancias = pd.read_csv(ruta_abund, sep="\t", index_col=0)
    metadata = pd.read_csv(ruta_meta, sep="\t", index_col=0)

    # Cada columna de abundancias es una muestra; cada fila es un
    # OTU. Se transpone para tener una fila por muestra.
    abundancias = abundancias.T

    muestras_comunes = abundancias.index.intersection(metadata.index)
    abundancias = abundancias.loc[muestras_comunes]
    metadata = metadata.loc[muestras_comunes]

    print(
        f"Muestras en metadata: {len(metadata.index)} "
        f"(antes de cruzar con abundancias)"
    )
    print(f"Muestras con datos de abundancia: {len(muestras_comunes)}")

    return abundancias, metadata


def calcular_shannon(abundancias):
    """Calcula el indice de diversidad de Shannon por muestra.

    Shannon = -suma(p_i * ln(p_i)), donde p_i es la proporcion
    de cada OTU dentro de una muestra. Un valor mas alto indica
    una comunidad mas diversa (mas OTUs y mas equilibrados en
    abundancia).
    """
    valores = abundancias.to_numpy(dtype=float)
    totales = valores.sum(axis=1, keepdims=True)
    proporciones = np.divide(
        valores, totales, out=np.zeros_like(valores),
        where=totales > 0,
    )
    # log(0) no esta definido; se ignoran las proporciones en 0.
    log_proporciones = np.where(
        proporciones > 0, np.log(proporciones), 0.0
    )
    shannon = -(proporciones * log_proporciones).sum(axis=1)
    return pd.Series(shannon, index=abundancias.index, name="shannon")


def resumen_por_transecto(tabla):
    """Imprime y guarda media, mediana y rango de Shannon por
    transecto (Baquedano vs. Yungay)."""
    resumen = tabla.groupby("transect-name")["shannon"].agg(
        media="mean",
        mediana="median",
        minimo="min",
        maximo="max",
        n="count",
    )
    print("\nResumen de diversidad de Shannon por transecto:")
    print(resumen.round(3))

    ruta = os.path.join(
        OUTPUT_DIR, "h1_resumen_shannon_por_transecto.tsv"
    )
    resumen.round(3).to_csv(ruta, sep="\t")
    print(f"Guardado: {ruta}")
    return resumen


def boxplot_por_transecto(tabla):
    """Boxplot de Shannon por transecto, guardado en PNG y PDF."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=tabla, x="transect-name", y="shannon",
        hue="transect-name", palette="Set2", legend=False, ax=ax,
    )
    sns.stripplot(
        data=tabla, x="transect-name", y="shannon",
        color="black", alpha=0.5, ax=ax,
    )
    ax.set_xlabel("Transect")
    ax.set_ylabel("Shannon diversity index")
    ax.set_title("Alpha diversity (Shannon) by transect")

    for ext in ("png", "pdf"):
        ruta = os.path.join(
            OUTPUT_DIR, f"fig1_shannon_por_transecto.{ext}"
        )
        fig.savefig(ruta, dpi=300, bbox_inches="tight")
        print(f"Guardado: {ruta}")
    plt.close(fig)


def correlacion_shannon_humedad(tabla):
    """Correlacion de Spearman entre Shannon y humedad relativa
    del suelo (AvgSoilRH), con scatter + regresion lineal."""
    columna_humedad = "average-soil-relative-humidity"
    datos = tabla[["shannon", columna_humedad]].dropna()

    rho, p_valor = stats.spearmanr(
        datos[columna_humedad], datos["shannon"]
    )
    pendiente, intercepto, r_lineal, _, _ = stats.linregress(
        datos[columna_humedad], datos["shannon"]
    )
    r2 = r_lineal ** 2
    n = len(datos)

    print("\nCorrelacion Shannon vs. humedad relativa del suelo:")
    print(f"  Spearman rho = {rho:.3f}, p-valor = {p_valor:.4g}")
    print(f"  R^2 (regresion lineal) = {r2:.3f}, n = {n}")
    if r2 < 0.05:
        print(
            "  Aviso: R^2 < 0.05. La relacion lineal es muy debil; "
            "revisar antes de continuar."
        )

    tabla_resultado = pd.DataFrame(
        {
            "spearman_rho": [rho],
            "p_valor": [p_valor],
            "r2_lineal": [r2],
            "n": [n],
        }
    )
    ruta_tabla = os.path.join(
        OUTPUT_DIR, "h1_correlacion_shannon_vs_humedad.tsv"
    )
    tabla_resultado.round(4).to_csv(ruta_tabla, sep="\t", index=False)
    print(f"Guardado: {ruta_tabla}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        data=datos, x=columna_humedad, y="shannon",
        ci=95, ax=ax,
        scatter_kws={"alpha": 0.7, "color": "#3b528b"},
        line_kws={"color": "#440154"},
    )
    ax.set_xlabel("Average soil relative humidity (%)")
    ax.set_ylabel("Shannon diversity index")
    ax.set_title("Shannon diversity vs. soil relative humidity")
    texto = f"R² = {r2:.3f}, p = {p_valor:.3g}, n = {n}"
    ax.text(
        0.05, 0.95, texto, transform=ax.transAxes,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    for ext in ("png", "pdf"):
        ruta = os.path.join(
            OUTPUT_DIR, f"fig1_shannon_vs_avgsoilrh.{ext}"
        )
        fig.savefig(ruta, dpi=300, bbox_inches="tight")
        print(f"Guardado: {ruta}")
    plt.close(fig)

    return tabla_resultado


def calcular_pcoa_bray_curtis(abundancias):
    """Calcula distancias Bray-Curtis entre muestras y hace un
    PCoA sobre esa matriz de distancias.

    Bray-Curtis compara la composicion de OTUs entre pares de
    muestras (0 = comunidades identicas, 1 = sin OTUs en comun).
    El PCoA proyecta esas distancias en un espacio de pocos ejes
    para poder visualizarlas.
    """
    distancias = beta_diversity(
        "braycurtis",
        abundancias.to_numpy(dtype=float),
        ids=abundancias.index,
    )
    resultado = pcoa(distancias)
    return resultado


def graficar_pcoa(tabla, resultado_pcoa):
    """Grafica PC1 vs PC2, coloreando por humedad relativa del
    suelo (viridis) y usando la forma del punto para el
    transecto, para evaluar si Yungay y Baquedano se separan
    (H2)."""
    columna_humedad = "average-soil-relative-humidity"
    coordenadas = resultado_pcoa.samples.loc[tabla.index, ["PC1", "PC2"]]
    datos = coordenadas.join(tabla[[columna_humedad, "transect-name"]])

    var_pc1 = resultado_pcoa.proportion_explained["PC1"] * 100
    var_pc2 = resultado_pcoa.proportion_explained["PC2"] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    formas = {"Baquedano": "o", "Yungay": "^"}
    for transecto, forma in formas.items():
        subset = datos[datos["transect-name"] == transecto]
        dispersion = ax.scatter(
            subset["PC1"], subset["PC2"],
            c=subset[columna_humedad], cmap="viridis",
            marker=forma, s=70, edgecolor="black",
            linewidth=0.5, vmin=datos[columna_humedad].min(),
            vmax=datos[columna_humedad].max(), label=transecto,
        )

    barra = fig.colorbar(dispersion, ax=ax)
    barra.set_label("Average soil relative humidity (%)")
    ax.legend(title="Transect")
    ax.set_xlabel(f"PC1 ({var_pc1:.1f}% variance explained)")
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}% variance explained)")
    ax.set_title("PCoA on Bray-Curtis dissimilarity")

    for ext in ("png", "pdf"):
        ruta = os.path.join(OUTPUT_DIR, f"fig2_pcoa_braycurtis.{ext}")
        fig.savefig(ruta, dpi=300, bbox_inches="tight")
        print(f"Guardado: {ruta}")
    plt.close(fig)


def varianza_explicada_pcoa(resultado_pcoa):
    """Imprime y guarda el porcentaje de varianza explicado por
    PC1 y PC2."""
    varianza = resultado_pcoa.proportion_explained[["PC1", "PC2"]] * 100
    tabla_varianza = varianza.reset_index()
    tabla_varianza.columns = ["eje", "porcentaje_varianza"]

    print("\nVarianza explicada por el PCoA:")
    print(tabla_varianza.round(2).to_string(index=False))

    ruta = os.path.join(OUTPUT_DIR, "h2_varianza_explicada_pcoa.tsv")
    tabla_varianza.to_csv(ruta, sep="\t", index=False)
    print(f"Guardado: {ruta}")
    return tabla_varianza


def permanova_variable_continua(distancias, predictor, n_perm=999, semilla=0):
    """PERMANOVA (McArdle & Anderson 2001) de una distancia
    Bray-Curtis contra una variable ambiental continua.

    Reparte la variacion total de las distancias entre muestras
    en la parte explicada por el predictor (F, R2) y evalua su
    significancia permutando el predictor 999 veces.
    """
    presentes = ~predictor.isna().to_numpy()
    ids = predictor.index[presentes]
    n = len(ids)
    matriz = distancias.filter(ids).data
    x = predictor.loc[ids].to_numpy(dtype=float)

    # Matriz de Gower centrada: representa la variacion total de
    # las distancias respecto al centroide de todas las muestras.
    productos_internos = -0.5 * (matriz ** 2)
    unos = np.ones((n, 1))
    centrado = np.eye(n) - unos @ unos.T / n
    gower = centrado @ productos_internos @ centrado
    ss_total = np.trace(gower)

    def pseudo_f(valores):
        valores = valores - valores.mean()
        ss_modelo = (valores @ gower @ valores) / (valores @ valores)
        ss_residual = ss_total - ss_modelo
        df_modelo = 1
        df_residual = n - 2
        f = (ss_modelo / df_modelo) / (ss_residual / df_residual)
        return f, ss_modelo

    f_obs, ss_modelo_obs = pseudo_f(x)
    r2 = ss_modelo_obs / ss_total

    generador = np.random.default_rng(semilla)
    mayores_o_iguales = 0
    for _ in range(n_perm):
        f_perm, _ = pseudo_f(generador.permutation(x))
        if f_perm >= f_obs:
            mayores_o_iguales += 1
    p_valor = (mayores_o_iguales + 1) / (n_perm + 1)

    return f_obs, r2, p_valor, n


def permanova_variables_ambientales(abundancias, metadata):
    """PERMANOVA univariada de humedad, temperatura y elevacion
    contra la composicion microbiana (Bray-Curtis), ordenada de
    mayor a menor R2."""
    distancias = beta_diversity(
        "braycurtis",
        abundancias.to_numpy(dtype=float),
        ids=abundancias.index,
    )

    variables = [
        ("humedad_relativa", "average-soil-relative-humidity"),
        ("temperatura", "average-soil-temperature"),
        ("elevacion", "elevation"),
    ]

    filas = []
    for nombre, columna in variables:
        n_validos = metadata[columna].dropna().shape[0]
        if n_validos < 10:
            print(
                f"  Aviso: {nombre} tiene menos de 10 muestras "
                "validas; las permutaciones pueden ser insuficientes."
            )
        f, r2, p, n = permanova_variable_continua(
            distancias, metadata[columna]
        )
        filas.append(
            {
                "variable": nombre,
                "columna_original": columna,
                "F": f,
                "R2": r2,
                "p_valor": p,
                "n_permutaciones": 999,
                "n": n,
            }
        )

    tabla = pd.DataFrame(filas).sort_values("R2", ascending=False)

    print("\nPERMANOVA de variables ambientales (Bray-Curtis):")
    print(tabla.round(4).to_string(index=False))

    for _, fila in tabla.iterrows():
        if not (0.20 <= fila["R2"] <= 0.50) and (
            fila["variable"] == "humedad_relativa"
        ):
            print(
                "  Aviso: R2 de humedad_relativa fuera del rango "
                "esperado por H3 (0.20-0.50). Revisar antes de "
                "interpretar como confirmacion fuerte de H3."
            )

    ruta = os.path.join(
        OUTPUT_DIR, "h3_permanova_variables_ambientales.tsv"
    )
    tabla.to_csv(ruta, sep="\t", index=False)
    print(f"Guardado: {ruta}")
    return tabla


def graficar_r2_permanova(tabla):
    """Grafico de barras con el R2 de cada variable ambiental,
    para comparar de un vistazo cuanta varianza composicional
    explica cada una."""
    etiquetas = {
        "humedad_relativa": "Soil relative\nhumidity",
        "temperatura": "Soil\ntemperature",
        "elevacion": "Elevation",
    }
    datos = tabla.sort_values("R2", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    colores = sns.color_palette("Set2", n_colors=len(datos))
    barras = ax.bar(
        [etiquetas[v] for v in datos["variable"]],
        datos["R2"],
        color=colores,
    )

    for barra, (_, fila) in zip(barras, datos.iterrows()):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height(),
            f"p = {fila['p_valor']:.3f}",
            ha="center", va="bottom",
        )

    ax.set_ylabel("R² (variance in composition explained)")
    ax.set_title(
        "PERMANOVA: variance explained by environmental variables"
    )
    ax.set_ylim(0, max(datos["R2"]) * 1.3)

    for ext in ("png", "pdf"):
        ruta = os.path.join(
            OUTPUT_DIR, f"fig3_permanova_r2_por_variable.{ext}"
        )
        fig.savefig(ruta, dpi=300, bbox_inches="tight")
        print(f"Guardado: {ruta}")
    plt.close(fig)


def main():
    abundancias, metadata = cargar_datos()
    shannon = calcular_shannon(abundancias)

    tabla = metadata.join(shannon)

    resumen_por_transecto(tabla)
    boxplot_por_transecto(tabla)
    correlacion_shannon_humedad(tabla)

    resultado_pcoa = calcular_pcoa_bray_curtis(abundancias)
    graficar_pcoa(tabla, resultado_pcoa)
    varianza_explicada_pcoa(resultado_pcoa)

    tabla_permanova = permanova_variables_ambientales(
        abundancias, metadata
    )
    graficar_r2_permanova(tabla_permanova)


if __name__ == "__main__":
    main()
