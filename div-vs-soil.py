"""Diversidad alfa (Shannon) vs. humedad relativa del suelo.

Reproduce parte del analisis de Neilson et al. (2017): calcula el
indice de Shannon por muestra a partir de las abundancias de OTU,
lo compara entre transectos (Baquedano vs. Yungay) y lo
correlaciona con la humedad relativa promedio del suelo
(average-soil-relative-humidity), usando correlacion de Spearman
como en el paper original.

Genera:
- outputs/fig1_shannon_por_transecto.png / .pdf
- outputs/fig1_shannon_vs_avgsoilrh.png / .pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RUTA_ABUNDANCIAS = "data/abundancias.tsv"
RUTA_METADATA = "data/metadata.tsv"
CARPETA_SALIDA = "outputs"
COLUMNA_HUMEDAD = "average-soil-relative-humidity"
COLUMNA_TRANSECTO = "transect-name"


def calcular_shannon(abundancias):
    """Calcula el indice de Shannon (base natural) por muestra.

    `abundancias` tiene una fila por OTU y una columna por
    muestra. Para cada muestra se calcula la proporcion de cada
    OTU sobre el total y luego H = -suma(p_i * ln(p_i)).
    """
    proporciones = abundancias.div(abundancias.sum(axis=0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terminos = proporciones * np.log(proporciones)
    terminos = terminos.fillna(0.0)
    shannon = -terminos.sum(axis=0)
    return shannon.rename("shannon")


def cargar_datos():
    abundancias = pd.read_csv(
        RUTA_ABUNDANCIAS, sep="\t", index_col=0
    )
    metadata = pd.read_csv(RUTA_METADATA, sep="\t")

    shannon = calcular_shannon(abundancias)

    datos = metadata.merge(
        shannon, left_on="sample-id", right_index=True, how="inner"
    )
    return datos


def resumen_por_transecto(datos):
    resumen = datos.groupby(COLUMNA_TRANSECTO)["shannon"].agg(
        n="count", media="mean", mediana="median",
        minimo="min", maximo="max",
    )
    print("\nResumen de Shannon por transecto:")
    print(resumen.round(2))
    return resumen


def graficar_boxplot(datos):
    transectos = sorted(datos[COLUMNA_TRANSECTO].unique())
    colores = plt.get_cmap("Set2").colors

    fig, ax = plt.subplots(figsize=(8, 6))
    grupos = [
        datos.loc[datos[COLUMNA_TRANSECTO] == t, "shannon"]
        for t in transectos
    ]
    cajas = ax.boxplot(
        grupos, tick_labels=transectos, patch_artist=True,
    )
    for caja, color in zip(cajas["boxes"], colores):
        caja.set_facecolor(color)

    ax.set_xlabel("Transecto")
    ax.set_ylabel("Indice de Shannon")
    ax.set_title(
        "Diversidad alfa (Shannon) por transecto\n"
        "Desierto de Atacama (Neilson et al. 2017)"
    )
    fig.tight_layout()

    fig.savefig(
        f"{CARPETA_SALIDA}/fig1_shannon_por_transecto.png", dpi=300
    )
    fig.savefig(f"{CARPETA_SALIDA}/fig1_shannon_por_transecto.pdf")
    plt.close(fig)


def graficar_regresion(datos):
    datos_validos = datos.dropna(subset=[COLUMNA_HUMEDAD])
    excluidas = len(datos) - len(datos_validos)
    if excluidas:
        print(
            f"\nAVISO: {excluidas} muestra(s) sin dato de humedad "
            "del suelo fueron excluidas de la correlacion."
        )

    x = datos_validos[COLUMNA_HUMEDAD].to_numpy()
    y = datos_validos["shannon"].to_numpy()
    n = len(x)

    rho, p_spearman = stats.spearmanr(x, y)

    pendiente, intercepto, r_valor, p_lineal, _ = stats.linregress(
        x, y
    )
    r2_lineal = r_valor ** 2

    x_linea = np.linspace(x.min(), x.max(), 100)
    y_linea = pendiente * x_linea + intercepto

    residuales = y - (pendiente * x + intercepto)
    gl = n - 2
    error_cuadratico = np.sum(residuales ** 2) / gl
    x_media = x.mean()
    sxx = np.sum((x - x_media) ** 2)
    error_estandar = np.sqrt(
        error_cuadratico * (1 / n + (x_linea - x_media) ** 2 / sxx)
    )
    t_critico = stats.t.ppf(0.975, gl)
    banda = t_critico * error_estandar

    fig, ax = plt.subplots(figsize=(8, 6))
    colores_viridis = plt.get_cmap("viridis")
    dispersion = ax.scatter(
        x, y, c=x, cmap=colores_viridis, edgecolor="black",
        linewidth=0.5, zorder=3,
    )
    ax.plot(x_linea, y_linea, color="black", linewidth=1.5,
            zorder=2)
    ax.fill_between(
        x_linea, y_linea - banda, y_linea + banda,
        color="gray", alpha=0.3, zorder=1,
        label="Intervalo de confianza 95%",
    )

    cbar = fig.colorbar(dispersion, ax=ax)
    cbar.set_label("Humedad relativa del suelo (%)")

    texto = (
        f"Spearman ρ = {rho:.2f}, p = {p_spearman:.3g}\n"
        f"R² (lineal) = {r2_lineal:.2f}, p = {p_lineal:.3g}\n"
        f"n = {n}"
    )
    ax.text(
        0.03, 0.97, texto, transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Humedad relativa promedio del suelo (%)")
    ax.set_ylabel("Indice de Shannon")
    ax.set_title(
        "Diversidad alfa vs. humedad relativa del suelo\n"
        "Desierto de Atacama (Neilson et al. 2017)"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()

    fig.savefig(
        f"{CARPETA_SALIDA}/fig1_shannon_vs_avgsoilrh.png", dpi=300
    )
    fig.savefig(f"{CARPETA_SALIDA}/fig1_shannon_vs_avgsoilrh.pdf")
    plt.close(fig)

    return {
        "rho_spearman": rho, "p_spearman": p_spearman,
        "r2_lineal": r2_lineal, "p_lineal": p_lineal, "n": n,
    }


def main():
    datos = cargar_datos()
    print(f"Muestras cruzadas (abundancia + metadata): {len(datos)}")

    resumen_por_transecto(datos)
    graficar_boxplot(datos)
    resultado = graficar_regresion(datos)

    print("\nCorrelacion Shannon vs. humedad relativa del suelo:")
    print(f"  Spearman rho = {resultado['rho_spearman']:.3f}")
    print(f"  p-valor (Spearman) = {resultado['p_spearman']:.4g}")
    print(f"  R2 (lineal) = {resultado['r2_lineal']:.3f}")
    print(f"  p-valor (lineal) = {resultado['p_lineal']:.4g}")
    print(f"  n = {resultado['n']}")

    if resultado["r2_lineal"] < 0.05:
        print(
            "\nAVISO: R2 < 0.05. La relacion lineal es muy debil. "
            "Esto puede indicar ruido en los datos, un efecto real "
            "pero pequeno, o que la relacion no es lineal. Revisa "
            "los resultados antes de sacar conclusiones."
        )

    print(f"\nFiguras guardadas en '{CARPETA_SALIDA}/'.")


if __name__ == "__main__":
    main()
