"""Diversidad alfa (Shannon) vs. aridez del suelo (Atacama).

Pregunta biologica: a medida que el suelo se vuelve mas arido,
como cambia la diversidad de la comunidad microbiana?

Hipotesis H1: a menor humedad relativa del suelo (AvgSoilRH),
menor diversidad alfa (Shannon).
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

RUTA_METADATA = "data/metadata.tsv"
RUTA_ABUNDANCIAS = "data/abundancias.tsv"
CARPETA_SALIDA = "outputs"

COLUMNA_HUMEDAD = "average-soil-relative-humidity"
COLUMNA_TRANSECTO = "transect-name"


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


if __name__ == "__main__":
    main()
