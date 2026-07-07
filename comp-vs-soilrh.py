"""Composicion microbiana (PCoA Bray-Curtis) vs. humedad del suelo.

Reproduce la parte de composicion (beta-diversidad) del analisis
de Neilson et al. (2017): calcula distancias Bray-Curtis entre
muestras a partir de las abundancias de OTU, hace una ordenacion
PCoA y verifica si los transectos mas aridos (Yungay) se separan
de los menos aridos (Baquedano) a lo largo del primer eje (H2).

Genera:
- outputs/fig2_pcoa_braycurtis.png / .pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

RUTA_ABUNDANCIAS = "data/abundancias.tsv"
RUTA_METADATA = "data/metadata.tsv"
CARPETA_SALIDA = "outputs"
COLUMNA_HUMEDAD = "average-soil-relative-humidity"
COLUMNA_TRANSECTO = "transect-name"


def cargar_datos():
    abundancias = pd.read_csv(
        RUTA_ABUNDANCIAS, sep="\t", index_col=0
    )
    metadata = pd.read_csv(RUTA_METADATA, sep="\t")

    muestras_comunes = [
        m for m in abundancias.columns
        if m in set(metadata["sample-id"])
    ]
    abundancias = abundancias[muestras_comunes]
    metadata = (
        metadata.set_index("sample-id")
        .loc[muestras_comunes]
        .reset_index()
    )
    return abundancias, metadata


def calcular_pcoa(abundancias):
    """Hace una ordenacion PCoA clasica sobre distancias
    Bray-Curtis (metodo de Gower: doble centrado de la matriz de
    distancias al cuadrado y descomposicion en autovalores).
    """
    matriz_muestras = abundancias.T.to_numpy()
    distancias = squareform(
        pdist(matriz_muestras, metric="braycurtis")
    )

    n = distancias.shape[0]
    identidad = np.eye(n)
    unos = np.ones((n, n)) / n
    centrado = identidad - unos
    b = -0.5 * centrado @ (distancias ** 2) @ centrado

    autovalores, autovectores = np.linalg.eigh(b)
    orden = np.argsort(autovalores)[::-1]
    autovalores = autovalores[orden]
    autovectores = autovectores[:, orden]

    varianza_total = autovalores[autovalores > 0].sum()
    porcentaje_var = 100 * autovalores / varianza_total

    coordenadas = autovectores * np.sqrt(np.abs(autovalores))
    pc1 = coordenadas[:, 0]
    pc2 = coordenadas[:, 1]

    return pc1, pc2, porcentaje_var[0], porcentaje_var[1]


def graficar_pcoa(pc1, pc2, var_pc1, var_pc2, metadata):
    humedad = metadata[COLUMNA_HUMEDAD].to_numpy()
    sin_dato = np.isnan(humedad)

    fig, ax = plt.subplots(figsize=(8, 6))

    if sin_dato.any():
        ax.scatter(
            pc1[sin_dato], pc2[sin_dato], c="lightgray",
            edgecolor="black", linewidth=0.5,
            label="Sin dato de humedad",
        )

    dispersion = ax.scatter(
        pc1[~sin_dato], pc2[~sin_dato], c=humedad[~sin_dato],
        cmap=plt.get_cmap("viridis"), edgecolor="black",
        linewidth=0.5,
    )
    cbar = fig.colorbar(dispersion, ax=ax)
    cbar.set_label("Humedad relativa promedio del suelo (%)")

    ax.set_xlabel(f"PC1 ({var_pc1:.1f}% varianza explicada)")
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}% varianza explicada)")
    ax.set_title(
        "PCoA (Bray-Curtis) de composicion microbiana\n"
        "Desierto de Atacama (Neilson et al. 2017)"
    )
    if sin_dato.any():
        ax.legend(loc="best")
    fig.tight_layout()

    fig.savefig(
        f"{CARPETA_SALIDA}/fig2_pcoa_braycurtis.png", dpi=300
    )
    fig.savefig(f"{CARPETA_SALIDA}/fig2_pcoa_braycurtis.pdf")
    plt.close(fig)


def revisar_separacion_transectos(pc1, metadata):
    metadata = metadata.copy()
    metadata["pc1"] = pc1
    resumen = metadata.groupby(COLUMNA_TRANSECTO)["pc1"].agg(
        n="count", media="mean", mediana="median",
    )
    print("\nPC1 por transecto (chequeo de H2):")
    print(resumen.round(3))


def main():
    abundancias, metadata = cargar_datos()
    print(f"Muestras cruzadas (abundancia + metadata): "
          f"{len(metadata)}")

    pc1, pc2, var_pc1, var_pc2 = calcular_pcoa(abundancias)

    print("\nVarianza explicada:")
    print(f"  PC1 = {var_pc1:.2f}%")
    print(f"  PC2 = {var_pc2:.2f}%")

    revisar_separacion_transectos(pc1, metadata)
    graficar_pcoa(pc1, pc2, var_pc1, var_pc2, metadata)

    print(f"\nFigura guardada en '{CARPETA_SALIDA}/'.")


if __name__ == "__main__":
    main()
