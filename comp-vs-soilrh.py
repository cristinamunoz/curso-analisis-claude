"""Composicion microbiana (PCoA Bray-Curtis) vs. humedad del suelo.

Reproduce la parte de composicion (beta-diversidad) del analisis
de Neilson et al. (2017): calcula distancias Bray-Curtis entre
muestras a partir de las abundancias de OTU, hace una ordenacion
PCoA para visualizar, y prueba con PERMANOVA (999 permutaciones)
si los transectos mas aridos (Yungay) y menos aridos (Baquedano)
tienen comunidades composicionalmente distintas (H2).

Genera:
- outputs/fig2_pcoa_braycurtis.png / .pdf
- outputs/h2_permanova_transecto.tsv
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
N_PERMUTACIONES = 999
SEMILLA_ALEATORIA = 0


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


def calcular_distancias(abundancias):
    matriz_muestras = abundancias.T.to_numpy()
    return squareform(pdist(matriz_muestras, metric="braycurtis"))


def calcular_pcoa(distancias):
    """Hace una ordenacion PCoA clasica sobre una matriz de
    distancias (metodo de Gower: doble centrado de la matriz de
    distancias al cuadrado y descomposicion en autovalores).
    """
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
    fig.tight_layout()

    fig.savefig(
        f"{CARPETA_SALIDA}/fig2_pcoa_braycurtis.png", dpi=300
    )
    fig.savefig(f"{CARPETA_SALIDA}/fig2_pcoa_braycurtis.pdf")
    plt.close(fig)


def _suma_cuadrados_intra(distancias_cuad, etiquetas):
    """Suma, para cada grupo, la suma de distancias (al cuadrado)
    entre sus propias muestras dividida por su tamano. Es la
    pieza que compara PERMANOVA contra la suma de cuadrados total
    para saber cuanta variacion queda "dentro" de cada grupo.
    """
    suma = 0.0
    for grupo in np.unique(etiquetas):
        indices = np.where(etiquetas == grupo)[0]
        n_grupo = len(indices)
        submatriz = distancias_cuad[np.ix_(indices, indices)]
        pares = submatriz[np.triu_indices(n_grupo, k=1)]
        suma += pares.sum() / n_grupo
    return suma


def permanova(distancias, etiquetas):
    """PERMANOVA de una via (Anderson, 2001): prueba si las
    distancias Bray-Curtis entre muestras del mismo grupo
    (transecto) son en promedio mas chicas que entre muestras de
    grupos distintos, usando permutaciones para el p-valor en vez
    de asumir una distribucion teorica.
    """
    etiquetas = np.asarray(etiquetas)
    n = distancias.shape[0]
    k = len(np.unique(etiquetas))
    distancias_cuad = distancias ** 2

    pares_todos = distancias_cuad[np.triu_indices(n, k=1)]
    ss_total = pares_todos.sum() / n

    ss_intra_obs = _suma_cuadrados_intra(distancias_cuad, etiquetas)
    ss_entre_obs = ss_total - ss_intra_obs
    f_obs = (ss_entre_obs / (k - 1)) / (ss_intra_obs / (n - k))
    r2 = ss_entre_obs / ss_total

    generador = np.random.default_rng(SEMILLA_ALEATORIA)
    conteo_iguales_o_mayores = 0
    for _ in range(N_PERMUTACIONES):
        permutadas = generador.permutation(etiquetas)
        ss_intra_p = _suma_cuadrados_intra(
            distancias_cuad, permutadas
        )
        ss_entre_p = ss_total - ss_intra_p
        f_p = (ss_entre_p / (k - 1)) / (ss_intra_p / (n - k))
        if f_p >= f_obs:
            conteo_iguales_o_mayores += 1

    p_valor = (conteo_iguales_o_mayores + 1) / (N_PERMUTACIONES + 1)
    return f_obs, r2, p_valor, n


def revisar_separacion_transectos(distancias, metadata):
    etiquetas = metadata[COLUMNA_TRANSECTO].to_numpy()
    tamanos = metadata.groupby(COLUMNA_TRANSECTO).size()
    if (tamanos < 10).any():
        print(
            "\nAVISO: hay un transecto con menos de 10 muestras; "
            "las permutaciones del PERMANOVA pueden ser poco "
            "confiables."
        )

    f_obs, r2, p_valor, n = permanova(distancias, etiquetas)

    print("\nPERMANOVA por transecto (Bray-Curtis, 999 "
          "permutaciones):")
    print(f"  F = {f_obs:.3f}")
    print(f"  R2 = {r2:.4f}")
    print(f"  p-valor = {p_valor:.4f}")
    print(f"  n = {n}")

    tabla = pd.DataFrame([{
        "variable": "transecto",
        "columna_original": COLUMNA_TRANSECTO,
        "F": f_obs,
        "R2": r2,
        "p_valor": p_valor,
        "n_permutaciones": N_PERMUTACIONES,
        "n": n,
    }])
    tabla.to_csv(
        f"{CARPETA_SALIDA}/h2_permanova_transecto.tsv",
        sep="\t", index=False,
    )
    return r2, p_valor


def main():
    abundancias, metadata = cargar_datos()
    print(f"Muestras cruzadas (abundancia + metadata): "
          f"{len(metadata)}")

    distancias = calcular_distancias(abundancias)
    pc1, pc2, var_pc1, var_pc2 = calcular_pcoa(distancias)

    print("\nVarianza explicada (ejes de la ordenacion):")
    print(f"  PC1 = {var_pc1:.2f}%")
    print(f"  PC2 = {var_pc2:.2f}%")

    r2, p_valor = revisar_separacion_transectos(distancias, metadata)
    graficar_pcoa(pc1, pc2, var_pc1, var_pc2, metadata)

    if r2 < 0.05:
        print(
            "\nAVISO: R2 < 0.05. La composicion casi no difiere "
            "entre transectos; revisa si tiene sentido biologico "
            "antes de seguir."
        )

    print(f"\nFigura y tabla guardadas en '{CARPETA_SALIDA}/'.")


if __name__ == "__main__":
    main()
