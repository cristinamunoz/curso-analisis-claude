"""Analisis de composicion (beta-diversidad): H2 y H3.

H2: PCoA con distancias Bray-Curtis, coloreado por AvgSoilRH, mas
    PERMANOVA categorica (Baquedano vs. Yungay) para probar la
    separacion por transecto con un p-valor formal.
H3: PERMANOVA univariada de humedad, temperatura y elevacion
    sobre la matriz de distancias Bray-Curtis.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

RUTA_ABUNDANCIAS = "data/abundancias.tsv"
RUTA_METADATA = "data/metadata.tsv"
CARPETA_SALIDA = "outputs"
N_PERMUTACIONES = 999
SEMILLA = 42

VARIABLES_AMBIENTALES = [
    ("humedad_relativa", "average-soil-relative-humidity"),
    ("temperatura", "average-soil-temperature"),
    ("elevacion", "elevation"),
]


def cargar_datos():
    abundancias = pd.read_csv(
        RUTA_ABUNDANCIAS, sep="\t", index_col=0
    ).T
    metadata = pd.read_csv(RUTA_METADATA, sep="\t")
    ids_comunes = metadata["sample-id"].isin(abundancias.index)
    metadata = metadata.loc[ids_comunes].copy()
    abundancias = abundancias.loc[metadata["sample-id"]]
    return abundancias, metadata


def pcoa_clasico(matriz_distancias):
    """PCoA clasico (Gower) a partir de una matriz de distancias.

    1) Se centra la matriz de distancias al cuadrado (matriz de
       Gower B).
    2) Se descompone en autovalores/autovectores.
    3) Los ejes son los autovectores escalados por sqrt(autovalor).
    4) El % de varianza de cada eje = autovalor / suma de
       autovalores positivos.
    """
    n = matriz_distancias.shape[0]
    D2 = matriz_distancias ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    autovalores, autovectores = np.linalg.eigh(B)
    orden = np.argsort(autovalores)[::-1]
    autovalores = autovalores[orden]
    autovectores = autovectores[:, orden]

    positivos = autovalores > 1e-8
    coordenadas = (
        autovectores[:, positivos]
        * np.sqrt(autovalores[positivos])
    )
    varianza_explicada = (
        autovalores[positivos] / autovalores[positivos].sum() * 100
    )
    return coordenadas, varianza_explicada


def permanova_continua(matriz_distancias, predictor, n_perm, rng):
    """PERMANOVA para un predictor continuo (McArdle & Anderson 2001).

    Se parte la matriz de Gower (B) en variacion explicada por el
    predictor (SS_modelo) y variacion residual (SS_residual),
    usando la matriz de proyeccion (hat matrix) del predictor.
    El p-valor se obtiene permutando el predictor n_perm veces.
    """
    n = matriz_distancias.shape[0]
    D2 = matriz_distancias ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J @ D2 @ J
    ss_total = np.trace(G)

    def estadistico_f(x):
        X = np.column_stack([np.ones(n), x])
        H = X @ np.linalg.pinv(X.T @ X) @ X.T
        ss_modelo = np.trace(H @ G)
        ss_residual = ss_total - ss_modelo
        f = (ss_modelo / 1) / (ss_residual / (n - 2))
        r2 = ss_modelo / ss_total
        return f, r2

    f_obs, r2_obs = estadistico_f(predictor)

    conteo = 0
    for _ in range(n_perm):
        permutado = rng.permutation(predictor)
        f_perm, _ = estadistico_f(permutado)
        if f_perm >= f_obs:
            conteo += 1
    p_valor = (conteo + 1) / (n_perm + 1)

    return f_obs, r2_obs, p_valor


def main():
    abundancias, metadata = cargar_datos()
    rng = np.random.default_rng(SEMILLA)

    # --- H2: PCoA ---
    distancias = squareform(
        pdist(abundancias.values, metric="braycurtis")
    )
    coordenadas, varianza = pcoa_clasico(distancias)

    tabla_varianza = pd.DataFrame({
        "eje": ["PC1", "PC2"],
        "porcentaje_varianza": varianza[:2],
    })
    print("Varianza explicada por PCoA:")
    print(tabla_varianza.to_string(index=False))
    tabla_varianza.to_csv(
        f"{CARPETA_SALIDA}/h2_varianza_explicada_pcoa.tsv",
        sep="\t", index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    dispersión = ax.scatter(
        coordenadas[:, 0], coordenadas[:, 1],
        c=metadata["average-soil-relative-humidity"],
        cmap="viridis", s=70, edgecolor="black", linewidth=0.3,
    )
    barra = fig.colorbar(dispersión, ax=ax)
    barra.set_label("Humedad relativa promedio del suelo (%)")
    ax.set_xlabel(f"PC1 ({varianza[0]:.1f}% de varianza)")
    ax.set_ylabel(f"PC2 ({varianza[1]:.1f}% de varianza)")
    ax.set_title(
        "PCoA (Bray-Curtis) de la composicion microbiana del suelo"
    )

    postit = (
        "Post-it: que es un PCoA con Bray-Curtis?\n"
        "Bray-Curtis mide, entre cada par de muestras, que tan\n"
        "distinta es su composicion de OTUs (0=identicas,\n"
        "1=no comparten ninguna especie). El PCoA toma esa tabla\n"
        "de 'distancias entre todos los pares' (54x54) y la\n"
        "aplana a 2 ejes (PC1, PC2) que se pueden graficar,\n"
        "preservando lo mejor posible esas distancias originales.\n"
        "El % de varianza de cada eje indica cuanto de la\n"
        "diferencia total entre muestras captura ese eje --aqui\n"
        "es bajo (8% y 6%) porque hay 1109 OTUs, mucha variacion\n"
        "no cabe en solo 2 dimensiones."
    )
    fig.text(
        0.5, -0.05, postit, ha="center", va="top", fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(
        f"{CARPETA_SALIDA}/fig2_pcoa_braycurtis.png",
        dpi=300, bbox_inches="tight",
    )
    fig.savefig(
        f"{CARPETA_SALIDA}/fig2_pcoa_braycurtis.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # --- H2: PERMANOVA categorica por transecto ---
    grupo_yungay = (metadata["transect-name"] == "Yungay").astype(
        float
    ).values
    f_transecto, r2_transecto, p_transecto = permanova_continua(
        distancias, grupo_yungay, N_PERMUTACIONES, rng
    )
    tabla_transecto = pd.DataFrame([{
        "variable": "transect-name (Baquedano=0, Yungay=1)",
        "F": f_transecto,
        "R2": r2_transecto,
        "p_valor": p_transecto,
        "n_permutaciones": N_PERMUTACIONES,
        "n": len(metadata),
    }])
    print("\nPERMANOVA categorica por transecto:")
    print(tabla_transecto.to_string(index=False))
    tabla_transecto.to_csv(
        f"{CARPETA_SALIDA}/h2_permanova_transecto.tsv",
        sep="\t", index=False,
    )

    # --- H3: PERMANOVA ---
    filas = []
    for nombre, columna in VARIABLES_AMBIENTALES:
        submeta = metadata.dropna(subset=[columna])
        subabund = abundancias.loc[submeta["sample-id"]]
        subdist = squareform(
            pdist(subabund.values, metric="braycurtis")
        )
        f, r2, p = permanova_continua(
            subdist, submeta[columna].values, N_PERMUTACIONES, rng
        )
        filas.append({
            "variable": nombre,
            "columna_original": columna,
            "F": f,
            "R2": r2,
            "p_valor": p,
            "n_permutaciones": N_PERMUTACIONES,
            "n": len(submeta),
        })

    tabla_permanova = pd.DataFrame(filas).sort_values(
        "R2", ascending=False
    )
    print("\nPERMANOVA por variable ambiental (ordenado por R2):")
    print(tabla_permanova.to_string(index=False))
    tabla_permanova.to_csv(
        f"{CARPETA_SALIDA}/h3_permanova_variables_ambientales.tsv",
        sep="\t", index=False,
    )

    n_min = tabla_permanova["n"].min()
    if n_min < 10:
        print(
            f"\nAVISO: n minimo = {n_min} (<10). Las "
            "permutaciones pueden ser insuficientes."
        )

    print(f"\nArchivos guardados en '{CARPETA_SALIDA}/'.")


if __name__ == "__main__":
    main()
