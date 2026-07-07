"""PERMANOVA univariado de variables ambientales (H3).

Reproduce la parte de H3 del proyecto: para cada variable
ambiental (humedad relativa, temperatura, elevacion), prueba con
PERMANOVA si esa variable, por si sola, explica parte de la
variacion en composicion microbiana (distancias Bray-Curtis)
entre las 54 muestras. A diferencia del PERMANOVA por transecto
de H2 (una variable categorica con 2 grupos), aqui cada variable
es continua, asi que se usa el metodo de McArdle & Anderson
(2001): una regresion sobre la matriz de distancias en vez de
comparar grupos.

Genera:
- outputs/h3_permanova_variables_ambientales.tsv
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

RUTA_ABUNDANCIAS = "data/abundancias.tsv"
RUTA_METADATA = "data/metadata.tsv"
CARPETA_SALIDA = "outputs"
N_PERMUTACIONES = 999
SEMILLA_ALEATORIA = 0

COLUMNAS_AMBIENTALES = {
    "humedad_relativa": "average-soil-relative-humidity",
    "temperatura": "average-soil-temperature",
    "elevacion": "elevation",
}


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


def _suma_cuadrados_modelo(matriz_centrada, x):
    """Suma de cuadrados "explicada" por la variable x, proyectando
    la matriz de distancias centrada (Gower) sobre la matriz
    sombrero (hat matrix) de una regresion con intercepto y x.
    """
    n = len(x)
    diseno = np.column_stack([np.ones(n), x])
    sombrero = diseno @ np.linalg.pinv(diseno.T @ diseno) @ diseno.T
    return np.trace(sombrero @ matriz_centrada @ sombrero)


def permanova_continuo(distancias, valores):
    """PERMANOVA de una variable ambiental continua (McArdle &
    Anderson, 2001): reparte la variacion total en composicion en
    la parte explicada linealmente por la variable y el resto, y
    usa permutaciones para el p-valor.
    """
    validos = ~np.isnan(valores)
    d = distancias[np.ix_(validos, validos)]
    x = valores[validos]
    n = d.shape[0]

    identidad = np.eye(n)
    unos = np.ones((n, n)) / n
    centrado = identidad - unos
    matriz_centrada = -0.5 * centrado @ (d ** 2) @ centrado
    ss_total = np.trace(matriz_centrada)

    gl_modelo = 1
    gl_residual = n - 2

    ss_modelo_obs = _suma_cuadrados_modelo(matriz_centrada, x)
    ss_residual_obs = ss_total - ss_modelo_obs
    f_obs = (ss_modelo_obs / gl_modelo) / (
        ss_residual_obs / gl_residual
    )
    r2 = ss_modelo_obs / ss_total

    generador = np.random.default_rng(SEMILLA_ALEATORIA)
    conteo_iguales_o_mayores = 0
    for _ in range(N_PERMUTACIONES):
        x_permutado = generador.permutation(x)
        ss_modelo_p = _suma_cuadrados_modelo(
            matriz_centrada, x_permutado
        )
        ss_residual_p = ss_total - ss_modelo_p
        f_p = (ss_modelo_p / gl_modelo) / (
            ss_residual_p / gl_residual
        )
        if f_p >= f_obs:
            conteo_iguales_o_mayores += 1

    p_valor = (conteo_iguales_o_mayores + 1) / (N_PERMUTACIONES + 1)
    return f_obs, r2, p_valor, n


def main():
    abundancias, metadata = cargar_datos()
    print(f"Muestras cruzadas (abundancia + metadata): "
          f"{len(metadata)}")

    distancias = calcular_distancias(abundancias)

    filas = []
    for nombre, columna in COLUMNAS_AMBIENTALES.items():
        valores = metadata[columna].to_numpy(dtype=float)
        excluidas = int(np.isnan(valores).sum())
        if excluidas:
            print(
                f"\nAVISO: {excluidas} muestra(s) sin dato de "
                f"'{columna}' fueron excluidas para esa variable."
            )
        f_obs, r2, p_valor, n = permanova_continuo(
            distancias, valores
        )
        filas.append({
            "variable": nombre,
            "columna_original": columna,
            "F": f_obs,
            "R2": r2,
            "p_valor": p_valor,
            "n_permutaciones": N_PERMUTACIONES,
            "n": n,
        })

    tabla = pd.DataFrame(filas).sort_values(
        "R2", ascending=False
    ).reset_index(drop=True)

    print("\nPERMANOVA por variable ambiental (Bray-Curtis, 999 "
          "permutaciones), ordenado de mayor a menor R2:")
    print(tabla.round(4).to_string(index=False))

    tabla.to_csv(
        f"{CARPETA_SALIDA}/h3_permanova_variables_ambientales.tsv",
        sep="\t", index=False,
    )
    print(f"\nTabla guardada en '{CARPETA_SALIDA}/"
          f"h3_permanova_variables_ambientales.tsv'.")


if __name__ == "__main__":
    main()
