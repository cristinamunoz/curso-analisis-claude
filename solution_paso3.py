"""
Paso 3: PERMANOVA univariada -- que variable ambiental explica
mas varianza en la composicion microbiana?

Pregunta (H3): se espera que la humedad del suelo (AvgSoilRH)
explique mas varianza composicional (R2 esperado 0.20-0.50) que
la temperatura o la elevacion.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

N_PERMUTACIONES = 999
SEMILLA = 42

# --- 1. Cargar datos (igual que en pasos anteriores) -------------

abund = pd.read_csv(
    "data/abundancias.tsv", sep="\t", index_col=0
)
meta = pd.read_csv(
    "data/metadata.tsv", sep="\t", index_col="sample-id"
)

muestras_comunes = abund.columns.intersection(meta.index)
abund = abund[muestras_comunes]
meta = meta.loc[muestras_comunes]

# --- 2. Distancia Bray-Curtis (igual que en el paso 2) -----------

distancias = pdist(abund.T.values, metric="braycurtis")
dist_cuadrada = pd.DataFrame(
    squareform(distancias),
    index=abund.columns, columns=abund.columns,
)


# --- 3. PERMANOVA univariada (metodo de McArdle & Anderson) ------

def permanova_univariada(dist_df, variable, n_perm, semilla):
    """PERMANOVA para 1 variable ambiental continua.

    Compara cuanta varianza de las distancias Bray-Curtis explica
    esa variable, y usa permutaciones para calcular el p-valor
    (mismo metodo que 'adonis' en R / QIIME2).
    """
    datos = variable.dropna()
    muestras = dist_df.index.intersection(datos.index)
    d = dist_df.loc[muestras, muestras].values
    x = datos.loc[muestras].values
    n = len(muestras)

    d2 = d ** 2
    centrado = np.eye(n) - np.ones((n, n)) / n
    g = -0.5 * centrado @ d2 @ centrado  # doble centrado (Gower)
    ss_total = np.trace(g)

    def pseudo_f(x_vec):
        xc = (x_vec - x_vec.mean()).reshape(-1, 1)
        h = xc @ np.linalg.inv(xc.T @ xc) @ xc.T
        ss_reg = np.trace(h @ g)
        ss_res = ss_total - ss_reg
        f = (ss_reg / 1) / (ss_res / (n - 2))
        r2 = ss_reg / ss_total
        return f, r2

    f_obs, r2_obs = pseudo_f(x)

    rng = np.random.default_rng(semilla)
    conteo = 0
    for _ in range(n_perm):
        x_perm = rng.permutation(x)
        f_perm, _ = pseudo_f(x_perm)
        if f_perm >= f_obs:
            conteo += 1
    p_valor = (conteo + 1) / (n_perm + 1)

    return f_obs, r2_obs, p_valor, n


variables = {
    "humedad_relativa": "average-soil-relative-humidity",
    "temperatura": "average-soil-temperature",
    "elevacion": "elevation",
}

filas = []
for nombre, columna in variables.items():
    f, r2, p, n = permanova_univariada(
        dist_cuadrada, meta[columna], N_PERMUTACIONES, SEMILLA
    )
    filas.append({
        "variable": nombre,
        "columna_original": columna,
        "F": f,
        "R2": r2,
        "p_valor": p,
        "n_permutaciones": N_PERMUTACIONES,
        "n": n,
    })

tabla = pd.DataFrame(filas).sort_values("R2", ascending=False)
tabla.to_csv(
    "outputs/h3_permanova_variables_ambientales.tsv",
    sep="\t", index=False,
)

print("PERMANOVA univariada (ordenada por R2 descendente):")
print(tabla.round(4).to_string(index=False))

if (tabla["n"] < 10).any():
    print(
        "\nAVISO: alguna variable tiene n < 10 -- las "
        "permutaciones pueden ser insuficientes."
    )
