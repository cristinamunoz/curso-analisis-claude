"""Analisis multivariado: PERMANOVA combinado, Mantel y GLM.

Extiende H1/H3 combinando las 3 variables ambientales (humedad,
temperatura, elevacion) a la vez, en vez de probarlas una por
una, siguiendo el enfoque BEST/Mantel del paper de Neilson et al.
(2017).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform

RUTA_ABUNDANCIAS = "data/abundancias.tsv"
RUTA_METADATA = "data/metadata.tsv"
CARPETA_SALIDA = "outputs"
N_PERMUTACIONES = 999
SEMILLA = 42

VARIABLES = {
    "humedad_relativa": "average-soil-relative-humidity",
    "temperatura": "average-soil-temperature",
    "elevacion": "elevation",
}


def cargar_datos():
    abundancias = pd.read_csv(
        RUTA_ABUNDANCIAS, sep="\t", index_col=0
    ).T
    metadata = pd.read_csv(RUTA_METADATA, sep="\t")
    ids_comunes = metadata["sample-id"].isin(abundancias.index)
    metadata = metadata.loc[ids_comunes].copy()
    abundancias = abundancias.loc[metadata["sample-id"]]
    return abundancias, metadata


def calcular_shannon(abundancias):
    proporciones = abundancias.div(abundancias.sum(axis=1), axis=0)
    log_proporciones = np.log(proporciones.where(proporciones > 0))
    return -(proporciones * log_proporciones).sum(axis=1)


def gower_centrado(matriz_distancias):
    n = matriz_distancias.shape[0]
    D2 = matriz_distancias ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ D2 @ J


def ss_modelo(X, G):
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    return np.trace(H @ G)


def permanova_multivariado(G, predictores, n_perm, rng):
    """PERMANOVA con varias variables ambientales a la vez.

    Reporta el R2 combinado del modelo completo (las 3 variables
    juntas) y el aporte unico de cada variable (R2 del modelo
    completo menos el R2 del modelo sin esa variable), cada uno
    con su propio p-valor por permutacion.
    """
    n = G.shape[0]
    ss_total = np.trace(G)
    nombres = list(predictores.columns)
    X_completo = np.column_stack(
        [np.ones(n)] + [predictores[c].values for c in nombres]
    )
    ss_completo = ss_modelo(X_completo, G)
    df_modelo = len(nombres)
    df_residual = n - df_modelo - 1
    f_completo = (
        (ss_completo / df_modelo)
        / ((ss_total - ss_completo) / df_residual)
    )
    r2_completo = ss_completo / ss_total

    conteo = 0
    for _ in range(n_perm):
        orden = rng.permutation(n)
        X_perm = np.column_stack(
            [np.ones(n)]
            + [predictores[c].values[orden] for c in nombres]
        )
        if ss_modelo(X_perm, G) >= ss_completo:
            conteo += 1
    p_completo = (conteo + 1) / (n_perm + 1)

    filas_unicas = []
    for nombre in nombres:
        otras = [c for c in nombres if c != nombre]
        X_reducido = np.column_stack(
            [np.ones(n)] + [predictores[c].values for c in otras]
        )
        ss_reducido = ss_modelo(X_reducido, G)
        r2_unico = (ss_completo - ss_reducido) / ss_total

        conteo_var = 0
        valores_originales = predictores[nombre].values
        for _ in range(n_perm):
            permutado = rng.permutation(valores_originales)
            X_perm = np.column_stack(
                [np.ones(n)]
                + [
                    predictores[c].values if c != nombre
                    else permutado
                    for c in nombres
                ]
            )
            ss_perm = ss_modelo(X_perm, G) - ss_reducido
            if ss_perm >= (ss_completo - ss_reducido):
                conteo_var += 1
        p_unico = (conteo_var + 1) / (n_perm + 1)

        filas_unicas.append({
            "variable": nombre,
            "r2_unico": r2_unico,
            "p_valor": p_unico,
        })

    return r2_completo, f_completo, p_completo, filas_unicas


def mantel_test(matriz_a, matriz_b, n_perm, rng):
    """Test de Mantel: correlacion entre dos matrices de distancia.

    Se compara el triangulo superior de ambas matrices (aplanado)
    con una correlacion de Pearson, y se evalua significancia
    permutando las filas/columnas de una de las matrices.
    """
    n = matriz_a.shape[0]
    idx_triang = np.triu_indices(n, k=1)
    vector_a = matriz_a[idx_triang]
    vector_b_original = matriz_b[idx_triang]
    r_obs = np.corrcoef(vector_a, vector_b_original)[0, 1]

    conteo = 0
    for _ in range(n_perm):
        orden = rng.permutation(n)
        b_permutada = matriz_b[orden][:, orden]
        vector_b = b_permutada[idx_triang]
        r_perm = np.corrcoef(vector_a, vector_b)[0, 1]
        if r_perm >= r_obs:
            conteo += 1
    p_valor = (conteo + 1) / (n_perm + 1)
    return r_obs, p_valor


def main():
    abundancias, metadata = cargar_datos()
    rng = np.random.default_rng(SEMILLA)

    metadata = metadata.copy()
    metadata["shannon"] = calcular_shannon(abundancias).values

    columnas_env = list(VARIABLES.values())
    datos = metadata.dropna(
        subset=columnas_env + ["shannon"]
    ).reset_index(drop=True)
    subabund = abundancias.loc[datos["sample-id"]]
    n = len(datos)
    print(f"Muestras usadas (n={n}, con las 3 variables completas)")

    distancias = squareform(
        pdist(subabund.values, metric="braycurtis")
    )
    G = gower_centrado(distancias)

    predictores = datos[list(VARIABLES.values())].rename(
        columns={v: k for k, v in VARIABLES.items()}
    )

    # --- PERMANOVA multivariado ---
    r2_completo, f_completo, p_completo, filas_unicas = (
        permanova_multivariado(
            G, predictores, N_PERMUTACIONES, rng
        )
    )
    print("\nPERMANOVA multivariado (humedad + temperatura + "
          "elevacion, juntas):")
    print(f"  R2 combinado = {r2_completo:.4f}, F = "
          f"{f_completo:.3f}, p = {p_completo:.4f}")

    tabla_unicas = pd.DataFrame(filas_unicas)
    print("\nAporte unico de cada variable (R2 del modelo "
          "completo menos R2 sin esa variable):")
    print(tabla_unicas.to_string(index=False))

    resumen_permanova = pd.DataFrame([{
        "modelo": "combinado (humedad+temperatura+elevacion)",
        "R2": r2_completo,
        "F": f_completo,
        "p_valor": p_completo,
        "n_permutaciones": N_PERMUTACIONES,
        "n": n,
    }])
    resumen_permanova.to_csv(
        f"{CARPETA_SALIDA}/multivariado_permanova_combinado.tsv",
        sep="\t", index=False,
    )
    tabla_unicas.to_csv(
        f"{CARPETA_SALIDA}/multivariado_permanova_aporte_unico.tsv",
        sep="\t", index=False,
    )

    # --- Test de Mantel ---
    print("\nTest de Mantel (Bray-Curtis vs. distancia euclidiana "
          "de cada variable):")
    filas_mantel = []
    for nombre in VARIABLES:
        valores = predictores[nombre].values.reshape(-1, 1)
        dist_var = squareform(pdist(valores, metric="euclidean"))
        r_mantel, p_mantel = mantel_test(
            distancias, dist_var, N_PERMUTACIONES, rng
        )
        filas_mantel.append({
            "variable": nombre,
            "mantel_r": r_mantel,
            "p_valor": p_mantel,
        })
        print(f"  {nombre}: r = {r_mantel:.4f}, p = {p_mantel:.4f}")

    valores_std = (
        predictores - predictores.mean()
    ) / predictores.std()
    dist_combinada = squareform(
        pdist(valores_std.values, metric="euclidean")
    )
    r_mantel_comb, p_mantel_comb = mantel_test(
        distancias, dist_combinada, N_PERMUTACIONES, rng
    )
    filas_mantel.append({
        "variable": "combinada (las 3 estandarizadas)",
        "mantel_r": r_mantel_comb,
        "p_valor": p_mantel_comb,
    })
    print(f"  combinada: r = {r_mantel_comb:.4f}, "
          f"p = {p_mantel_comb:.4f}")

    tabla_mantel = pd.DataFrame(filas_mantel)
    tabla_mantel.to_csv(
        f"{CARPETA_SALIDA}/multivariado_mantel.tsv",
        sep="\t", index=False,
    )

    # --- GLM: Shannon ~ humedad + temperatura + elevacion ---
    X_glm = sm.add_constant(predictores)
    modelo_glm = sm.GLM(
        datos["shannon"], X_glm, family=sm.families.Gaussian()
    ).fit()
    print("\nGLM (familia Gaussiana): shannon ~ humedad + "
          "temperatura + elevacion")
    print(modelo_glm.summary())

    predicciones = modelo_glm.predict(X_glm)
    ss_res = ((datos["shannon"] - predicciones) ** 2).sum()
    ss_tot = (
        (datos["shannon"] - datos["shannon"].mean()) ** 2
    ).sum()
    r2_glm = 1 - ss_res / ss_tot

    tabla_glm = pd.DataFrame({
        "variable": modelo_glm.params.index,
        "coeficiente": modelo_glm.params.values,
        "error_estandar": modelo_glm.bse.values,
        "p_valor": modelo_glm.pvalues.values,
    })
    tabla_glm.loc[len(tabla_glm)] = [
        "R2_glm", r2_glm, np.nan, np.nan
    ]
    print(f"\nR2 del GLM: {r2_glm:.4f}")
    tabla_glm.to_csv(
        f"{CARPETA_SALIDA}/multivariado_glm_shannon.tsv",
        sep="\t", index=False,
    )

    print(f"\nArchivos guardados en '{CARPETA_SALIDA}/'.")


if __name__ == "__main__":
    main()
