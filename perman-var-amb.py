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
- outputs/fig3_r2_variables_ambientales.png / .pdf
- outputs/fig4_pcoa_por_variable.png / .pdf
"""

import matplotlib.pyplot as plt
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

ETIQUETAS_VARIABLES = {
    "humedad_relativa": "Humedad relativa\ndel suelo",
    "temperatura": "Temperatura\ndel suelo",
    "elevacion": "Elevación",
}

ETIQUETAS_COLORBAR = {
    "humedad_relativa": "Humedad relativa del suelo (%)",
    "temperatura": "Temperatura del suelo (°C)",
    "elevacion": "Elevación (m)",
}

R2_MINIMO_ESPERADO_H3 = 0.20
R2_MAXIMO_ESPERADO_H3 = 0.50


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


def calcular_pcoa(distancias):
    """Ordenacion PCoA clasica (metodo de Gower), igual que en
    comp-vs-soilrh.py, para poder colorear los mismos dos ejes por
    distintas variables ambientales.
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
    return (
        coordenadas[:, 0], coordenadas[:, 1],
        porcentaje_var[0], porcentaje_var[1],
    )


def graficar_r2_variables(tabla):
    tabla_ordenada = tabla.sort_values("R2", ascending=False)
    nombres = [
        ETIQUETAS_VARIABLES[v] for v in tabla_ordenada["variable"]
    ]
    valores_r2 = tabla_ordenada["R2"].to_numpy()
    colores = plt.get_cmap("Set2").colors

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.axhspan(
        R2_MINIMO_ESPERADO_H3, R2_MAXIMO_ESPERADO_H3,
        color="gray", alpha=0.15, zorder=0,
    )
    ax.text(
        len(nombres) - 0.5,
        (R2_MINIMO_ESPERADO_H3 + R2_MAXIMO_ESPERADO_H3) / 2,
        "Rango esperado\nen H3",
        ha="right", va="center", fontsize=9, color="dimgray",
    )

    barras = ax.bar(
        nombres, valores_r2, color=colores[:len(nombres)],
        edgecolor="black", linewidth=0.6, zorder=3,
    )
    for barra, p_valor in zip(barras, tabla_ordenada["p_valor"]):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 0.012,
            f"R²={barra.get_height():.3f}\np={p_valor:.3f}",
            ha="center", va="bottom", fontsize=8.5,
        )

    ax.set_ylabel("R² (varianza en composición explicada)")
    ax.set_ylim(0, R2_MAXIMO_ESPERADO_H3 + 0.08)
    ax.set_title(
        "¿Cuánto explica cada variable ambiental?\n"
        "PERMANOVA univariado, Bray-Curtis, 999 permutaciones"
    )
    fig.tight_layout()

    fig.savefig(
        f"{CARPETA_SALIDA}/fig3_r2_variables_ambientales.png",
        dpi=300,
    )
    fig.savefig(
        f"{CARPETA_SALIDA}/fig3_r2_variables_ambientales.pdf"
    )
    plt.close(fig)


def graficar_pcoa_por_variable(
    pc1, pc2, var_pc1, var_pc2, metadata
):
    fig, ejes = plt.subplots(
        1, 3, figsize=(15, 5.3), sharex=True, sharey=True
    )

    for ax, nombre in zip(ejes, COLUMNAS_AMBIENTALES):
        columna = COLUMNAS_AMBIENTALES[nombre]
        valores = metadata[columna].to_numpy(dtype=float)
        sin_dato = np.isnan(valores)

        dispersion = ax.scatter(
            pc1[~sin_dato], pc2[~sin_dato], c=valores[~sin_dato],
            cmap=plt.get_cmap("viridis"), edgecolor="black",
            linewidth=0.4, s=45,
        )
        cbar = fig.colorbar(dispersion, ax=ax, fraction=0.046)
        cbar.set_label(ETIQUETAS_COLORBAR[nombre], fontsize=8.5)
        ax.set_xlabel(f"PC1 ({var_pc1:.1f}%)")
        ax.set_title(ETIQUETAS_VARIABLES[nombre].replace("\n", " "))

    ejes[0].set_ylabel(f"PC2 ({var_pc2:.1f}%)")
    fig.suptitle(
        "PCoA (Bray-Curtis) coloreado por cada variable ambiental\n"
        "Desierto de Atacama (Neilson et al. 2017)"
    )
    fig.tight_layout()

    fig.savefig(
        f"{CARPETA_SALIDA}/fig4_pcoa_por_variable.png", dpi=300
    )
    fig.savefig(f"{CARPETA_SALIDA}/fig4_pcoa_por_variable.pdf")
    plt.close(fig)


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

    graficar_r2_variables(tabla)

    pc1, pc2, var_pc1, var_pc2 = calcular_pcoa(distancias)
    graficar_pcoa_por_variable(pc1, pc2, var_pc1, var_pc2, metadata)

    print(f"\nFiguras guardadas en '{CARPETA_SALIDA}/': "
          f"fig3_r2_variables_ambientales.png/.pdf, "
          f"fig4_pcoa_por_variable.png/.pdf.")


if __name__ == "__main__":
    main()
