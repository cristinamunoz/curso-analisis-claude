"""
Analisis de diversidad alfa (Shannon) y beta (Bray-Curtis / PCoA)
entre los transectos Baquedano y Yungay del desierto de Atacama.

Reproduce H1 (Shannon vs. humedad relativa del suelo), H2
(separacion de transectos en la ordenacion PCoA) y H3 (PERMANOVA
univariada de humedad, temperatura y elevacion sobre la
composicion, distancias Bray-Curtis).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from skbio.diversity import alpha_diversity, beta_diversity
from skbio.stats.ordination import pcoa

# --- Configuracion general ---------------------------------------

RUTA_ABUNDANCIAS = "data/abundancias.tsv"
RUTA_METADATA = "data/metadata.tsv"
CARPETA_SALIDA = "outputs"

os.makedirs(CARPETA_SALIDA, exist_ok=True)

sns.set_style("whitegrid")
PALETA_TRANSECTOS = {"Baquedano": "#3B7EA1", "Yungay": "#D9822B"}


# --- 1. Cargar y cruzar los datos ----------------------------------

def cargar_datos():
    """Carga abundancias y metadata, y las cruza por sample-id."""
    abundancias = pd.read_csv(
        RUTA_ABUNDANCIAS, sep="\t", index_col=0
    )
    metadata = pd.read_csv(RUTA_METADATA, sep="\t")
    metadata = metadata.set_index("sample-id")

    # Solo 54 de las 75 muestras de metadata tienen secuenciacion.
    # Nos quedamos con la interseccion de IDs entre ambos archivos.
    muestras_comunes = sorted(
        set(abundancias.columns) & set(metadata.index)
    )
    abundancias = abundancias[muestras_comunes]
    metadata = metadata.loc[muestras_comunes]

    return abundancias, metadata


# --- 2. Diversidad alfa (Shannon) ----------------------------------

def calcular_shannon(abundancias):
    """Calcula el indice de Shannon por muestra (OTUs en filas)."""
    tabla = abundancias.T  # skbio espera muestras en filas
    valores = alpha_diversity(
        "shannon", tabla.values, ids=tabla.index
    )
    return valores.rename("shannon")


def resumen_shannon_por_transecto(shannon, metadata):
    """Imprime media, mediana y rango de Shannon por transecto."""
    df = shannon.to_frame().join(metadata["transect-name"])
    resumen = df.groupby("transect-name")["shannon"].agg(
        media="mean", mediana="median", minimo="min", maximo="max",
        n="count",
    )
    print("\nResumen de diversidad de Shannon por transecto:")
    print(resumen.round(3))
    resumen.to_csv(
        os.path.join(CARPETA_SALIDA, "h1_resumen_shannon.tsv"),
        sep="\t",
    )
    return df


def graficar_boxplot_shannon(df):
    """Boxplot de Shannon por transecto."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=df, x="transect-name", y="shannon",
        hue="transect-name", palette=PALETA_TRANSECTOS,
        ax=ax, legend=False,
    )
    sns.stripplot(
        data=df, x="transect-name", y="shannon",
        color="black", alpha=0.5, ax=ax,
    )
    ax.set_xlabel("Transecto")
    ax.set_ylabel("Indice de diversidad de Shannon")
    ax.set_title("Diversidad alfa (Shannon) por transecto")
    fig.tight_layout()
    fig.savefig(
        os.path.join(CARPETA_SALIDA, "fig1_shannon_por_transecto.png"),
        dpi=300,
    )
    fig.savefig(
        os.path.join(CARPETA_SALIDA, "fig1_shannon_por_transecto.pdf")
    )
    plt.close(fig)


def graficar_shannon_vs_humedad(df, metadata):
    """Scatter Shannon vs. AvgSoilRH con regresion y estadisticos."""
    df = df.join(
        metadata["average-soil-relative-humidity"].rename("humedad")
    )
    df = df.dropna(subset=["humedad", "shannon"])

    rho, p = stats.spearmanr(df["humedad"], df["shannon"])
    pendiente, intercepto, r, _, _ = stats.linregress(
        df["humedad"], df["shannon"]
    )
    r2 = r ** 2
    n = len(df)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        data=df, x="humedad", y="shannon", ax=ax,
        scatter_kws={"alpha": 0.6}, color="#3B7EA1",
    )
    ax.set_xlabel("Humedad relativa promedio del suelo (%)")
    ax.set_ylabel("Indice de diversidad de Shannon")
    ax.set_title("Shannon vs. humedad relativa del suelo")
    texto = f"Spearman rho = {rho:.2f}\nR2 = {r2:.2f}\np = {p:.3f}\nn = {n}"
    ax.text(
        0.05, 0.95, texto, transform=ax.transAxes,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(CARPETA_SALIDA, "fig1_shannon_vs_avgsoilrh.png"),
        dpi=300,
    )
    fig.savefig(
        os.path.join(CARPETA_SALIDA, "fig1_shannon_vs_avgsoilrh.pdf")
    )
    plt.close(fig)

    resultado = pd.DataFrame([{
        "spearman_rho": rho, "r2": r2, "p_valor": p, "n": n,
    }])
    resultado.to_csv(
        os.path.join(
            CARPETA_SALIDA, "h1_correlacion_shannon_vs_humedad.tsv"
        ),
        sep="\t", index=False,
    )
    print("\nCorrelacion Shannon vs. humedad relativa del suelo:")
    print(resultado.round(4))
    if r2 < 0.05:
        print(
            "\nAVISO: R2 < 0.05. La relacion es muy debil; conviene "
            "revisar los datos antes de sacar conclusiones."
        )


# --- 3. Diversidad beta (Bray-Curtis / PCoA) ------------------------

def calcular_pcoa(abundancias):
    """Calcula distancias Bray-Curtis y hace la ordenacion PCoA."""
    tabla = abundancias.T
    distancias = beta_diversity(
        "braycurtis", tabla.values, ids=tabla.index
    )
    resultado = pcoa(distancias)
    return resultado


def graficar_pcoa(resultado_pcoa, metadata):
    """Grafica PC1 vs PC2 coloreado por transecto y por humedad."""
    coords = resultado_pcoa.samples[["PC1", "PC2"]].copy()
    coords = coords.join(metadata[
        ["transect-name", "average-soil-relative-humidity"]
    ])
    var_explicada = resultado_pcoa.proportion_explained

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for transecto, color in PALETA_TRANSECTOS.items():
        subset = coords[coords["transect-name"] == transecto]
        ax.scatter(
            subset["PC1"], subset["PC2"], label=transecto,
            color=color, alpha=0.8, s=60,
        )
    ax.set_xlabel(f"PC1 ({var_explicada.iloc[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explicada.iloc[1] * 100:.1f}%)")
    ax.set_title("PCoA (Bray-Curtis) por transecto")
    ax.legend(title="Transecto")

    ax = axes[1]
    disp = ax.scatter(
        coords["PC1"], coords["PC2"],
        c=coords["average-soil-relative-humidity"],
        cmap="viridis", s=60,
    )
    ax.set_xlabel(f"PC1 ({var_explicada.iloc[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explicada.iloc[1] * 100:.1f}%)")
    ax.set_title("PCoA (Bray-Curtis) por humedad del suelo")
    fig.colorbar(disp, ax=ax, label="Humedad relativa del suelo (%)")

    fig.suptitle("Diversidad beta entre transectos (Bray-Curtis)")
    fig.tight_layout()
    fig.savefig(
        os.path.join(CARPETA_SALIDA, "fig2_pcoa_braycurtis.png"),
        dpi=300,
    )
    fig.savefig(
        os.path.join(CARPETA_SALIDA, "fig2_pcoa_braycurtis.pdf")
    )
    plt.close(fig)

    resumen = pd.DataFrame({
        "eje": ["PC1", "PC2"],
        "proporcion_varianza_explicada": [
            var_explicada.iloc[0], var_explicada.iloc[1]
        ],
    })
    resumen.to_csv(
        os.path.join(
            CARPETA_SALIDA, "h2_varianza_explicada_pcoa.tsv"
        ),
        sep="\t", index=False,
    )
    print("\nVarianza explicada por la ordenacion PCoA:")
    print(resumen.round(4))


# --- 4. PERMANOVA univariada (variables ambientales continuas) -----

def permanova_variable_continua(dist_df, variable, n_perm=999,
                                 semilla=0):
    """PERMANOVA de una distancia contra una variable continua.

    Implementa la particion de sumas de cuadrados de McArdle y
    Anderson (2001), equivalente a un PERMANOVA/adonis univariado:
    regresion de la matriz de distancias (centrada a la Gower)
    contra la variable ambiental, con significancia evaluada por
    permutacion de las muestras.
    """
    ids = [i for i in dist_df.index if not pd.isna(variable[i])]
    d = dist_df.loc[ids, ids].to_numpy()
    x = variable.loc[ids].to_numpy(dtype=float)
    n = len(ids)

    # Matriz de distancias centrada a la Gower (G).
    a = -0.5 * (d ** 2)
    centrado = np.eye(n) - np.ones((n, n)) / n
    g = centrado @ a @ centrado
    suma_cuadrados_total = np.trace(g)

    def suma_cuadrados_regresion(x_valores):
        diseno = np.column_stack([np.ones(n), x_valores])
        hat = diseno @ np.linalg.pinv(diseno.T @ diseno) @ diseno.T
        return np.trace(hat @ g)

    ss_reg = suma_cuadrados_regresion(x)
    ss_res = suma_cuadrados_total - ss_reg
    df_reg, df_res = 1, n - 2
    f_observado = (ss_reg / df_reg) / (ss_res / df_res)
    r2 = ss_reg / suma_cuadrados_total

    generador = np.random.default_rng(semilla)
    mayores_o_iguales = 0
    for _ in range(n_perm):
        permutacion = generador.permutation(n)
        ss_reg_perm = suma_cuadrados_regresion(x[permutacion])
        ss_res_perm = suma_cuadrados_total - ss_reg_perm
        f_perm = (ss_reg_perm / df_reg) / (ss_res_perm / df_res)
        if f_perm >= f_observado:
            mayores_o_iguales += 1
    p_valor = (mayores_o_iguales + 1) / (n_perm + 1)

    return f_observado, r2, p_valor, n


def permanova_variables_ambientales(abundancias, metadata):
    """PERMANOVA univariada de humedad, temperatura y elevacion."""
    tabla = abundancias.T
    distancias = beta_diversity(
        "braycurtis", tabla.values, ids=tabla.index
    )
    dist_df = distancias.to_data_frame()

    variables = [
        ("humedad_relativa", "average-soil-relative-humidity"),
        ("temperatura", "average-soil-temperature"),
        ("elevacion", "elevation"),
    ]

    filas = []
    for nombre, columna in variables:
        n_validos = metadata[columna].notna().sum()
        if n_validos < 10:
            print(
                f"\nAVISO: {nombre} tiene solo {n_validos} muestras "
                "validas (< 10). Las permutaciones pueden ser "
                "insuficientes."
            )
        f, r2, p, n = permanova_variable_continua(
            dist_df, metadata[columna]
        )
        filas.append({
            "variable": nombre, "columna_original": columna,
            "F": f, "R2": r2, "p_valor": p,
            "n_permutaciones": 999, "n": n,
        })

    resultado = pd.DataFrame(filas).sort_values(
        "R2", ascending=False
    ).reset_index(drop=True)
    resultado.to_csv(
        os.path.join(
            CARPETA_SALIDA, "h3_permanova_variables_ambientales.tsv"
        ),
        sep="\t", index=False,
    )
    print(
        "\nPERMANOVA univariada de variables ambientales "
        "(Bray-Curtis, 999 permutaciones):"
    )
    print(resultado.round(4))
    return resultado


# --- Ejecucion principal --------------------------------------------

def main():
    abundancias, metadata = cargar_datos()
    print(
        f"Muestras cruzadas entre abundancias y metadata: "
        f"{abundancias.shape[1]}"
    )

    shannon = calcular_shannon(abundancias)
    df_shannon = resumen_shannon_por_transecto(shannon, metadata)
    graficar_boxplot_shannon(df_shannon)
    graficar_shannon_vs_humedad(df_shannon, metadata)

    resultado_pcoa = calcular_pcoa(abundancias)
    graficar_pcoa(resultado_pcoa, metadata)

    permanova_variables_ambientales(abundancias, metadata)

    print("\nListo. Figuras y tablas guardadas en 'outputs/'.")


if __name__ == "__main__":
    main()
