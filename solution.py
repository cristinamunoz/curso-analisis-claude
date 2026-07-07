"""H1: relacion entre humedad relativa del suelo y diversidad
alfa (Shannon) en el desierto de Atacama.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

RUTA_ABUNDANCIAS = "data/abundancias.tsv"
RUTA_METADATA = "data/metadata.tsv"
COLUMNA_HUMEDAD = "average-soil-relative-humidity"
COLUMNA_TRANSECTO = "transect-name"

COLORES_TRANSECTO = {
    "Baquedano": "#66c2a5",
    "Yungay": "#fc8d62",
}


def cargar_datos():
    """Carga abundancias y metadata, y las cruza por sample-id."""
    abundancias = pd.read_csv(
        RUTA_ABUNDANCIAS, sep="\t", index_col=0
    )
    metadata = pd.read_csv(RUTA_METADATA, sep="\t")

    muestras_comunes = sorted(
        set(abundancias.columns) & set(metadata["sample-id"])
    )
    abundancias = abundancias[muestras_comunes]
    metadata = metadata[
        metadata["sample-id"].isin(muestras_comunes)
    ].set_index("sample-id").loc[muestras_comunes]

    return abundancias, metadata


def calcular_shannon(abundancias):
    """Calcula el indice de Shannon para cada muestra (columna)."""
    valores_shannon = {}
    for muestra in abundancias.columns:
        conteos = abundancias[muestra].to_numpy(dtype=float)
        conteos = conteos[conteos > 0]
        proporciones = conteos / conteos.sum()
        shannon = -np.sum(proporciones * np.log(proporciones))
        valores_shannon[muestra] = shannon
    return pd.Series(valores_shannon, name="shannon")


def resumen_por_transecto(datos):
    """Imprime media, mediana y rango de Shannon por transecto."""
    resumen = datos.groupby(COLUMNA_TRANSECTO)["shannon"].agg(
        n="count", media="mean", mediana="median",
        minimo="min", maximo="max",
    )
    print("\nResumen de diversidad de Shannon por transecto:")
    print(resumen.round(3))
    return resumen


def graficar_boxplot(datos):
    """Boxplot de Shannon por transecto, antes de la regresion."""
    transectos = sorted(datos[COLUMNA_TRANSECTO].unique())
    grupos = [
        datos.loc[
            datos[COLUMNA_TRANSECTO] == transecto, "shannon"
        ]
        for transecto in transectos
    ]

    figura, eje = plt.subplots(figsize=(8, 6))
    caja = eje.boxplot(
        grupos, tick_labels=transectos, patch_artist=True
    )
    for parche, transecto in zip(caja["boxes"], transectos):
        parche.set_facecolor(COLORES_TRANSECTO[transecto])

    eje.set_xlabel("Transecto")
    eje.set_ylabel("Indice de Shannon")
    eje.set_title(
        "Diversidad alfa (Shannon) por transecto\n"
        "Desierto de Atacama"
    )
    figura.tight_layout()
    figura.savefig(
        "outputs/fig1_shannon_por_transecto.png", dpi=300
    )
    figura.savefig("outputs/fig1_shannon_por_transecto.pdf")
    plt.close(figura)


def calcular_regresion(datos):
    """Correlacion de Spearman y regresion lineal Shannon ~
    humedad relativa del suelo.
    """
    humedad = datos[COLUMNA_HUMEDAD].to_numpy()
    shannon = datos["shannon"].to_numpy()

    rho, p_spearman = stats.spearmanr(humedad, shannon)
    regresion = stats.linregress(humedad, shannon)
    r2 = regresion.rvalue ** 2

    resultados = {
        "n": len(datos),
        "spearman_rho": rho,
        "spearman_p_valor": p_spearman,
        "r2_lineal": r2,
        "p_valor_lineal": regresion.pvalue,
        "pendiente": regresion.slope,
        "intercepto": regresion.intercept,
    }
    return resultados, regresion


def banda_confianza_95(humedad, shannon, regresion):
    """Banda de confianza al 95% para la recta ajustada."""
    n = len(humedad)
    x_linea = np.linspace(humedad.min(), humedad.max(), 200)
    y_linea = regresion.intercept + regresion.slope * x_linea

    predicho = regresion.intercept + regresion.slope * humedad
    error_residual = np.sqrt(
        np.sum((shannon - predicho) ** 2) / (n - 2)
    )
    x_media = humedad.mean()
    suma_cuadrados_x = np.sum((humedad - x_media) ** 2)

    valor_t = stats.t.ppf(0.975, df=n - 2)
    error_estandar = error_residual * np.sqrt(
        1 / n + (x_linea - x_media) ** 2 / suma_cuadrados_x
    )
    margen = valor_t * error_estandar

    return x_linea, y_linea, margen


def graficar_scatter(datos, resultados, regresion):
    """Scatter Shannon vs. humedad, coloreado por transecto,
    con recta ajustada y banda de confianza al 95%.
    """
    humedad = datos[COLUMNA_HUMEDAD].to_numpy()
    shannon = datos["shannon"].to_numpy()
    x_linea, y_linea, margen = banda_confianza_95(
        humedad, shannon, regresion
    )

    figura, eje = plt.subplots(figsize=(8, 6))

    for transecto, color in COLORES_TRANSECTO.items():
        subset = datos[datos[COLUMNA_TRANSECTO] == transecto]
        eje.scatter(
            subset[COLUMNA_HUMEDAD], subset["shannon"],
            label=transecto, color=color, edgecolor="black",
            alpha=0.85, zorder=3,
        )

    eje.plot(
        x_linea, y_linea, color="#404040",
        label="Regresion lineal", zorder=2,
    )
    eje.fill_between(
        x_linea, y_linea - margen, y_linea + margen,
        color="#404040", alpha=0.15,
        label="Intervalo de confianza 95%", zorder=1,
    )

    texto = (
        f"R² = {resultados['r2_lineal']:.3f}\n"
        f"p = {resultados['p_valor_lineal']:.3g}\n"
        f"Spearman ρ = {resultados['spearman_rho']:.3f}\n"
        f"n = {resultados['n']}"
    )
    eje.text(
        0.03, 0.97, texto, transform=eje.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.8
        ),
    )

    eje.set_xlabel("Humedad relativa promedio del suelo (%)")
    eje.set_ylabel("Indice de Shannon")
    eje.set_title(
        "Diversidad alfa vs. humedad relativa del suelo\n"
        "Desierto de Atacama"
    )
    eje.legend(loc="lower right")
    figura.tight_layout()
    figura.savefig(
        "outputs/fig1_shannon_vs_avgsoilrh.png", dpi=300
    )
    figura.savefig("outputs/fig1_shannon_vs_avgsoilrh.pdf")
    plt.close(figura)


def guardar_tablas(resumen_transecto, resultados):
    """Guarda las tablas de resultados en outputs/."""
    resumen_transecto.round(3).to_csv(
        "outputs/h1_resumen_shannon_por_transecto.tsv", sep="\t"
    )
    pd.DataFrame([resultados]).to_csv(
        "outputs/h1_correlacion_shannon_vs_humedad.tsv",
        sep="\t", index=False,
    )


def calcular_distancia_bray_curtis(abundancias):
    """Matriz de distancias Bray-Curtis entre muestras."""
    matriz = squareform(
        pdist(abundancias.T.to_numpy(), metric="braycurtis")
    )
    return matriz


def matriz_gower(matriz_distancias):
    """Matriz de Gower (doble centrada), base tanto de la
    PCoA como del estadistico pseudo-F de PERMANOVA.
    """
    n = matriz_distancias.shape[0]
    distancias_cuadrado = matriz_distancias ** 2
    centrado = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * centrado @ distancias_cuadrado @ centrado


def calcular_pcoa(matriz_distancias):
    """PCoA clasico (Gower) a partir de una matriz de
    distancias, via eigendescomposicion.
    """
    b = matriz_gower(matriz_distancias)
    autovalores, autovectores = np.linalg.eigh(b)
    orden = np.argsort(autovalores)[::-1]
    autovalores = autovalores[orden]
    autovectores = autovectores[:, orden]

    autovalores_positivos = autovalores[autovalores > 0]
    varianza_explicada = (
        autovalores_positivos
        / autovalores_positivos.sum()
        * 100
    )

    coordenadas = autovectores * np.sqrt(
        np.clip(autovalores, 0, None)
    )
    return coordenadas, varianza_explicada


def graficar_pcoa(datos, var_pc1, var_pc2):
    """PCoA coloreado por gradiente continuo de humedad."""
    figura, eje = plt.subplots(figsize=(8, 6))

    con_humedad = datos[COLUMNA_HUMEDAD].notna()
    dispersion = eje.scatter(
        datos.loc[con_humedad, "pc1"],
        datos.loc[con_humedad, "pc2"],
        c=datos.loc[con_humedad, COLUMNA_HUMEDAD],
        cmap="viridis", edgecolor="black", s=70, zorder=3,
    )

    sin_humedad = ~con_humedad
    if sin_humedad.any():
        eje.scatter(
            datos.loc[sin_humedad, "pc1"],
            datos.loc[sin_humedad, "pc2"],
            color="lightgray", edgecolor="black", s=70,
            label="Sin dato de humedad", zorder=3,
        )
        eje.legend(loc="best")

    barra_color = figura.colorbar(dispersion, ax=eje)
    barra_color.set_label(
        "Humedad relativa promedio del suelo (%)"
    )

    eje.set_xlabel(f"PC1 ({var_pc1:.1f}% varianza explicada)")
    eje.set_ylabel(f"PC2 ({var_pc2:.1f}% varianza explicada)")
    eje.set_title("PCoA (Bray-Curtis) de composicion microbiana")
    figura.tight_layout()
    figura.savefig("outputs/fig2_pcoa_braycurtis.png", dpi=300)
    figura.savefig("outputs/fig2_pcoa_braycurtis.pdf")
    plt.close(figura)


def comparar_pc1_por_transecto(datos):
    """Compara las coordenadas PC1 entre Baquedano y Yungay
    con un test de Mann-Whitney U (H2).
    """
    baquedano = datos.loc[
        datos[COLUMNA_TRANSECTO] == "Baquedano", "pc1"
    ]
    yungay = datos.loc[
        datos[COLUMNA_TRANSECTO] == "Yungay", "pc1"
    ]
    estadistico, p_valor = stats.mannwhitneyu(
        baquedano, yungay, alternative="two-sided"
    )

    resumen = pd.DataFrame({
        "transect-name": ["Baquedano", "Yungay"],
        "n": [len(baquedano), len(yungay)],
        "media_pc1": [baquedano.mean(), yungay.mean()],
        "mediana_pc1": [
            baquedano.median(), yungay.median()
        ],
    })
    print("\nPC1 por transecto:")
    print(resumen.round(3))
    print(
        f"\nMann-Whitney U = {estadistico:.1f}, "
        f"p = {p_valor:.3g}"
    )
    if len(baquedano) < 10 or len(yungay) < 10:
        print(
            "\nATENCION: n < 10 en algun grupo, las "
            "pruebas pueden ser insuficientes."
        )

    resultado_test = pd.DataFrame([{
        "estadistico_mannwhitney": estadistico,
        "p_valor": p_valor,
        "n_baquedano": len(baquedano),
        "n_yungay": len(yungay),
    }])
    return resumen, resultado_test


def calcular_correlacion_pc1_humedad(datos):
    """Correlacion entre PC1 y la humedad relativa del suelo
    (complementa el test por transecto de H2).
    """
    datos_validos = datos[datos[COLUMNA_HUMEDAD].notna()]
    humedad = datos_validos[COLUMNA_HUMEDAD].to_numpy()
    pc1 = datos_validos["pc1"].to_numpy()

    rho, p_spearman = stats.spearmanr(humedad, pc1)
    regresion = stats.linregress(humedad, pc1)
    r2 = regresion.rvalue ** 2

    resultados = {
        "n": len(datos_validos),
        "spearman_rho": rho,
        "spearman_p_valor": p_spearman,
        "r2_lineal": r2,
        "p_valor_lineal": regresion.pvalue,
        "pendiente": regresion.slope,
        "intercepto": regresion.intercept,
    }
    return resultados, regresion, datos_validos


def graficar_pc1_vs_humedad(datos_validos, resultados,
                             regresion, var_pc1):
    """Scatter de PC1 vs. humedad, coloreado por transecto,
    con recta ajustada y banda de confianza al 95%.
    """
    humedad = datos_validos[COLUMNA_HUMEDAD].to_numpy()
    pc1 = datos_validos["pc1"].to_numpy()
    x_linea, y_linea, margen = banda_confianza_95(
        humedad, pc1, regresion
    )

    figura, eje = plt.subplots(figsize=(8, 6))
    for transecto, color in COLORES_TRANSECTO.items():
        subset = datos_validos[
            datos_validos[COLUMNA_TRANSECTO] == transecto
        ]
        eje.scatter(
            subset[COLUMNA_HUMEDAD], subset["pc1"],
            label=transecto, color=color, edgecolor="black",
            alpha=0.85, zorder=3,
        )
    eje.plot(
        x_linea, y_linea, color="#404040",
        label="Regresion lineal", zorder=2,
    )
    eje.fill_between(
        x_linea, y_linea - margen, y_linea + margen,
        color="#404040", alpha=0.15,
        label="Intervalo de confianza 95%", zorder=1,
    )

    texto = (
        f"R² = {resultados['r2_lineal']:.3f}\n"
        f"p = {resultados['p_valor_lineal']:.3g}\n"
        f"Spearman ρ = {resultados['spearman_rho']:.3f}\n"
        f"n = {resultados['n']}"
    )
    eje.text(
        0.03, 0.03, texto, transform=eje.transAxes,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.8
        ),
    )

    eje.set_xlabel("Humedad relativa promedio del suelo (%)")
    eje.set_ylabel(f"PC1 ({var_pc1:.1f}% varianza explicada)")
    eje.set_title(
        "PC1 (composicion microbiana) vs. humedad del suelo\n"
        "Desierto de Atacama"
    )
    eje.legend(loc="upper right")
    figura.tight_layout()
    figura.savefig("outputs/fig2_pc1_vs_humedad.png", dpi=300)
    figura.savefig("outputs/fig2_pc1_vs_humedad.pdf")
    plt.close(figura)


def guardar_tabla_pc1_humedad(resultados):
    """Guarda la tabla de correlacion PC1 vs. humedad."""
    pd.DataFrame([resultados]).to_csv(
        "outputs/h2_correlacion_pc1_vs_humedad.tsv",
        sep="\t", index=False,
    )


def guardar_tablas_h2(varianza_explicada, resumen_pc1,
                       resultado_test):
    """Guarda las tablas de resultados de H2 en outputs/."""
    pd.DataFrame({
        "eje": ["PC1", "PC2"],
        "porcentaje_varianza": varianza_explicada[:2],
    }).to_csv(
        "outputs/h2_varianza_explicada_pcoa.tsv",
        sep="\t", index=False,
    )
    resumen_pc1.round(3).to_csv(
        "outputs/h2_pc1_por_transecto.tsv",
        sep="\t", index=False,
    )
    resultado_test.to_csv(
        "outputs/h2_test_pc1_por_transecto.tsv",
        sep="\t", index=False,
    )


def permanova_univariada(matriz_distancias, variable,
                          n_permutaciones=999, semilla=0):
    """PERMANOVA univariada (McArdle & Anderson 2001):
    pseudo-F, R2 y p-valor por permutacion para una variable
    ambiental continua.
    """
    g = matriz_gower(matriz_distancias)
    n = g.shape[0]
    ss_total = np.trace(g)
    df_variable = 1
    df_residual = n - 2

    def suma_cuadrados_explicada(valores):
        x = (valores - valores.mean()).reshape(-1, 1)
        h = x @ np.linalg.pinv(x.T @ x) @ x.T
        return np.trace(h @ g)

    def pseudo_f(valores):
        ss_explicada = suma_cuadrados_explicada(valores)
        ss_residual = ss_total - ss_explicada
        return (
            (ss_explicada / df_variable)
            / (ss_residual / df_residual)
        ), ss_explicada

    f_observado, ss_explicada = pseudo_f(variable)

    generador = np.random.default_rng(semilla)
    f_permutados = np.empty(n_permutaciones)
    for i in range(n_permutaciones):
        variable_permutada = generador.permutation(variable)
        f_permutados[i], _ = pseudo_f(variable_permutada)

    p_valor = (
        np.sum(f_permutados >= f_observado) + 1
    ) / (n_permutaciones + 1)

    return {
        "F": f_observado,
        "R2": ss_explicada / ss_total,
        "p_valor": p_valor,
        "n_permutaciones": n_permutaciones,
        "n": n,
    }


def calcular_permanova_h3(abundancias, datos_pcoa):
    """PERMANOVA univariada de humedad, temperatura y
    elevacion sobre la composicion microbiana (H3). Cada
    variable usa sus propias muestras disponibles.
    """
    variables = [
        ("humedad_relativa", COLUMNA_HUMEDAD),
        ("temperatura", "average-soil-temperature"),
        ("elevacion", "elevation"),
    ]

    resultados = []
    for etiqueta, columna in variables:
        disponibles = datos_pcoa[columna].notna()
        muestras = datos_pcoa.index[disponibles]
        if disponibles.sum() < 10:
            print(
                f"\nATENCION: n < 10 para '{etiqueta}', las "
                "permutaciones pueden ser insuficientes."
            )

        matriz = calcular_distancia_bray_curtis(
            abundancias[muestras]
        )
        valores = datos_pcoa.loc[muestras, columna].to_numpy(
            dtype=float
        )
        resultado = permanova_univariada(matriz, valores)
        resultado["variable"] = etiqueta
        resultado["columna_original"] = columna
        resultados.append(resultado)

    tabla = pd.DataFrame(resultados)[
        ["variable", "columna_original", "F", "R2",
         "p_valor", "n_permutaciones", "n"]
    ].sort_values("R2", ascending=False).reset_index(drop=True)
    return tabla


def guardar_tabla_h3(tabla):
    """Guarda la tabla de PERMANOVA de H3 en outputs/."""
    tabla.to_csv(
        "outputs/h3_permanova_variables_ambientales.tsv",
        sep="\t", index=False,
    )


def main():
    abundancias, metadata = cargar_datos()
    shannon = calcular_shannon(abundancias)
    datos = metadata.join(shannon)

    resumen_transecto = resumen_por_transecto(datos)
    graficar_boxplot(datos)

    n_total = len(datos)
    datos_regresion = datos.dropna(subset=[COLUMNA_HUMEDAD])
    n_excluidas = n_total - len(datos_regresion)
    if n_excluidas:
        print(
            f"\n{n_excluidas} muestra(s) sin dato de humedad "
            "se excluyeron de la regresion (se mantienen en "
            "el boxplot por transecto)."
        )

    resultados, regresion = calcular_regresion(datos_regresion)
    print("\nRegresion Shannon ~ humedad relativa del suelo:")
    for clave, valor in resultados.items():
        print(f"  {clave}: {valor}")

    if resultados["r2_lineal"] < 0.05:
        print(
            "\nATENCION: R² menor a 0.05. Puede indicar un "
            "problema con los datos o el analisis."
        )

    graficar_scatter(datos_regresion, resultados, regresion)
    guardar_tablas(resumen_transecto, resultados)
    print("\nFiguras y tablas de H1 guardadas en outputs/")

    matriz_distancias = calcular_distancia_bray_curtis(
        abundancias
    )
    coordenadas, varianza_explicada = calcular_pcoa(
        matriz_distancias
    )
    datos_pcoa = datos.copy()
    datos_pcoa["pc1"] = coordenadas[:, 0]
    datos_pcoa["pc2"] = coordenadas[:, 1]

    print(
        "\nVarianza explicada: "
        f"PC1 = {varianza_explicada[0]:.1f}%, "
        f"PC2 = {varianza_explicada[1]:.1f}%"
    )
    graficar_pcoa(
        datos_pcoa, varianza_explicada[0], varianza_explicada[1]
    )
    resumen_pc1, resultado_test = comparar_pc1_por_transecto(
        datos_pcoa
    )
    guardar_tablas_h2(
        varianza_explicada, resumen_pc1, resultado_test
    )

    resultados_pc1, regresion_pc1, datos_validos = (
        calcular_correlacion_pc1_humedad(datos_pcoa)
    )
    print("\nCorrelacion PC1 vs. humedad del suelo:")
    for clave, valor in resultados_pc1.items():
        print(f"  {clave}: {valor}")
    graficar_pc1_vs_humedad(
        datos_validos, resultados_pc1, regresion_pc1,
        varianza_explicada[0],
    )
    guardar_tabla_pc1_humedad(resultados_pc1)
    print("\nFiguras y tablas de H2 guardadas en outputs/")

    tabla_h3 = calcular_permanova_h3(abundancias, datos_pcoa)
    print("\nPERMANOVA univariada (H3):")
    print(tabla_h3.round(4))
    guardar_tabla_h3(tabla_h3)
    print("\nTabla de H3 guardada en outputs/")


if __name__ == "__main__":
    main()
