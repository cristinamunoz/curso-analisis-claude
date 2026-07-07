"""
Analisis H1: relacion entre diversidad alfa (Shannon) y humedad
relativa del suelo (AvgSoilRH) en el desierto de Atacama.

Hipotesis: a menor humedad relativa del suelo, menor diversidad
alfa (Shannon).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import squareform, pdist

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_DIR_H2 = "outputs2"
os.makedirs(OUTPUT_DIR_H2, exist_ok=True)

OUTPUT_DIR_H3 = "outputs3"
os.makedirs(OUTPUT_DIR_H3, exist_ok=True)

PALETTE_TRANSECTOS = {
    "Baquedano": "#1b9e77",
    "Yungay": "#d95f02",
}


def cargar_datos():
    """Carga abundancias y metadata, y las cruza por sample-id."""
    abundancias = pd.read_csv(
        "data/abundancias.tsv", sep="\t", index_col=0
    )
    metadata = pd.read_csv(
        "data/metadata.tsv", sep="\t", index_col="sample-id"
    )

    # Solo 54 de las 75 muestras en metadata tienen secuenciacion.
    # Se usa la interseccion de IDs entre ambos archivos.
    muestras_comunes = abundancias.columns.intersection(
        metadata.index
    )
    abundancias = abundancias[muestras_comunes]
    metadata = metadata.loc[muestras_comunes]

    return abundancias, metadata


def calcular_shannon(abundancias):
    """Calcula el indice de diversidad de Shannon por muestra."""
    shannon_por_muestra = {}
    for muestra in abundancias.columns:
        conteos = abundancias[muestra].values
        conteos = conteos[conteos > 0]
        proporciones = conteos / conteos.sum()
        shannon = -np.sum(proporciones * np.log(proporciones))
        shannon_por_muestra[muestra] = shannon
    return pd.Series(shannon_por_muestra, name="shannon")


def resumen_por_transecto(datos):
    """Imprime media, mediana y rango de Shannon por transecto."""
    resumen = datos.groupby("transect-name")["shannon"].agg(
        media="mean",
        mediana="median",
        minimo="min",
        maximo="max",
        n="count",
    )
    print("\nResumen de diversidad de Shannon por transecto:")
    print(resumen.round(3))
    return resumen


def boxplot_por_transecto(datos):
    """Genera y guarda el boxplot de Shannon por transecto."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=datos,
        x="transect-name",
        y="shannon",
        hue="transect-name",
        palette=PALETTE_TRANSECTOS,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=datos,
        x="transect-name",
        y="shannon",
        color="black",
        alpha=0.5,
        ax=ax,
    )
    ax.set_title(
        "Diversidad de Shannon por transecto\n"
        "Desierto de Atacama"
    )
    ax.set_xlabel("Transecto")
    ax.set_ylabel("Indice de diversidad de Shannon")
    fig.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, "fig1_shannon_por_transecto.png"),
        dpi=300,
    )
    fig.savefig(
        os.path.join(OUTPUT_DIR, "fig1_shannon_por_transecto.pdf")
    )
    plt.close(fig)


def correlacion_shannon_humedad(datos):
    """Calcula correlacion de Spearman y regresion lineal."""
    columna_humedad = "average-soil-relative-humidity"
    n_faltantes = datos[columna_humedad].isna().sum()
    if n_faltantes > 0:
        print(
            f"\n  AVISO: {n_faltantes} muestra(s) sin dato de "
            "humedad relativa del suelo - se excluyen de la "
            "correlacion y la regresion."
        )
    datos_validos = datos.dropna(subset=[columna_humedad])

    x = datos_validos[columna_humedad]
    y = datos_validos["shannon"]

    rho, p_valor_spearman = stats.spearmanr(x, y)

    pendiente, intercepto, r_valor, p_valor_reg, error_est = (
        stats.linregress(x, y)
    )
    r2 = r_valor ** 2
    n = len(datos_validos)

    print("\nCorrelacion Shannon vs. humedad relativa del suelo:")
    print(f"  Spearman rho = {rho:.3f} (p = {p_valor_spearman:.4f})")
    print(f"  R2 (regresion lineal) = {r2:.3f}")
    print(f"  p-valor (regresion) = {p_valor_reg:.4f}")
    print(f"  n = {n}")

    if r2 < 0.05:
        print(
            "\n  AVISO: R2 menor a 0.05 - la relacion lineal es muy "
            "debil. Revisar antes de continuar."
        )

    resultado = pd.DataFrame(
        {
            "spearman_rho": [rho],
            "spearman_p_valor": [p_valor_spearman],
            "r2_regresion": [r2],
            "p_valor_regresion": [p_valor_reg],
            "pendiente": [pendiente],
            "n": [n],
        }
    )
    resultado.to_csv(
        os.path.join(
            OUTPUT_DIR, "h1_correlacion_shannon_vs_humedad.tsv"
        ),
        sep="\t",
        index=False,
    )
    return rho, p_valor_spearman, r2, p_valor_reg, n


def scatter_shannon_humedad(datos, r2, p_valor):
    """Genera el scatter Shannon vs. humedad con banda de IC 95%."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.regplot(
        data=datos,
        x="average-soil-relative-humidity",
        y="shannon",
        ci=95,
        scatter_kws={"alpha": 0.6, "color": "#7570b3"},
        line_kws={"color": "#d95f02"},
        ax=ax,
    )

    ax.set_title(
        "Diversidad de Shannon vs. humedad relativa del suelo\n"
        "Desierto de Atacama"
    )
    ax.set_xlabel("Humedad relativa promedio del suelo (%)")
    ax.set_ylabel("Indice de diversidad de Shannon")

    texto = f"R2 = {r2:.3f}\np = {p_valor:.4f}"
    ax.text(
        0.05,
        0.05,
        texto,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.8
        ),
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            OUTPUT_DIR, "fig1_shannon_vs_avgsoilrh.png"
        ),
        dpi=300,
    )
    fig.savefig(
        os.path.join(OUTPUT_DIR, "fig1_shannon_vs_avgsoilrh.pdf")
    )
    plt.close(fig)


def calcular_distancias_braycurtis(abundancias):
    """Calcula la matriz de distancias Bray-Curtis entre muestras."""
    muestras = abundancias.columns
    matriz_conteos = abundancias.T.values
    distancias = squareform(
        pdist(matriz_conteos, metric="braycurtis")
    )
    return pd.DataFrame(distancias, index=muestras, columns=muestras)


def centrar_gower(distancias):
    """Aplica el centrado de Gower a una matriz de distancias."""
    n = distancias.shape[0]
    matriz_a = -0.5 * distancias ** 2
    centrador = np.eye(n) - np.ones((n, n)) / n
    return centrador @ matriz_a @ centrador


def calcular_pcoa(abundancias):
    """
    Calcula PCoA clasico a partir de distancias Bray-Curtis.

    Usa el metodo de Gower: centra la matriz de distancias al
    cuadrado y extrae los eigenvalores/eigenvectores. Es el mismo
    procedimiento que usan librerias como scikit-bio, sin depender
    de esa libreria.
    """
    muestras = abundancias.columns
    distancias = calcular_distancias_braycurtis(abundancias).values
    matriz_g = centrar_gower(distancias)

    eigenvalores, eigenvectores = np.linalg.eigh(matriz_g)

    # np.linalg.eigh ordena de menor a mayor; se invierte el orden.
    orden = np.argsort(eigenvalores)[::-1]
    eigenvalores = eigenvalores[orden]
    eigenvectores = eigenvectores[:, orden]

    varianza_total = eigenvalores[eigenvalores > 0].sum()
    porcentaje_varianza = (
        eigenvalores / varianza_total * 100
    )

    coordenadas = eigenvectores * np.sqrt(
        np.clip(eigenvalores, a_min=0, a_max=None)
    )

    coordenadas_df = pd.DataFrame(
        coordenadas[:, :2],
        index=muestras,
        columns=["PC1", "PC2"],
    )

    return coordenadas_df, porcentaje_varianza[:2]


def guardar_varianza_pcoa(porcentaje_varianza):
    """Guarda el % de varianza explicado por PC1 y PC2."""
    tabla = pd.DataFrame(
        {
            "eje": ["PC1", "PC2"],
            "porcentaje_varianza": porcentaje_varianza,
        }
    )
    print("\nVarianza explicada por la ordenacion PCoA:")
    print(tabla.round(2).to_string(index=False))
    tabla.to_csv(
        os.path.join(
            OUTPUT_DIR_H2, "h2_varianza_explicada_pcoa.tsv"
        ),
        sep="\t",
        index=False,
    )
    return tabla


def graficar_pcoa(coordenadas, metadata, porcentaje_varianza):
    """
    Genera el scatter de PCoA (PC1 vs PC2).

    Color: gradiente continuo de humedad relativa del suelo
    (paleta viridis). Forma: transecto (Baquedano vs Yungay), para
    evaluar visualmente la hipotesis H2.
    """
    datos = coordenadas.join(metadata)
    columna_humedad = "average-soil-relative-humidity"
    con_humedad = datos[datos[columna_humedad].notna()]
    sin_humedad = datos[datos[columna_humedad].isna()]

    fig, ax = plt.subplots(figsize=(8, 6))

    marcadores = {"Baquedano": "o", "Yungay": "^"}

    dispersión = ax.scatter(
        con_humedad["PC1"],
        con_humedad["PC2"],
        c=con_humedad[columna_humedad],
        cmap="viridis",
        s=90,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    if len(sin_humedad) > 0:
        ax.scatter(
            sin_humedad["PC1"],
            sin_humedad["PC2"],
            facecolor="#d3d3d3",
            hatch="///",
            s=90,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            label="Sin dato de humedad",
        )

    for transecto, marcador in marcadores.items():
        subconjunto = datos[datos["transect-name"] == transecto]
        ax.scatter(
            subconjunto["PC1"],
            subconjunto["PC2"],
            facecolor="none",
            edgecolor="black",
            marker=marcador,
            s=140,
            linewidth=1.2,
            label=transecto,
        )

    barra_color = fig.colorbar(dispersión, ax=ax)
    barra_color.set_label(
        "Humedad relativa promedio del suelo (%)"
    )

    ax.set_xlabel(f"PC1 ({porcentaje_varianza[0]:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({porcentaje_varianza[1]:.1f}% varianza)")
    ax.set_title(
        "PCoA (Bray-Curtis) de la composicion microbiana\n"
        "Desierto de Atacama"
    )
    ax.legend(title="Transecto / dato faltante", loc="best")

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR_H2, "fig2_pcoa_braycurtis.png"),
        dpi=300,
    )
    fig.savefig(
        os.path.join(OUTPUT_DIR_H2, "fig2_pcoa_braycurtis.pdf")
    )
    plt.close(fig)


def main_h2():
    abundancias, metadata = cargar_datos()

    coordenadas, porcentaje_varianza = calcular_pcoa(abundancias)
    guardar_varianza_pcoa(porcentaje_varianza)
    graficar_pcoa(coordenadas, metadata, porcentaje_varianza)

    print(
        "\nListo. Figuras y tabla de H2 guardadas en la carpeta "
        "'outputs2/'."
    )


def main():
    abundancias, metadata = cargar_datos()
    shannon = calcular_shannon(abundancias)

    datos = metadata.copy()
    datos["shannon"] = shannon

    resumen = resumen_por_transecto(datos)
    resumen.to_csv(
        os.path.join(
            OUTPUT_DIR, "h1_resumen_shannon_por_transecto.tsv"
        ),
        sep="\t",
    )

    boxplot_por_transecto(datos)

    rho, p_spearman, r2, p_reg, n = correlacion_shannon_humedad(
        datos
    )
    scatter_shannon_humedad(datos, r2, p_reg)

    print(
        "\nListo. Figuras y tablas guardadas en la carpeta "
        "'outputs/'."
    )


def permanova_univariada(distancias, variable, n_permutaciones=999,
                          semilla=0):
    """
    PERMANOVA univariada (McArdle & Anderson, 2001) para una sola
    variable ambiental continua contra una matriz de distancias.

    Reproduce el pseudo-F de adonis/adonis2 (paquete vegan de R):
    parte la suma de cuadrados total de la matriz centrada de Gower
    en la parte explicada por la variable y el residuo, usando la
    matriz de proyeccion (hat matrix) de un modelo lineal con
    intercepto + la variable.
    """
    aleatorio = np.random.RandomState(semilla)

    matriz_g = centrar_gower(distancias.values)
    n = matriz_g.shape[0]

    x = np.column_stack([np.ones(n), variable.values])
    hat = x @ np.linalg.pinv(x.T @ x) @ x.T
    identidad = np.eye(n)

    ss_total = np.trace(matriz_g)

    def pseudo_f(hat_matrix):
        ss_modelo = np.trace(hat_matrix @ matriz_g @ hat_matrix)
        ss_residuo = np.trace(
            (identidad - hat_matrix) @ matriz_g
            @ (identidad - hat_matrix)
        )
        gl_modelo = 1
        gl_residuo = n - 2
        f = (ss_modelo / gl_modelo) / (ss_residuo / gl_residuo)
        r2 = ss_modelo / ss_total
        return f, r2

    f_observado, r2_observado = pseudo_f(hat)

    contador = 0
    for _ in range(n_permutaciones):
        orden = aleatorio.permutation(n)
        x_permutado = np.column_stack(
            [np.ones(n), variable.values[orden]]
        )
        hat_permutado = (
            x_permutado
            @ np.linalg.pinv(x_permutado.T @ x_permutado)
            @ x_permutado.T
        )
        f_permutado, _ = pseudo_f(hat_permutado)
        if f_permutado >= f_observado:
            contador += 1

    p_valor = (contador + 1) / (n_permutaciones + 1)

    return f_observado, r2_observado, p_valor, n


def permanova_variables_ambientales(abundancias, metadata):
    """
    Corre PERMANOVA univariada para humedad, temperatura y
    elevacion, y ordena el resultado de mayor a menor R2.
    """
    distancias = calcular_distancias_braycurtis(abundancias)

    variables = {
        "humedad_relativa": "average-soil-relative-humidity",
        "temperatura": "average-soil-temperature",
        "elevacion": "elevation",
    }

    filas = []
    for nombre, columna in variables.items():
        valores = metadata[columna]
        muestras_validas = valores.dropna().index
        distancias_validas = distancias.loc[
            muestras_validas, muestras_validas
        ]
        f, r2, p_valor, n = permanova_univariada(
            distancias_validas, valores.loc[muestras_validas]
        )
        filas.append(
            {
                "variable": nombre,
                "columna_original": columna,
                "F": f,
                "R2": r2,
                "p_valor": p_valor,
                "n_permutaciones": 999,
                "n": n,
            }
        )

    tabla = pd.DataFrame(filas).sort_values(
        "R2", ascending=False
    ).reset_index(drop=True)
    return tabla


def guardar_permanova(tabla):
    """Imprime y guarda la tabla de PERMANOVA de H3."""
    print("\nPERMANOVA univariada (variables ambientales):")
    print(
        tabla.round(4).to_string(index=False)
    )
    tabla.to_csv(
        os.path.join(
            OUTPUT_DIR_H3, "h3_permanova_variables_ambientales.tsv"
        ),
        sep="\t",
        index=False,
    )
    return tabla


def graficar_permanova(tabla):
    """Barra horizontal de R2 por variable ambiental, con p-valor."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colores = sns.color_palette("Set2", len(tabla))

    barras = ax.barh(
        tabla["variable"], tabla["R2"], color=colores
    )

    for barra, p_valor in zip(barras, tabla["p_valor"]):
        ax.text(
            barra.get_width() + 0.002,
            barra.get_y() + barra.get_height() / 2,
            f"p = {p_valor:.3f}",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel(
        "R2 (proporcion de varianza composicional explicada)"
    )
    ax.set_ylabel("Variable ambiental")
    ax.set_title(
        "PERMANOVA univariada por variable ambiental\n"
        "Desierto de Atacama (Bray-Curtis, 999 permutaciones)"
    )
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            OUTPUT_DIR_H3, "fig3_permanova_variables_ambientales.png"
        ),
        dpi=300,
    )
    fig.savefig(
        os.path.join(
            OUTPUT_DIR_H3, "fig3_permanova_variables_ambientales.pdf"
        )
    )
    plt.close(fig)


def _hat_matrix(x):
    """Matriz de proyeccion (hat matrix) de un modelo lineal."""
    return x @ np.linalg.pinv(x.T @ x) @ x.T


def _suma_cuadrados_modelo(matriz_g, hat):
    """SS explicada por un modelo con matriz de proyeccion 'hat'."""
    return np.trace(hat @ matriz_g @ hat)


def permanova_multivariada(distancias, variables, n_permutaciones=999,
                            semilla=0):
    """
    PERMANOVA multivariada: humedad + temperatura + elevacion
    juntas en un solo modelo (equivalente a adonis2 del paquete
    vegan de R).

    Reporta:
    - El modelo completo (las tres variables juntas): cuanta
      varianza composicional explican en conjunto.
    - La contribucion marginal de cada variable (su R2 al
      quitarla del modelo completo y ver cuanto se pierde),
      equivalente a adonis2(..., by="margin").
    """
    aleatorio = np.random.RandomState(semilla)
    matriz_g = centrar_gower(distancias.values)
    n = matriz_g.shape[0]
    identidad = np.eye(n)
    ss_total = np.trace(matriz_g)

    nombres = list(variables.keys())
    columnas = {
        nombre: variables[nombre].values for nombre in nombres
    }
    p = len(nombres)

    x_completo = np.column_stack(
        [np.ones(n)] + [columnas[nombre] for nombre in nombres]
    )
    hat_completo = _hat_matrix(x_completo)
    ss_modelo_completo = _suma_cuadrados_modelo(
        matriz_g, hat_completo
    )
    ss_residuo_completo = ss_total - ss_modelo_completo
    gl_modelo = p
    gl_residuo = n - p - 1

    f_completo = (
        (ss_modelo_completo / gl_modelo)
        / (ss_residuo_completo / gl_residuo)
    )
    r2_completo = ss_modelo_completo / ss_total

    contador = 0
    for _ in range(n_permutaciones):
        orden = aleatorio.permutation(n)
        hat_permutado = _hat_matrix(x_completo[orden])
        ss_permutada = _suma_cuadrados_modelo(
            matriz_g, hat_permutado
        )
        ss_residuo_permutada = ss_total - ss_permutada
        f_permutado = (
            (ss_permutada / gl_modelo)
            / (ss_residuo_permutada / gl_residuo)
        )
        if f_permutado >= f_completo:
            contador += 1
    p_completo = (contador + 1) / (n_permutaciones + 1)

    fila_completo = {
        "termino": "modelo completo (las 3 variables)",
        "F": f_completo,
        "R2": r2_completo,
        "p_valor": p_completo,
        "n_permutaciones": n_permutaciones,
        "n": n,
    }

    filas_marginales = []
    for nombre in nombres:
        columnas_otras = [
            columnas[otro] for otro in nombres if otro != nombre
        ]
        x_reducido = np.column_stack(
            [np.ones(n)] + columnas_otras
        )
        hat_reducido = _hat_matrix(x_reducido)
        ss_modelo_reducido = _suma_cuadrados_modelo(
            matriz_g, hat_reducido
        )
        ss_marginal = ss_modelo_completo - ss_modelo_reducido
        f_marginal = (
            (ss_marginal / 1) / (ss_residuo_completo / gl_residuo)
        )
        r2_marginal = ss_marginal / ss_total

        contador_marginal = 0
        for _ in range(n_permutaciones):
            orden = aleatorio.permutation(n)
            columnas_permutadas = columnas_otras + [
                columnas[nombre][orden]
            ]
            x_completo_permutado = np.column_stack(
                [np.ones(n)] + columnas_permutadas
            )
            hat_permutado = _hat_matrix(x_completo_permutado)
            ss_permutada = _suma_cuadrados_modelo(
                matriz_g, hat_permutado
            )
            ss_marginal_permutada = (
                ss_permutada - ss_modelo_reducido
            )
            f_permutado = (
                (ss_marginal_permutada / 1)
                / (ss_residuo_completo / gl_residuo)
            )
            if f_permutado >= f_marginal:
                contador_marginal += 1
        p_marginal = (
            (contador_marginal + 1) / (n_permutaciones + 1)
        )

        filas_marginales.append(
            {
                "termino": nombre,
                "F": f_marginal,
                "R2": r2_marginal,
                "p_valor": p_marginal,
                "n_permutaciones": n_permutaciones,
                "n": n,
            }
        )

    filas_marginales = sorted(
        filas_marginales, key=lambda fila: fila["R2"], reverse=True
    )
    tabla = pd.DataFrame([fila_completo] + filas_marginales)
    return tabla


def h3_modelo_multivariado(abundancias, metadata):
    """
    Corre el PERMANOVA multivariado de H3: humedad, temperatura y
    elevacion juntas en un solo modelo, en vez de una a la vez.
    """
    columnas = {
        "humedad_relativa": "average-soil-relative-humidity",
        "temperatura": "average-soil-temperature",
        "elevacion": "elevation",
    }

    muestras_validas = metadata[list(columnas.values())].dropna(
    ).index
    metadata_validas = metadata.loc[muestras_validas]
    abundancias_validas = abundancias[muestras_validas]

    distancias = calcular_distancias_braycurtis(
        abundancias_validas
    )
    variables = {
        nombre: metadata_validas[columna]
        for nombre, columna in columnas.items()
    }

    tabla = permanova_multivariada(distancias, variables)

    print("\nPERMANOVA multivariado (las 3 variables juntas):")
    print(tabla.round(4).to_string(index=False))

    tabla.to_csv(
        os.path.join(
            OUTPUT_DIR_H3,
            "h3_permanova_modelo_multivariado.tsv",
        ),
        sep="\t",
        index=False,
    )
    return tabla


def main_h3():
    abundancias, metadata = cargar_datos()

    tabla = permanova_variables_ambientales(abundancias, metadata)
    guardar_permanova(tabla)
    graficar_permanova(tabla)

    h3_modelo_multivariado(abundancias, metadata)

    print(
        "\nListo. Figura y tabla de H3 guardadas en la carpeta "
        "'outputs3/'."
    )


if __name__ == "__main__":
    main()
    main_h2()
    main_h3()
