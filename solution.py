"""
Paso 1: diversidad alfa (Shannon) vs. humedad relativa del suelo.

Pregunta (H1): ¿a menor humedad relativa del suelo, menor
diversidad de microorganismos?
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# --- 1. Cargar los datos ---------------------------------------

# Tabla de abundancias: filas = OTUs (tipos de microorganismos),
# columnas = muestras de suelo.
abund = pd.read_csv(
    "data/abundancias.tsv", sep="\t", index_col=0
)

# Metadata: una fila por muestra, con transecto y variables
# ambientales.
meta = pd.read_csv(
    "data/metadata.tsv", sep="\t", index_col="sample-id"
)

# Solo el 54 de las 75 muestras de metadata tienen datos de
# abundancia. Nos quedamos con la interseccion de IDs.
muestras_comunes = abund.columns.intersection(meta.index)
abund = abund[muestras_comunes]
meta = meta.loc[muestras_comunes]

print(f"Muestras en comun (con secuenciacion + metadata): "
      f"{len(muestras_comunes)}")

# --- 2. Calcular diversidad de Shannon por muestra --------------


def shannon(conteos):
    """Indice de Shannon a partir de conteos de OTUs de 1 muestra.

    A mayor valor, mas diversa es la comunidad microbiana (mas
    tipos de microorganismos y mas repartidos entre ellos).
    """
    total = conteos.sum()
    if total == 0:
        return np.nan
    proporciones = conteos[conteos > 0] / total
    return -(proporciones * np.log(proporciones)).sum()


shannon_por_muestra = abund.apply(shannon, axis=0)

datos = pd.DataFrame({
    "shannon": shannon_por_muestra,
    "transecto": meta["transect-name"],
    "humedad_suelo": meta["average-soil-relative-humidity"],
})

# --- 3. Resumen estadistico por transecto -----------------------

# Usamos todas las muestras con Shannon valido, aunque a algunas
# les falte el dato de humedad (ese caso se filtra mas abajo, solo
# para la correlacion con humedad).
resumen = datos.groupby("transecto")["shannon"].agg(
    ["mean", "median", "min", "max", "count"]
)
resumen.columns = ["media", "mediana", "minimo", "maximo", "n"]
print("\nResumen de diversidad de Shannon por transecto:")
print(resumen.round(3))

resumen.to_csv(
    "outputs/h1_resumen_shannon_por_transecto.tsv", sep="\t"
)

# --- 4. Figura 1: boxplot de Shannon por transecto ---------------

fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(
    data=datos, x="transecto", y="shannon", hue="transecto",
    palette="Set2", legend=False, ax=ax,
)
sns.stripplot(
    data=datos, x="transecto", y="shannon",
    color="black", alpha=0.5, size=4, ax=ax,
)

ax.set_xlabel("Transecto")
ax.set_ylabel("Diversidad de Shannon (H')")
ax.set_title(
    "Diversidad alfa (Shannon) por transecto\n"
    "Desierto de Atacama"
)
fig.tight_layout()
fig.savefig("outputs/fig1_shannon_por_transecto.png", dpi=300)
fig.savefig("outputs/fig1_shannon_por_transecto.pdf")
plt.close(fig)

# --- 5. Correlacion de Spearman: Shannon vs. humedad -------------

# Aqui si excluimos las muestras sin dato de humedad (3 de
# Baquedano), porque la correlacion los necesita a ambos.
datos_humedad = datos.dropna(subset=["humedad_suelo"])

rho, p_valor = stats.spearmanr(
    datos_humedad["humedad_suelo"], datos_humedad["shannon"]
)
r2 = rho ** 2
n = len(datos_humedad)

tabla_corr = pd.DataFrame({
    "rho_spearman": [rho],
    "r2": [r2],
    "p_valor": [p_valor],
    "n": [n],
})
tabla_corr.to_csv(
    "outputs/h1_correlacion_shannon_vs_humedad.tsv",
    sep="\t", index=False,
)

print("\nCorrelacion de Spearman (Shannon vs. humedad del suelo):")
print(f"  rho = {rho:.3f}, R2 = {r2:.3f}, "
      f"p = {p_valor:.4g}, n = {n}")

if r2 < 0.05:
    print(
        "\nAVISO: R2 < 0.05 -- la relacion es muy debil. Revisar "
        "datos o analisis antes de continuar."
    )

# --- 6. Figura 2: Shannon vs. humedad del suelo (con regresion) --

fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(
    data=datos_humedad, x="humedad_suelo", y="shannon",
    scatter_kws={"alpha": 0.6},
    line_kws={"color": "black"},
    color="teal", ci=95, ax=ax,
)
ax.set_xlabel("Humedad relativa promedio del suelo (%)")
ax.set_ylabel("Diversidad de Shannon (H')")
ax.set_title(
    "Diversidad de Shannon vs. humedad del suelo\n"
    "Desierto de Atacama"
)
texto = (
    f"Spearman rho = {rho:.2f}\n"
    f"R2 = {r2:.2f}\n"
    f"p = {p_valor:.3g}\n"
    f"n = {n}"
)
ax.text(
    0.05, 0.95, texto, transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)
fig.tight_layout()
fig.savefig("outputs/fig1_shannon_vs_avgsoilrh.png", dpi=300)
fig.savefig("outputs/fig1_shannon_vs_avgsoilrh.pdf")
plt.close(fig)

print("\nListo. Figuras y tablas guardadas en outputs/.")
