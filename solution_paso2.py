"""
Paso 2: composicion de comunidades (Bray-Curtis + PCoA).

Pregunta (H2): los sitios mas aridos (Yungay) se separan de los
menos aridos (Baquedano) a lo largo del primer eje de la
ordenacion?
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# --- 1. Cargar datos (igual que en el paso 1) --------------------

abund = pd.read_csv(
    "data/abundancias.tsv", sep="\t", index_col=0
)
meta = pd.read_csv(
    "data/metadata.tsv", sep="\t", index_col="sample-id"
)

muestras_comunes = abund.columns.intersection(meta.index)
abund = abund[muestras_comunes]
meta = meta.loc[muestras_comunes]

# --- 2. Distancia Bray-Curtis entre muestras ---------------------

# pdist espera muestras en filas y OTUs en columnas -> transponer.
matriz_muestras = abund.T.values
distancias = pdist(matriz_muestras, metric="braycurtis")
dist_cuadrada = squareform(distancias)


# --- 3. PCoA (escalamiento multidimensional clasico) -------------

def pcoa(dist_cuadrada):
    """PCoA clasico (metodo de Gower) desde una matriz de
    distancias. Devuelve autovalores y autovectores ordenados de
    mayor a menor varianza explicada."""
    n = dist_cuadrada.shape[0]
    d2 = dist_cuadrada ** 2
    centrado = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * centrado @ d2 @ centrado  # doble centrado
    valores, vectores = np.linalg.eigh(b)
    orden = np.argsort(valores)[::-1]
    return valores[orden], vectores[:, orden]


valores, vectores = pcoa(dist_cuadrada)

# Autovalores negativos no representan varianza real (artefacto
# numerico de Bray-Curtis, que no es una distancia euclidiana).
valores_pos = np.clip(valores, 0, None)
porcentaje_var = 100 * valores_pos / valores_pos.sum()

coords = vectores * np.sqrt(valores_pos)
pcoa_df = pd.DataFrame(
    coords[:, :2], columns=["PC1", "PC2"], index=abund.columns
)
pcoa_df["transecto"] = meta["transect-name"]
pcoa_df["humedad_suelo"] = meta[
    "average-soil-relative-humidity"
]

# --- 4. Tabla de varianza explicada -------------------------------

tabla_var = pd.DataFrame({
    "eje": ["PC1", "PC2"],
    "porcentaje_varianza": porcentaje_var[:2],
})
tabla_var.to_csv(
    "outputs/h2_varianza_explicada_pcoa.tsv",
    sep="\t", index=False,
)
print("Varianza explicada por eje:")
print(tabla_var.round(2))

# --- 5. Figura: PCoA coloreado por humedad, forma por transecto --

fig, ax = plt.subplots(figsize=(8, 6))

con_humedad = pcoa_df.dropna(subset=["humedad_suelo"])
sin_humedad = pcoa_df[pcoa_df["humedad_suelo"].isna()]

marcadores = {"Baquedano": "o", "Yungay": "^"}
sc = None
for transecto, marcador in marcadores.items():
    sub = con_humedad[con_humedad["transecto"] == transecto]
    sc = ax.scatter(
        sub["PC1"], sub["PC2"], c=sub["humedad_suelo"],
        cmap="viridis", marker=marcador, s=80,
        edgecolor="black", linewidth=0.5, label=transecto,
        vmin=con_humedad["humedad_suelo"].min(),
        vmax=con_humedad["humedad_suelo"].max(),
    )

# Las 3 muestras sin dato de humedad se marcan en gris, aparte.
if len(sin_humedad) > 0:
    for transecto, marcador in marcadores.items():
        sub = sin_humedad[sin_humedad["transecto"] == transecto]
        ax.scatter(
            sub["PC1"], sub["PC2"], c="lightgray",
            marker=marcador, s=80, edgecolor="black",
            linewidth=0.5,
            label=f"{transecto} (sin dato de humedad)",
        )

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Humedad relativa del suelo (%)")

ax.set_xlabel(f"PC1 ({porcentaje_var[0]:.1f}% de la varianza)")
ax.set_ylabel(f"PC2 ({porcentaje_var[1]:.1f}% de la varianza)")
ax.set_title(
    "PCoA (Bray-Curtis) de comunidades microbianas\n"
    "Desierto de Atacama"
)
ax.legend(title="Transecto", fontsize=8)
fig.tight_layout()
fig.savefig("outputs/fig2_pcoa_braycurtis.png", dpi=300)
fig.savefig("outputs/fig2_pcoa_braycurtis.pdf")
plt.close(fig)

print("\nListo. Figura y tabla guardadas en outputs/.")
