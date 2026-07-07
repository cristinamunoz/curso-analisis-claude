import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- 1. Cargar datos ---
abundancias = pd.read_csv("data/abundancias.tsv", sep="\t", index_col=0)
metadata = pd.read_csv("data/metadata.tsv", sep="\t", index_col=0)

# Transponer: filas = muestras, columnas = OTUs
abundancias = abundancias.T

# --- 2. Filtrar muestras comunes (solo las 54 con ambos archivos) ---
muestras_comunes = abundancias.index.intersection(metadata.index)
abundancias = abundancias.loc[muestras_comunes]
metadata = metadata.loc[muestras_comunes]

print(f"Muestras con datos completos: {len(muestras_comunes)}")

# --- 3. Calcular indice de Shannon por muestra ---
def shannon(counts):
    counts = counts[counts > 0]
    proporciones = counts / counts.sum()
    return -np.sum(proporciones * np.log(proporciones))

metadata = metadata.copy()
metadata["shannon"] = abundancias.apply(shannon, axis=1)

# --- 4. Resumen estadistico por transecto ---
resumen = metadata.groupby("transect-name")["shannon"].agg(
    ["mean", "median", "min", "max", "std", "count"]
)
resumen.columns = ["Media", "Mediana", "Min", "Max", "DE", "n"]
print("\nResumen de diversidad Shannon por transecto:")
print(resumen.round(3))

# --- 5. Correlacion de Spearman (Shannon vs humedad) ---
col_humedad = "average-soil-relative-humidity"
datos_validos = metadata[["shannon", col_humedad]].dropna()
humedad = datos_validos[col_humedad]
shannon_vals = datos_validos["shannon"]

rho, pval = stats.spearmanr(humedad, shannon_vals)
n = len(datos_validos)
print(f"\nCorrelacion Spearman (Shannon vs humedad del suelo):")
print(f"  rho = {rho:.3f}, p = {pval:.4f}, n = {n}")

# --- 6. Boxplot por transecto ---
fig, ax = plt.subplots(figsize=(8, 6))
grupos = [
    metadata[metadata["transect-name"] == t]["shannon"].dropna()
    for t in ["Baquedano", "Yungay"]
]
bp = ax.boxplot(
    grupos,
    tick_labels=["Baquedano\n(mas humedo)", "Yungay\n(mas arido)"],
    patch_artist=True
)
colores = ["#66c2a5", "#fc8d62"]
for patch, color in zip(bp["boxes"], colores):
    patch.set_facecolor(color)
ax.set_ylabel("Indice de Shannon")
ax.set_title("Diversidad alfa por transecto\n(Desierto de Atacama)")
plt.tight_layout()
plt.savefig("outputs/fig1_shannon_por_transecto.png", dpi=300)
plt.savefig("outputs/fig1_shannon_por_transecto.pdf")
plt.close()
print("Figura guardada: outputs/fig1_shannon_por_transecto.png/.pdf")

# --- 7. Scatter: Shannon vs humedad con linea de regresion ---
fig, ax = plt.subplots(figsize=(8, 6))
colores_transecto = metadata.loc[
    datos_validos.index, "transect-name"
].map({"Baquedano": "#66c2a5", "Yungay": "#fc8d62"})

ax.scatter(
    humedad, shannon_vals,
    c=colores_transecto, alpha=0.8,
    edgecolors="grey", linewidths=0.5, s=60
)

# Linea de regresion + banda de confianza 95%
slope, intercept, r, p_lin, se = stats.linregress(humedad, shannon_vals)
x_line = np.linspace(humedad.min(), humedad.max(), 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, color="black", linewidth=1.5)

y_err = se * np.sqrt(
    1 / n + (x_line - humedad.mean()) ** 2
    / ((n - 1) * humedad.std() ** 2)
)
t_crit = stats.t.ppf(0.975, df=n - 2)
ax.fill_between(
    x_line,
    y_line - t_crit * y_err,
    y_line + t_crit * y_err,
    alpha=0.2, color="black"
)

r2 = r ** 2
ax.set_xlabel("Humedad relativa promedio del suelo (%)")
ax.set_ylabel("Indice de Shannon")
ax.set_title(
    f"Diversidad alfa vs. humedad del suelo\n"
    f"Spearman rho = {rho:.2f}, p = {pval:.4f}, R2 = {r2:.2f}"
)
parches = [
    mpatches.Patch(color="#66c2a5", label="Baquedano"),
    mpatches.Patch(color="#fc8d62", label="Yungay")
]
ax.legend(handles=parches)
plt.tight_layout()
plt.savefig("outputs/fig1_shannon_vs_avgsoilrh.png", dpi=300)
plt.savefig("outputs/fig1_shannon_vs_avgsoilrh.pdf")
plt.close()
print("Figura guardada: outputs/fig1_shannon_vs_avgsoilrh.png/.pdf")
