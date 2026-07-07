"""Analisis H1: diversidad Shannon vs humedad relativa del suelo.

Calcula el indice de Shannon por muestra a partir de la tabla de
abundancias de OTUs, lo cruza con la metadata ambiental, y evalua
la correlacion con la humedad relativa promedio del suelo
(AvgSoilRH), siguiendo el metodo del paper (correlacion de
Spearman). Ademas ajusta una regresion lineal para el grafico.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = "data"
OUT_DIR = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)


def shannon_index(counts):
    """Calcula el indice de Shannon (base natural) de una muestra."""
    counts = counts[counts > 0]
    total = counts.sum()
    if total == 0:
        return np.nan
    proportions = counts / total
    return -np.sum(proportions * np.log(proportions))


def main():
    abund = pd.read_csv(
        os.path.join(DATA_DIR, "abundancias.tsv"),
        sep="\t",
        index_col=0,
    )
    metadata = pd.read_csv(
        os.path.join(DATA_DIR, "metadata.tsv"),
        sep="\t",
    )

    # Un valor de Shannon por muestra (columna de la tabla de OTUs).
    shannon = abund.apply(shannon_index, axis=0)
    shannon.name = "shannon"
    shannon_df = shannon.reset_index()
    shannon_df.columns = ["sample-id", "shannon"]

    # Cruce con metadata: solo muestras presentes en ambos archivos.
    merged = metadata.merge(shannon_df, on="sample-id", how="inner")
    merged = merged.dropna(subset=["shannon"])

    # Para la correlacion y la regresion se necesita ademas que
    # la humedad relativa de suelo no sea nula (excluye BAQ4697,
    # sitio donde se perdio el logger ambiental).
    merged_reg = merged.dropna(subset=["average-soil-relative-humidity"])

    print("Numero de muestras con Shannon (n):", len(merged))
    print(
        "Numero de muestras con Shannon y AvgSoilRH (n):",
        len(merged_reg),
    )

    # 1) Resumen por transecto (todas las muestras con Shannon).
    summary = merged.groupby("transect-name")["shannon"].agg(
        mean="mean",
        median="median",
        min="min",
        max="max",
        count="count",
    )
    print("\nResumen de diversidad Shannon por transecto:")
    print(summary)
    summary.to_csv(
        os.path.join(OUT_DIR, "h1_resumen_shannon_por_transecto.tsv"),
        sep="\t",
    )

    # 2) Boxplot por transecto (antes de la regresion).
    fig, ax = plt.subplots(figsize=(8, 6))
    groups = ["Baquedano", "Yungay"]
    colors = plt.get_cmap("Set2").colors
    data_by_group = [
        merged.loc[merged["transect-name"] == g, "shannon"]
        for g in groups
    ]
    bplot = ax.boxplot(
        data_by_group,
        tick_labels=groups,
        patch_artist=True,
    )
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Transect")
    ax.set_ylabel("Shannon diversity index")
    ax.set_title("Shannon diversity by transect")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT_DIR, "fig1_shannon_por_transecto.png"), dpi=300
    )
    fig.savefig(
        os.path.join(OUT_DIR, "fig1_shannon_por_transecto.pdf")
    )
    plt.close(fig)

    # 3) Correlacion de Spearman (mismo metodo que el paper).
    x = merged_reg["average-soil-relative-humidity"].to_numpy()
    y = merged_reg["shannon"].to_numpy()
    rho, p_spearman = stats.spearmanr(x, y)

    # 4) Regresion lineal (para la recta y el R2 del grafico).
    lin = stats.linregress(x, y)
    r_squared = lin.rvalue ** 2

    print("\nCorrelacion de Spearman (shannon vs AvgSoilRH):")
    print(f"  rho = {rho:.4f}, p-valor = {p_spearman:.6f}, n = {len(x)}")
    print("\nRegresion lineal (shannon ~ AvgSoilRH):")
    print(
        f"  R2 = {r_squared:.4f}, p-valor = {lin.pvalue:.6f}, "
        f"pendiente = {lin.slope:.4f}"
    )

    corr_table = pd.DataFrame(
        [{
            "variable_x": "average-soil-relative-humidity",
            "variable_y": "shannon",
            "spearman_rho": rho,
            "p_valor": p_spearman,
            "n": len(x),
        }]
    )
    corr_table.to_csv(
        os.path.join(OUT_DIR, "h1_correlacion_shannon_vs_humedad.tsv"),
        sep="\t",
        index=False,
    )

    # 5) Scatter plot con regresion y banda de confianza al 95%.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        x, y, c="#3B4CC0", alpha=0.8, edgecolor="white", label="Samples"
    )

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = lin.intercept + lin.slope * x_line

    # Banda de confianza al 95% para la recta de regresion.
    n = len(x)
    dof = n - 2
    t_val = stats.t.ppf(0.975, dof)
    x_mean = x.mean()
    s_err = np.sqrt(
        np.sum((y - (lin.intercept + lin.slope * x)) ** 2) / dof
    )
    se_line = s_err * np.sqrt(
        1 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)
    )
    ci = t_val * se_line

    ax.plot(x_line, y_line, color="#B40426", label="Linear fit")
    ax.fill_between(
        x_line,
        y_line - ci,
        y_line + ci,
        color="#B40426",
        alpha=0.2,
        label="95% CI",
    )

    text = (
        f"R2 = {r_squared:.3f}\n"
        f"p = {lin.pvalue:.4f}\n"
        f"Spearman rho = {rho:.3f}\n"
        f"n = {n}"
    )
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    ax.set_xlabel("Average soil relative humidity (%)")
    ax.set_ylabel("Shannon diversity index")
    ax.set_title("Shannon diversity vs. average soil relative humidity")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT_DIR, "fig1_shannon_vs_avgsoilrh.png"), dpi=300
    )
    fig.savefig(
        os.path.join(OUT_DIR, "fig1_shannon_vs_avgsoilrh.pdf")
    )
    plt.close(fig)

    if r_squared < 0.05:
        print(
            "\nAVISO: R2 < 0.05, la relacion lineal es muy debil. "
            "Revisar posibles problemas con los datos o el analisis."
        )


if __name__ == "__main__":
    main()
