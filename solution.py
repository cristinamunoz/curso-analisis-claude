"""Figure 2: PCoA (Bray-Curtis) of soil microbial community
composition, colored by average soil relative humidity.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

ABUNDANCE_PATH = "data/abundancias.tsv"
METADATA_PATH = "data/metadata.tsv"
OUTPUT_PNG = "outputs/fig2_pcoa_braycurtis.png"
OUTPUT_PDF = "outputs/fig2_pcoa_braycurtis.pdf"
RICHNESS_SHANNON_PNG = (
    "outputs/fig2_richness_shannon_vs_humidity.png"
)
RICHNESS_SHANNON_PDF = (
    "outputs/fig2_richness_shannon_vs_humidity.pdf"
)
HUMIDITY_COL = "average-soil-relative-humidity"


def load_data():
    abundance = pd.read_csv(ABUNDANCE_PATH, sep="\t", index_col=0)
    metadata = pd.read_csv(METADATA_PATH, sep="\t", index_col=0)
    samples = abundance.columns.intersection(metadata.index)
    abundance = abundance[samples].T
    metadata = metadata.loc[samples]
    return abundance, metadata


def pcoa(distance_matrix):
    n = distance_matrix.shape[0]
    gower = -0.5 * distance_matrix ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    gower_centered = centering @ gower @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gower_centered)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > 0
    coords = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
    variance_explained = eigenvalues[positive] / eigenvalues[positive].sum()
    return coords, variance_explained


def main():
    abundance, metadata = load_data()
    distances = squareform(
        pdist(abundance.values, metric="braycurtis")
    )
    coords, variance_explained = pcoa(distances)

    pc1, pc2 = coords[:, 0], coords[:, 1]
    pc1_pct = variance_explained[0] * 100
    pc2_pct = variance_explained[1] * 100
    humidity = metadata[HUMIDITY_COL].values
    has_humidity = ~pd.isna(humidity)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        pc1[~has_humidity], pc2[~has_humidity],
        c="lightgray", edgecolor="black", linewidth=0.5,
        s=70, label="No humidity data",
    )
    scatter = ax.scatter(
        pc1[has_humidity], pc2[has_humidity],
        c=humidity[has_humidity], cmap="viridis",
        edgecolor="black", linewidth=0.5, s=70,
    )
    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% variance explained)")
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% variance explained)")
    ax.set_title(
        "PCoA (Bray-Curtis) of soil microbial community composition"
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Average soil relative humidity (%)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300)
    fig.savefig(OUTPUT_PDF)


def shannon_diversity(abundance):
    proportions = abundance.div(abundance.sum(axis=1), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = proportions * np.log(proportions)
    terms = terms.replace([np.inf, -np.inf], np.nan).fillna(0)
    return -terms.sum(axis=1)


def plot_trend_line(ax, x, y):
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line, slope * x_line + intercept,
        color="black", linewidth=1,
    )


def annotate_spearman(ax, x, y):
    rho, p_value = spearmanr(x, y)
    label = f"$r_s$ = {rho:.2f}, P = {p_value:.1g}, n = {len(x)}"
    ax.text(
        0.05, 0.92, label, transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
    )


def richness_shannon_figure():
    abundance, metadata = load_data()
    humidity = metadata[HUMIDITY_COL]
    valid = humidity.notna()

    humidity = humidity[valid]
    richness = (abundance > 0).sum(axis=1)[valid]
    shannon = shannon_diversity(abundance)[valid]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(8, 9), sharex=True
    )
    panels = (
        (ax_top, richness, "Observed OTU richness\n(proxy for PD)"),
        (ax_bottom, shannon, "Shannon diversity index"),
    )
    for ax, values, ylabel in panels:
        ax.scatter(
            humidity, values, facecolor="white",
            edgecolor="black", linewidth=0.8, s=40,
        )
        plot_trend_line(ax, humidity.values, values.values)
        annotate_spearman(ax, humidity.values, values.values)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_bottom.set_xlabel("Average soil relative humidity (%)")
    fig.suptitle(
        "Correlation between soil relative humidity and\n"
        "microbial richness and diversity"
    )
    fig.tight_layout()
    fig.savefig(RICHNESS_SHANNON_PNG, dpi=300)
    fig.savefig(RICHNESS_SHANNON_PDF)


if __name__ == "__main__":
    main()
    richness_shannon_figure()
