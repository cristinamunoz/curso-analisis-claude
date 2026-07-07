"""Figure 2: PCoA (Bray-Curtis) of soil microbial community
composition, colored by average soil relative humidity.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

ABUNDANCE_PATH = "data/abundancias.tsv"
METADATA_PATH = "data/metadata.tsv"
OUTPUT_PNG = "outputs/fig2_pcoa_braycurtis.png"
OUTPUT_PDF = "outputs/fig2_pcoa_braycurtis.pdf"
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


if __name__ == "__main__":
    main()
