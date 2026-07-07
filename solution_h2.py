"""H2: community composition (beta diversity), Bray-Curtis.

Reproduces the beta-diversity analysis from Neilson et al.
(2017) for the Atacama Desert soil microbiome dataset: PCoA
(paper's method, Fig. S2) plus NMDS, PERMANOVA, ANOSIM and
PERMDISP (beta dispersion), all on Bray-Curtis distances at
the OTU level, grouped by transect (Baquedano vs. Yungay).
"""
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa
from skbio.stats.distance import permanova, anosim, permdisp

COLOR_BAQ = "#2a78d6"
COLOR_YUN = "#e34948"
PALETTE = {"Baquedano": COLOR_BAQ, "Yungay": COLOR_YUN}
ORDER = ["Baquedano", "Yungay"]
OUT = "outputs"
N_PERMUTATIONS = 999

sns.set_theme(style="white", context="talk")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_data():
    """Load metadata and OTU table, keep only shared samples."""
    meta = pd.read_csv("data/metadata.tsv", sep="\t")
    abund = pd.read_csv("data/abundancias.tsv", sep="\t")
    abund = abund.rename(columns={"#OTU ID": "otu_id"})

    sample_cols = [c for c in abund.columns if c != "otu_id"]
    shared = sorted(set(meta["sample-id"]) & set(sample_cols))

    meta = meta[meta["sample-id"].isin(shared)].copy()
    meta = meta.set_index("sample-id").loc[shared].reset_index()
    abund = abund[["otu_id"] + shared]
    return meta, abund


def bray_curtis_matrix(abund, shared):
    """Bray-Curtis distance matrix at the OTU level."""
    counts = abund.set_index("otu_id").T.loc[shared]
    return beta_diversity("braycurtis", counts.values, ids=shared)


def run_pcoa(dm):
    """PCoA ordination; returns sample scores + variance table."""
    ordination = pcoa(dm)
    scores = ordination.samples[["PC1", "PC2"]].copy()
    scores.index.name = "sample-id"
    variance = pd.DataFrame({
        "eje": ["PC1", "PC2"],
        "porcentaje_varianza": [
            ordination.proportion_explained["PC1"] * 100,
            ordination.proportion_explained["PC2"] * 100,
        ],
    })
    return scores.reset_index(), variance


def run_nmds(dm):
    """Non-metric MDS ordination on the same distance matrix."""
    mds = MDS(
        n_components=2, metric=False,
        dissimilarity="precomputed", normalized_stress=True,
        random_state=42, n_init=10,
    )
    coords = mds.fit_transform(dm.data)
    scores = pd.DataFrame(
        coords, columns=["NMDS1", "NMDS2"], index=dm.ids
    )
    scores.index.name = "sample-id"
    stress = pd.DataFrame({"stress": [mds.stress_]})
    return scores.reset_index(), stress


def run_group_tests(dm, grouping):
    """PERMANOVA, ANOSIM and PERMDISP by transect."""
    perm = permanova(
        dm, grouping, permutations=N_PERMUTATIONS
    )
    ano = anosim(dm, grouping, permutations=N_PERMUTATIONS)
    disp = permdisp(dm, grouping, permutations=N_PERMUTATIONS)
    return perm.to_frame().T, ano.to_frame().T, disp.to_frame().T


def plot_ordination(scores, meta, x, y, title, fname):
    """Scatter of an ordination colored by transect."""
    df = scores.merge(
        meta[["sample-id", "transect-name"]], on="sample-id"
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, color in PALETTE.items():
        group = df[df["transect-name"] == name]
        ax.scatter(
            group[x], group[y], color=color, s=70,
            edgecolor="white", label=name, zorder=3,
        )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(title="Transect")
    ax.axhline(0, color="#c3c2b7", lw=1, zorder=1)
    ax.axvline(0, color="#c3c2b7", lw=1, zorder=1)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}.png", dpi=300)
    fig.savefig(f"{OUT}/{fname}.pdf")
    plt.close(fig)


def main():
    meta, abund = load_data()
    shared = meta["sample-id"].tolist()
    dm = bray_curtis_matrix(abund, shared)

    pcoa_scores, pcoa_var = run_pcoa(dm)
    pcoa_var.to_csv(
        f"{OUT}/h2_varianza_explicada_pcoa.tsv",
        sep="\t", index=False,
    )
    plot_ordination(
        pcoa_scores, meta, "PC1", "PC2",
        "PCoA of Bray-Curtis distances\n"
        "Atacama Desert soil microbiome",
        "fig2_pcoa_braycurtis",
    )

    nmds_scores, nmds_stress = run_nmds(dm)
    nmds_stress.to_csv(
        f"{OUT}/h2_nmds_stress.tsv", sep="\t", index=False
    )
    plot_ordination(
        nmds_scores, meta, "NMDS1", "NMDS2",
        "NMDS of Bray-Curtis distances\n"
        "Atacama Desert soil microbiome",
        "fig2_nmds_braycurtis",
    )

    grouping = (
        meta.set_index("sample-id")["transect-name"]
        .reindex(list(dm.ids))
    )
    perm, ano, disp = run_group_tests(dm, grouping)
    perm.to_csv(
        f"{OUT}/h2_permanova.tsv", sep="\t", index=False
    )
    ano.to_csv(f"{OUT}/h2_anosim.tsv", sep="\t", index=False)
    disp.to_csv(
        f"{OUT}/h2_betadisper.tsv", sep="\t", index=False
    )

    print("== PCoA variance explained ==")
    print(pcoa_var.to_string(index=False))
    print()
    print("== NMDS stress ==")
    print(nmds_stress.to_string(index=False))
    print()
    print("== PERMANOVA (transect-name, 999 perms) ==")
    print(perm.to_string(index=False))
    print()
    print("== ANOSIM (transect-name, 999 perms) ==")
    print(ano.to_string(index=False))
    print()
    print("== PERMDISP / betadisper (transect-name) ==")
    print(disp.to_string(index=False))


if __name__ == "__main__":
    main()
