"""H1: alpha diversity (Shannon) vs. soil relative humidity.

Reproduces the alpha-diversity analysis from Neilson et al.
(2017) for the Atacama Desert soil microbiome dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp
from skbio.diversity import alpha_diversity

COLOR_BAQ = "#2a78d6"
COLOR_YUN = "#e34948"
PALETTE = {"Baquedano": COLOR_BAQ, "Yungay": COLOR_YUN}
ORDER = ["Baquedano", "Yungay"]
OUT = "outputs"

sns.set_theme(style="white", context="talk")


def load_data():
    """Load metadata and OTU table, keep only shared samples."""
    meta = pd.read_csv("data/metadata.tsv", sep="\t")
    abund = pd.read_csv("data/abundancias.tsv", sep="\t")
    abund = abund.rename(columns={"#OTU ID": "otu_id"})

    sample_cols = [c for c in abund.columns if c != "otu_id"]
    shared = sorted(set(meta["sample-id"]) & set(sample_cols))

    meta = meta[meta["sample-id"].isin(shared)].copy()
    abund = abund[["otu_id"] + shared]
    return meta, abund


def compute_shannon(abund):
    """Shannon diversity index (natural log) per sample."""
    counts = abund.set_index("otu_id").T
    values = alpha_diversity(
        "shannon", counts.values, ids=counts.index
    )
    return (
        values.rename("shannon")
        .rename_axis("sample-id")
        .reset_index()
    )


def summarize_by_transect(df):
    """Mean/median/min/max/count of Shannon per transect."""
    return (
        df.groupby("transect-name")["shannon"]
        .agg(mean="mean", median="median", min="min",
             max="max", count="count")
        .reset_index()
    )


def kruskal_dunn(df):
    """Kruskal-Wallis test + Dunn's post-hoc across transects."""
    groups = [
        g["shannon"].values
        for _, g in df.groupby("transect-name")
    ]
    h_stat, p_value = stats.kruskal(*groups)
    kw = pd.DataFrame({
        "test": ["kruskal-wallis"],
        "statistic": [h_stat],
        "p_value": [p_value],
    })

    dunn = sp.posthoc_dunn(
        df, val_col="shannon", group_col="transect-name",
        p_adjust="bonferroni",
    )
    return kw, dunn


def spearman_vs_humidity(df):
    """Spearman correlation between Shannon and soil humidity."""
    sub = df.dropna(subset=["average-soil-relative-humidity"])
    rho, p_value = stats.spearmanr(
        sub["average-soil-relative-humidity"], sub["shannon"]
    )
    result = pd.DataFrame({
        "variable_x": ["average-soil-relative-humidity"],
        "variable_y": ["shannon"],
        "spearman_rho": [rho],
        "p_valor": [p_value],
        "n": [len(sub)],
    })
    return result, sub


def plot_boxplot(df):
    """Boxplot of Shannon diversity by transect."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=df, x="transect-name", y="shannon",
        order=ORDER, hue="transect-name", palette=PALETTE,
        legend=False, width=0.5, ax=ax,
    )
    sns.stripplot(
        data=df, x="transect-name", y="shannon", order=ORDER,
        color="black", alpha=0.5, size=5, ax=ax,
    )
    ax.set_xlabel("Transect")
    ax.set_ylabel("Shannon diversity index (H')")
    ax.set_title(
        "Alpha diversity by transect\n"
        "Atacama Desert soil microbiome"
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_shannon_por_transecto.png", dpi=300)
    fig.savefig(f"{OUT}/fig1_shannon_por_transecto.pdf")
    plt.close(fig)


def plot_scatter(sub, rho, p_value):
    """Scatter of Shannon vs. soil humidity with linear fit."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fit = stats.linregress(
        sub["average-soil-relative-humidity"], sub["shannon"]
    )
    sns.regplot(
        data=sub, x="average-soil-relative-humidity",
        y="shannon", ax=ax, ci=95,
        scatter_kws={"alpha": 0}, line_kws={"color": "#52514e"},
    )
    for name, color in PALETTE.items():
        group = sub[sub["transect-name"] == name]
        ax.scatter(
            group["average-soil-relative-humidity"],
            group["shannon"], color=color, s=60,
            edgecolor="white", label=name, zorder=3,
        )
    ax.set_xlabel("Average soil relative humidity (%)")
    ax.set_ylabel("Shannon diversity index (H')")
    ax.set_title("Alpha diversity vs. soil relative humidity")
    text = (
        f"Spearman $\\rho$ = {rho:.2f}, p = {p_value:.4f}\n"
        f"Linear fit $R^2$ = {fit.rvalue ** 2:.2f}"
    )
    ax.text(
        0.03, 0.97, text, transform=ax.transAxes,
        va="top", ha="left", fontsize=11,
    )
    ax.legend(title="Transect", loc="lower right")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(
        f"{OUT}/fig1_shannon_vs_avgsoilrh.png", dpi=300
    )
    fig.savefig(f"{OUT}/fig1_shannon_vs_avgsoilrh.pdf")
    plt.close(fig)


def main():
    meta, abund = load_data()
    shannon = compute_shannon(abund)
    df = meta.merge(shannon, on="sample-id", how="inner")

    summary = summarize_by_transect(df)
    summary.to_csv(
        f"{OUT}/h1_resumen_shannon_por_transecto.tsv",
        sep="\t", index=False,
    )

    kw, dunn = kruskal_dunn(df)
    kw.to_csv(
        f"{OUT}/h1_kruskal_wallis.tsv", sep="\t", index=False
    )
    dunn.to_csv(f"{OUT}/h1_dunn_posthoc.tsv", sep="\t")

    corr, sub = spearman_vs_humidity(df)
    corr.to_csv(
        f"{OUT}/h1_correlacion_shannon_vs_humedad.tsv",
        sep="\t", index=False,
    )

    plot_boxplot(df)
    plot_scatter(
        sub, corr["spearman_rho"].iloc[0],
        corr["p_valor"].iloc[0],
    )

    print("== Resumen Shannon por transecto ==")
    print(summary.to_string(index=False))
    print()
    print("== Kruskal-Wallis ==")
    print(kw.to_string(index=False))
    print()
    print("== Dunn's post-hoc (p ajustado, Bonferroni) ==")
    print(dunn.to_string())
    print()
    print("== Spearman: Shannon vs. humedad relativa ==")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
