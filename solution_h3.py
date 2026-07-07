"""H3: variance in community composition explained by
environmental variables.

Compares how much of the Bray-Curtis compositional variance
is explained by soil relative humidity, temperature and
elevation, via a PERMANOVA/adonis-style test with a
continuous predictor (McArdle & Anderson, 2001) on Bray-Curtis
distances at the OTU level. Each variable is tested on its own
(univariate), 999 permutations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from solution_h2 import load_data, bray_curtis_matrix

OUT = "outputs"
N_PERMUTATIONS = 999
SEED = 42

VARIABLES = [
    ("humedad_relativa", "average-soil-relative-humidity"),
    ("temperatura", "average-soil-temperature"),
    ("elevacion", "elevation"),
]

VAR_LABELS = {
    "humedad_relativa": "Soil relative\nhumidity",
    "temperatura": "Soil\ntemperature",
    "elevacion": "Elevation",
}
VAR_COLORS = {
    "humedad_relativa": "#2a78d6",
    "temperatura": "#1baf7a",
    "elevacion": "#eda100",
}

sns.set_theme(style="white", context="talk")


def gower_center(dm, ids):
    """Double-centered Gower matrix from a distance submatrix."""
    d = dm.filter(ids).data
    a = -0.5 * d ** 2
    row_mean = a.mean(axis=1, keepdims=True)
    col_mean = a.mean(axis=0, keepdims=True)
    grand_mean = a.mean()
    return a - row_mean - col_mean + grand_mean


def _model_ss(g, values):
    """Sum of squares explained by an intercept + predictor."""
    n = g.shape[0]
    design = np.column_stack([np.ones(n), values])
    hat = design @ np.linalg.pinv(design.T @ design) @ design.T
    return np.trace(hat @ g)


def permanova_continuous(g, x, permutations, seed):
    """PERMANOVA (adonis-style) F, R2 and p-value for one
    continuous predictor against a Gower-centered matrix."""
    n = g.shape[0]
    ss_total = np.trace(g)
    ss_model = _model_ss(g, x)
    ss_resid = ss_total - ss_model
    f_obs = (ss_model / 1) / (ss_resid / (n - 2))
    r2 = ss_model / ss_total

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        x_perm = rng.permutation(x)
        ss_perm = _model_ss(g, x_perm)
        f_perm = (ss_perm / 1) / (
            (ss_total - ss_perm) / (n - 2)
        )
        if f_perm >= f_obs:
            exceed += 1
    p_value = (exceed + 1) / (permutations + 1)
    return f_obs, r2, p_value


def plot_r2_comparison(result):
    """Bar chart comparing R2 explained by each variable."""
    fig, ax = plt.subplots(figsize=(8, 6))
    order = result["variable"].tolist()
    colors = [VAR_COLORS[v] for v in order]
    labels = [VAR_LABELS[v] for v in order]
    bars = ax.bar(
        labels, result["R2"], color=colors,
        edgecolor="white", width=0.6,
    )
    for bar, (_, row) in zip(bars, result.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"R²={row['R2']:.3f}\np={row['p_valor']:.3f}",
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_ylabel(
        "Variance explained in community\ncomposition (PERMANOVA R²)"
    )
    ax.set_xlabel("Environmental variable")
    ax.set_title(
        "Compositional variance explained by\n"
        "environmental variables (univariate)"
    )
    ax.set_ylim(0, max(result["R2"]) * 1.35)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_r2_variables_ambientales.png", dpi=300)
    fig.savefig(f"{OUT}/fig3_r2_variables_ambientales.pdf")
    plt.close(fig)


def main():
    meta, abund = load_data()
    shared = meta["sample-id"].tolist()
    dm = bray_curtis_matrix(abund, shared)
    meta = meta.set_index("sample-id")

    rows = []
    for label, column in VARIABLES:
        sub = meta[column].dropna()
        ids = [i for i in dm.ids if i in sub.index]
        g = gower_center(dm, ids)
        x = sub.loc[ids].to_numpy(dtype=float)
        f_obs, r2, p_value = permanova_continuous(
            g, x, N_PERMUTATIONS, SEED
        )
        rows.append({
            "variable": label,
            "columna_original": column,
            "F": f_obs,
            "R2": r2,
            "p_valor": p_value,
            "n_permutaciones": N_PERMUTATIONS,
            "n": len(ids),
        })

    result = pd.DataFrame(rows).sort_values(
        "R2", ascending=False
    )
    result.to_csv(
        f"{OUT}/h3_permanova_variables_ambientales.tsv",
        sep="\t", index=False,
    )
    plot_r2_comparison(result)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
