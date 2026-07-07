"""Analisis H1-H3: diversidad y composicion del microbioma.

H1: diversidad Shannon vs humedad relativa del suelo (AvgSoilRH).
H2: composicion de comunidades (Bray-Curtis + PCoA) coloreada por
el gradiente de AvgSoilRH.
H3: modelado de la composicion en funcion de variables ambientales
(humedad, temperatura, elevacion) via PERMANOVA (McArdle y
Anderson, 999 permutaciones).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

DATA_DIR = "data"
OUT_DIR = "outputs"
N_PERMUTATIONS = 999
RANDOM_SEED = 0

os.makedirs(OUT_DIR, exist_ok=True)


def shannon_index(counts):
    """Calcula el indice de Shannon (base natural) de una muestra."""
    counts = counts[counts > 0]
    total = counts.sum()
    if total == 0:
        return np.nan
    proportions = counts / total
    return -np.sum(proportions * np.log(proportions))


def load_data():
    """Carga la tabla de OTUs y la metadata ambiental."""
    abund = pd.read_csv(
        os.path.join(DATA_DIR, "abundancias.tsv"),
        sep="\t",
        index_col=0,
    )
    metadata = pd.read_csv(
        os.path.join(DATA_DIR, "metadata.tsv"),
        sep="\t",
    )
    return abund, metadata


def run_h1(abund, metadata):
    """H1: diversidad Shannon vs humedad relativa del suelo."""
    print("=" * 60)
    print("H1: diversidad Shannon vs AvgSoilRH")
    print("=" * 60)

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
    plt.close(fig)

    if r_squared < 0.05:
        print(
            "\nAVISO: R2 < 0.05, la relacion lineal es muy debil. "
            "Revisar posibles problemas con los datos o el analisis."
        )


def bray_curtis_distance(abund):
    """Matriz de disimilitud Bray-Curtis entre muestras (columnas)."""
    sample_matrix = abund.T.to_numpy()
    dist = squareform(pdist(sample_matrix, metric="braycurtis"))
    return dist, list(abund.columns)


def pcoa(dist_matrix):
    """PCoA clasico (Gower) via doble centrado de la matriz de
    distancias al cuadrado. Devuelve las coordenadas de las
    muestras y el porcentaje de varianza explicado por cada eje
    con autovalor positivo."""
    n = dist_matrix.shape[0]
    d2 = dist_matrix ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    gower = -0.5 * centering @ d2 @ centering

    eigvals, eigvecs = np.linalg.eigh(gower)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = eigvals > 1e-8
    coords = eigvecs[:, positive] * np.sqrt(eigvals[positive])
    variance_pct = eigvals[positive] / eigvals[positive].sum() * 100
    return coords, variance_pct


def permanova(dist_matrix, predictor, n_perm=N_PERMUTATIONS, seed=0):
    """PERMANOVA univariado (McArdle y Anderson, 2001) para un
    predictor ambiental continuo, evaluado sobre una matriz de
    distancias. Devuelve el pseudo-F observado, el R2 y el
    p-valor obtenido por permutacion."""
    n = dist_matrix.shape[0]
    d2 = dist_matrix ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    gower = -0.5 * centering @ d2 @ centering
    ss_total = np.trace(gower)

    def pseudo_f(values):
        design = np.column_stack([np.ones(n), values])
        xtx_inv = np.linalg.inv(design.T @ design)
        hat = design @ xtx_inv @ design.T
        ss_model = np.trace(hat @ gower)
        ss_resid = ss_total - ss_model
        df_model = 1
        df_resid = n - 2
        f_stat = (ss_model / df_model) / (ss_resid / df_resid)
        r2 = ss_model / ss_total
        return f_stat, r2

    f_obs, r2_obs = pseudo_f(predictor)

    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_perm):
        permuted = rng.permutation(predictor)
        f_perm, _ = pseudo_f(permuted)
        if f_perm >= f_obs - 1e-12:
            count_ge += 1
    p_value = (count_ge + 1) / (n_perm + 1)

    return f_obs, r2_obs, p_value


def permanova_multivariate(
    dist_matrix, predictors, n_perm=N_PERMUTATIONS, seed=0
):
    """PERMANOVA multivariado (McArdle y Anderson) con varios
    predictores ambientales continuos evaluados en conjunto.
    predictors: array de forma (n_muestras, n_variables).
    Devuelve el pseudo-F, R2 y p-valor del modelo completo."""
    n = dist_matrix.shape[0]
    d2 = dist_matrix ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    gower = -0.5 * centering @ d2 @ centering
    ss_total = np.trace(gower)
    n_vars = predictors.shape[1]

    def pseudo_f(values):
        design = np.column_stack([np.ones(n), values])
        xtx_inv = np.linalg.inv(design.T @ design)
        hat = design @ xtx_inv @ design.T
        ss_model = np.trace(hat @ gower)
        ss_resid = ss_total - ss_model
        df_model = n_vars
        df_resid = n - n_vars - 1
        f_stat = (ss_model / df_model) / (ss_resid / df_resid)
        r2 = ss_model / ss_total
        return f_stat, r2

    f_obs, r2_obs = pseudo_f(predictors)

    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_perm):
        perm_idx = rng.permutation(n)
        permuted = predictors[perm_idx, :]
        f_perm, _ = pseudo_f(permuted)
        if f_perm >= f_obs - 1e-12:
            count_ge += 1
    p_value = (count_ge + 1) / (n_perm + 1)

    return f_obs, r2_obs, p_value


def run_h2_h3(abund, metadata):
    """H2: PCoA Bray-Curtis coloreado por AvgSoilRH.
    H3: PERMANOVA univariado de humedad, temperatura y elevacion.
    """
    print("\n" + "=" * 60)
    print("H2/H3: composicion (Bray-Curtis) vs variables ambientales")
    print("=" * 60)

    dist, sample_ids = bray_curtis_distance(abund)
    print("Numero de muestras en la matriz Bray-Curtis:", len(sample_ids))

    coords, variance_pct = pcoa(dist)
    pc1_pct, pc2_pct = variance_pct[0], variance_pct[1]
    print(f"\nVarianza explicada: PC1 = {pc1_pct:.2f}%, "
          f"PC2 = {pc2_pct:.2f}%")

    variance_table = pd.DataFrame(
        [
            {"eje": "PC1", "porcentaje_varianza": pc1_pct},
            {"eje": "PC2", "porcentaje_varianza": pc2_pct},
        ]
    )
    variance_table.to_csv(
        os.path.join(OUT_DIR, "h2_varianza_explicada_pcoa.tsv"),
        sep="\t",
        index=False,
    )

    # Humedad relativa de suelo alineada al orden de sample_ids,
    # para colorear el PCoA con un gradiente continuo (viridis).
    meta_indexed = metadata.set_index("sample-id")
    avgsoilrh = meta_indexed.reindex(sample_ids)[
        "average-soil-relative-humidity"
    ]
    has_rh = avgsoilrh.notna().to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    # Muestras sin dato de humedad: puntos grises huecos.
    ax.scatter(
        coords[~has_rh, 0],
        coords[~has_rh, 1],
        facecolors="none",
        edgecolors="gray",
        label="No AvgSoilRH data",
        zorder=2,
    )
    scatter = ax.scatter(
        coords[has_rh, 0],
        coords[has_rh, 1],
        c=avgsoilrh.to_numpy()[has_rh],
        cmap="viridis",
        edgecolor="white",
        label="Samples",
        zorder=3,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Average soil relative humidity (%)")
    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% variance explained)")
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% variance explained)")
    ax.set_title("PCoA of Bray-Curtis dissimilarity")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT_DIR, "fig2_pcoa_braycurtis.png"), dpi=300
    )
    plt.close(fig)

    # H3: PERMANOVA univariado por variable ambiental.
    variables = [
        ("humedad_relativa", "average-soil-relative-humidity"),
        ("temperatura", "average-soil-temperature"),
        ("elevacion", "elevation"),
    ]
    results = []
    for label, column in variables:
        values = meta_indexed.reindex(sample_ids)[column]
        valid = values.notna().to_numpy()
        sub_idx = np.where(valid)[0]
        sub_dist = dist[np.ix_(sub_idx, sub_idx)]
        sub_values = values.to_numpy()[sub_idx].astype(float)

        if len(sub_idx) < 10:
            print(
                f"AVISO: n < 10 para {label}, las permutaciones "
                "pueden ser insuficientes."
            )

        f_stat, r2, p_value = permanova(
            sub_dist, sub_values, seed=RANDOM_SEED
        )
        results.append(
            {
                "variable": label,
                "columna_original": column,
                "F": f_stat,
                "R2": r2,
                "p_valor": p_value,
                "n_permutaciones": N_PERMUTATIONS,
                "n": len(sub_idx),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        "R2", ascending=False
    ).reset_index(drop=True)

    print("\nPERMANOVA univariado (ordenado por R2 descendente):")
    print(results_df)

    results_df.to_csv(
        os.path.join(OUT_DIR, "h3_permanova_variables_ambientales.tsv"),
        sep="\t",
        index=False,
    )

    # Analisis extra: PERMANOVA multivariado combinando las tres
    # variables ambientales a la vez, para ver si el poder
    # explicativo conjunto se acerca al rango esperado por H3
    # (0.20-0.50) y si hay solapamiento (redundancia) entre ellas.
    print("\n" + "-" * 60)
    print("Analisis extra: PERMANOVA multivariado "
          "(humedad + temperatura + elevacion)")
    print("-" * 60)

    common_cols = [col for _, col in variables]
    aligned = meta_indexed.reindex(sample_ids)[common_cols]
    common_mask = aligned.notna().all(axis=1).to_numpy()
    common_pos = np.where(common_mask)[0]
    common_dist = dist[np.ix_(common_pos, common_pos)]
    common_predictors = aligned.to_numpy()[common_pos].astype(float)

    f_multi, r2_multi, p_multi = permanova_multivariate(
        common_dist, common_predictors, seed=RANDOM_SEED
    )

    # R2 individual de cada variable en ese mismo subconjunto de
    # muestras, para comparar "sola" vs "combinada" en igualdad
    # de condiciones.
    individual_rows = []
    for i, (label, column) in enumerate(variables):
        x = common_predictors[:, i]
        f_uni, r2_uni, p_uni = permanova(
            common_dist, x, seed=RANDOM_SEED
        )
        individual_rows.append(
            {"variable": label, "R2_individual": r2_uni}
        )

    sum_r2_individual = sum(
        row["R2_individual"] for row in individual_rows
    )

    multi_table = pd.DataFrame(individual_rows)
    multi_table["R2_combinado"] = r2_multi
    multi_table["F_combinado"] = f_multi
    multi_table["p_valor_combinado"] = p_multi
    multi_table["suma_R2_individuales"] = sum_r2_individual
    multi_table["n"] = len(common_pos)
    multi_table["n_permutaciones"] = N_PERMUTATIONS

    print(
        f"\nModelo combinado: F = {f_multi:.4f}, "
        f"R2 = {r2_multi:.4f}, p-valor = {p_multi:.4f}, "
        f"n = {len(common_pos)}"
    )
    print(
        f"Suma de R2 individuales (mismo subconjunto): "
        f"{sum_r2_individual:.4f}"
    )
    print(
        f"Diferencia (redundancia entre variables): "
        f"{sum_r2_individual - r2_multi:.4f}"
    )
    print(multi_table)

    multi_table.to_csv(
        os.path.join(OUT_DIR, "h3_permanova_multivariado.tsv"),
        sep="\t",
        index=False,
    )


def scatter_with_fit(x, y, xlabel, ylabel, title, out_path):
    """Scatter con regresion lineal y banda de confianza al 95%,
    anotado con R2, p-valor, rho de Spearman y n. Guarda solo en
    PNG (300 dpi), sin PDF."""
    lin = stats.linregress(x, y)
    r_squared = lin.rvalue ** 2
    rho, p_spearman = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        x, y, c="#3B4CC0", alpha=0.8, edgecolor="white", label="Samples"
    )

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = lin.intercept + lin.slope * x_line

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

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return {
        "rho": rho,
        "p_spearman": p_spearman,
        "r_squared": r_squared,
        "slope": lin.slope,
        "p_linreg": lin.pvalue,
        "n": n,
    }


def run_elevation_exploration(metadata):
    """Extra: explora si la elevacion se relaciona con la humedad
    y la temperatura del suelo (via precipitacion orografica y el
    gradiente termico altitudinal), y si eso la vuelve una mejor
    variable explicativa de la composicion que AvgSoilRH sola."""
    print("\n" + "=" * 60)
    print("Extra: elevacion vs humedad y temperatura del suelo")
    print("=" * 60)

    env = metadata[
        [
            "elevation",
            "average-soil-relative-humidity",
            "average-soil-temperature",
        ]
    ].dropna()
    x_elev = env["elevation"].to_numpy()

    stats_rh = scatter_with_fit(
        x_elev,
        env["average-soil-relative-humidity"].to_numpy(),
        "Elevation (m a.s.l.)",
        "Average soil relative humidity (%)",
        "Elevation vs. average soil relative humidity",
        os.path.join(OUT_DIR, "fig3_elevation_vs_avgsoilrh.png"),
    )
    stats_temp = scatter_with_fit(
        x_elev,
        env["average-soil-temperature"].to_numpy(),
        "Elevation (m a.s.l.)",
        "Average soil temperature (C)",
        "Elevation vs. average soil temperature",
        os.path.join(OUT_DIR, "fig3_elevation_vs_temperature.png"),
    )

    print(
        f"\nElevacion vs AvgSoilRH: rho = {stats_rh['rho']:.4f}, "
        f"p = {stats_rh['p_spearman']:.6f}, n = {stats_rh['n']}"
    )
    print(
        f"Elevacion vs temperatura: rho = {stats_temp['rho']:.4f}, "
        f"p = {stats_temp['p_spearman']:.6f}, n = {stats_temp['n']}"
    )

    table = pd.DataFrame(
        [
            {
                "variable_x": "elevation",
                "variable_y": "average-soil-relative-humidity",
                "spearman_rho": stats_rh["rho"],
                "p_valor": stats_rh["p_spearman"],
                "n": stats_rh["n"],
            },
            {
                "variable_x": "elevation",
                "variable_y": "average-soil-temperature",
                "spearman_rho": stats_temp["rho"],
                "p_valor": stats_temp["p_spearman"],
                "n": stats_temp["n"],
            },
        ]
    )
    table.to_csv(
        os.path.join(OUT_DIR, "h3b_correlacion_elevacion_variables.tsv"),
        sep="\t",
        index=False,
    )


def main():
    abund, metadata = load_data()
    run_h1(abund, metadata)
    run_h2_h3(abund, metadata)
    run_elevation_exploration(metadata)


if __name__ == "__main__":
    main()
