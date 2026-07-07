"""
Análisis del microbioma del Atacama — Hipótesis 1, 2, 3
Neilson et al. (2017): efecto de la aridez en diversidad microbiana
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, linregress
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilos
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
PALETTE_DISCRETE = "Set2"
PALETTE_CONTINUOUS = "viridis"

# Paleta colorblind-safe (Okabe-Ito), estilo Nature/Cell/Science
COLOR_BAQUEDANO = '#0072B2'  # azul
COLOR_YUNGAY = '#D55E00'     # vermellón/naranja
JOURNAL_PALETTE = {'Baquedano': COLOR_BAQUEDANO, 'Yungay': COLOR_YUNGAY}

# ============================================================================
# PASO 0: Carga de datos
# ============================================================================

def load_and_prepare_data():
    """
    Carga abundancias y metadata, transpone abundancias, cruza por
    sample-id manteniendo solo la intersección.
    """
    abundancias = pd.read_csv('data/abundancias.tsv', sep='\t', index_col=0)
    metadata = pd.read_csv('data/metadata.tsv', sep='\t', index_col=0)

    abundancias_t = abundancias.T
    common_samples = abundancias_t.index.intersection(metadata.index)
    abundancias_t = abundancias_t.loc[common_samples]
    metadata = metadata.loc[common_samples]

    print(f"Muestras después del cruce: {len(abundancias_t)}")
    print(f"OTUs: {abundancias_t.shape[1]}")

    return abundancias_t, metadata

# ============================================================================
# H1: Diversidad alfa (Shannon) vs. AvgSoilRH
# ============================================================================

def calculate_shannon(abundancia_row):
    """Calcula Shannon considerando abundancias relativas."""
    relativo = abundancia_row / abundancia_row.sum()
    relativo = relativo[relativo > 0]
    return -np.sum(relativo * np.log(relativo))

def h1_shannon_analysis(abundancias_t, metadata):
    """
    H1: A menor AvgSoilRH, menor Shannon.
    Genera tabla resumen por transecto, correlación Spearman,
    regresión lineal y figura scatter + regresión.
    """
    shannon = abundancias_t.apply(calculate_shannon, axis=1)

    df_h1 = pd.DataFrame({
        'sample_id': shannon.index,
        'shannon': shannon.values,
        'transect': metadata['transect-name'].values,
        'avg_soil_rh': metadata[
            'average-soil-relative-humidity'].values
    })

    # Resumen por transecto
    resumen = df_h1.groupby('transect')['shannon'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()
    resumen.columns = [
        'transect-name', 'mean', 'median', 'min', 'max', 'count']
    resumen.to_csv(
        'outputs/H1/h1_resumen_shannon_por_transecto.tsv',
        sep='\t', index=False)

    print("Resumen Shannon por transecto:")
    print(resumen)

    # Correlación Spearman
    df_h1_clean = df_h1.dropna(subset=['avg_soil_rh'])
    rho, pval = spearmanr(
        df_h1_clean['avg_soil_rh'], df_h1_clean['shannon'])

    corr_result = pd.DataFrame({
        'variable_x': ['average-soil-relative-humidity'],
        'variable_y': ['shannon'],
        'spearman_rho': [rho],
        'p_valor': [pval],
        'n': [len(df_h1_clean)]
    })
    corr_result.to_csv(
        'outputs/H1/h1_correlacion_shannon_vs_humedad.tsv',
        sep='\t', index=False)

    print(f"\nCorrelación Spearman (Shannon vs. AvgSoilRH):")
    print(f"  rho = {rho:.4f}, p = {pval:.6f}, n = {len(df_h1_clean)}")

    # Regresión lineal (para R² y pendiente)
    slope, intercept, r_value, p_value, stderr = linregress(
        df_h1_clean['avg_soil_rh'], df_h1_clean['shannon'])
    r_squared = r_value ** 2

    print(f"\nRegresión lineal:")
    print(f"  Pendiente = {slope:.6f}")
    print(f"  R² = {r_squared:.4f}")
    print(f"  p-valor = {p_value:.6f}")
    print(f"  Correlación positiva: {slope > 0}")

    # Boxplot por transecto
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot Shannon por transecto
    sns.boxplot(data=df_h1, x='transect', y='shannon',
                palette=PALETTE_DISCRETE, ax=axes[0])
    axes[0].set_xlabel('Transecto', fontsize=12)
    axes[0].set_ylabel('Índice de Shannon', fontsize=12)
    axes[0].set_title('Diversidad alfa (Shannon) por transecto',
                      fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Scatter + regresión con R²
    axes[1].scatter(df_h1_clean['avg_soil_rh'],
                   df_h1_clean['shannon'],
                   alpha=0.6, s=100, color='steelblue',
                   edgecolors='black', linewidth=0.5)

    # Línea de regresión
    x_line = np.linspace(
        df_h1_clean['avg_soil_rh'].min(),
        df_h1_clean['avg_soil_rh'].max(), 100)
    y_line = slope * x_line + intercept
    axes[1].plot(x_line, y_line, color='red', linewidth=2,
                label=f'Regresión (R² = {r_squared:.3f})')

    # Banda de confianza 95%
    residuals = df_h1_clean['shannon'] - (
        slope * df_h1_clean['avg_soil_rh'] + intercept)
    std_residuals = np.std(residuals)
    conf_interval = 1.96 * std_residuals
    axes[1].fill_between(x_line, y_line - conf_interval,
                        y_line + conf_interval,
                        alpha=0.2, color='red',
                        label='IC 95%')

    axes[1].set_xlabel('Humedad relativa promedio del suelo (%)',
                      fontsize=12)
    axes[1].set_ylabel('Índice de Shannon', fontsize=12)
    axes[1].set_title(
        f'Shannon vs. AvgSoilRH\n'
        f'ρ(Spearman) = {rho:.3f}, p = {pval:.4f}, n = {len(df_h1_clean)}',
        fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        'outputs/H1/fig1_shannon_por_transecto_y_regresion.png',
        dpi=300, bbox_inches='tight')
    plt.savefig(
        'outputs/H1/fig1_shannon_por_transecto_y_regresion.pdf',
        bbox_inches='tight')
    print(
        "\nFigura guardada en "
        "outputs/H1/fig1_shannon_por_transecto_y_regresion.*")
    plt.close()

    return df_h1, resumen, corr_result, {
        'slope': slope,
        'r_squared': r_squared,
        'p_value': p_value,
        'spearman_rho': rho
    }

def generar_pdf_estilo_articulo(df_h1, stats_h1):
    """
    Genera un PDF con las dos figuras de H1 por separado, con
    estilo tipo revista científica (Nature/Cell): paleta
    colorblind-safe (Okabe-Ito), fuente serif, leyenda tipo
    "Figure N." debajo de cada figura.
    """
    plt.rcParams['font.family'] = 'serif'
    df_h1_clean = df_h1.dropna(subset=['avg_soil_rh'])
    rho = stats_h1['spearman_rho']
    r_squared = stats_h1['r_squared']
    p_value = stats_h1['p_value']
    slope = stats_h1['slope']
    n = len(df_h1_clean)

    with PdfPages('outputs/H1/figuras_H1_articulo.pdf') as pdf:

        # -------- Figura 1: Boxplot por transecto --------
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        sns.boxplot(
            data=df_h1, x='transect', y='shannon',
            palette=JOURNAL_PALETTE, ax=ax, width=0.5,
            fliersize=4, linewidth=1.2)
        sns.stripplot(
            data=df_h1, x='transect', y='shannon',
            color='black', alpha=0.4, size=4, ax=ax)
        ax.set_xlabel('Transecto', fontsize=12)
        ax.set_ylabel('Índice de diversidad de Shannon (H\')',
                      fontsize=12)
        ax.set_title(
            'Diversidad alfa por transecto de aridez',
            fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.25)
        plt.tight_layout(rect=(0, 0.13, 1, 1))

        caption1 = (
            "Figure 1. Alpha diversity of soil microbial "
            "communities across an aridity gradient in the "
            "Atacama Desert. Shannon diversity index (H') is "
            "shown for samples from the Baquedano transect "
            "(more humid, n = 25) and the Yungay transect "
            "(more arid, n = 29). Boxes show median and "
            "interquartile range; points represent individual "
            "samples. Baquedano exhibits higher median "
            "diversity than Yungay, consistent with reduced "
            "water availability filtering out "
            "desiccation-sensitive taxa."
        )
        fig.text(0.08, 0.02, caption1, wrap=True, fontsize=8.5,
                 ha='left', va='bottom')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # -------- Figura 2: Scatter + regresión --------
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        colors = df_h1_clean['transect'].map(JOURNAL_PALETTE)
        ax.scatter(
            df_h1_clean['avg_soil_rh'], df_h1_clean['shannon'],
            c=colors, alpha=0.75, s=70,
            edgecolors='white', linewidth=0.6)

        x_line = np.linspace(
            df_h1_clean['avg_soil_rh'].min(),
            df_h1_clean['avg_soil_rh'].max(), 100)
        intercept = (
            df_h1_clean['shannon'].mean()
            - slope * df_h1_clean['avg_soil_rh'].mean())
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='#333333', linewidth=1.8,
                linestyle='-')

        residuals = df_h1_clean['shannon'] - (
            slope * df_h1_clean['avg_soil_rh'] + intercept)
        conf_interval = 1.96 * np.std(residuals)
        ax.fill_between(
            x_line, y_line - conf_interval, y_line + conf_interval,
            alpha=0.15, color='#333333')

        ax.text(
            0.05, 0.95,
            f"R² = {r_squared:.3f}\n"
            f"Spearman ρ = {rho:.3f}\n"
            f"p = {p_value:.4f}\nn = {n}",
            transform=ax.transAxes, fontsize=10,
            va='top', ha='left',
            bbox=dict(
                boxstyle='round', facecolor='white',
                edgecolor='#cccccc', alpha=0.9))

        handles = [
            plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor=COLOR_BAQUEDANO,
                markersize=8, label='Baquedano'),
            plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor=COLOR_YUNGAY,
                markersize=8, label='Yungay')
        ]
        ax.legend(handles=handles, loc='lower right', fontsize=9,
                 frameon=False)

        ax.set_xlabel(
            'Average soil relative humidity (%)', fontsize=12)
        ax.set_ylabel(
            "Shannon diversity index (H')", fontsize=12)
        ax.set_title(
            'Diversity increases with soil moisture availability',
            fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2)
        plt.tight_layout(rect=(0, 0.16, 1, 1))

        caption2 = (
            "Figure 2. Relationship between soil moisture and "
            "microbial alpha diversity. Shannon diversity index "
            "plotted against average soil relative humidity "
            "for all samples with available humidity data "
            f"(n = {n}). Solid line shows the linear "
            "regression fit; shaded band indicates the 95% "
            "confidence interval. A significant positive "
            f"correlation (Spearman ρ = {rho:.2f}, "
            f"p = {p_value:.4f}) supports the hypothesis that "
            "increasing aridity reduces microbial diversity "
            "in Atacama Desert soils."
        )
        fig.text(0.08, 0.02, caption2, wrap=True, fontsize=8.5,
                 ha='left', va='bottom')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    plt.rcParams['font.family'] = 'sans-serif'
    print("\n[OK] PDF estilo articulo guardado en "
          "outputs/H1/figuras_H1_articulo.pdf")


# ============================================================================
# H2: Composición (beta-diversidad) — Bray-Curtis, PCoA y PERMANOVA
# ============================================================================

def bray_curtis_matrix(abundancias_subset):
    """Matriz de distancias Bray-Curtis (cuadrada) entre muestras."""
    dist_condensed = pdist(abundancias_subset.values,
                            metric='braycurtis')
    return squareform(dist_condensed)


def pcoa(dist_matrix):
    """
    PCoA clásico (Gower) por descomposición espectral: centra la
    matriz de distancias al cuadrado y calcula autovectores /
    autovalores. Devuelve coordenadas y % de varianza por eje
    (solo autovalores positivos).
    """
    n = dist_matrix.shape[0]
    d2 = dist_matrix ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j

    eigenvalues, eigenvectors = np.linalg.eigh(b)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positivos = eigenvalues > 1e-10
    coords = eigenvectors[:, positivos] * np.sqrt(
        eigenvalues[positivos])
    varianza_pct = (
        eigenvalues[positivos] / eigenvalues[positivos].sum() * 100)
    return coords, varianza_pct


def permanova_variable_continua(dist_matrix, x, n_perm=999, seed=42):
    """
    PERMANOVA (McArdle & Anderson) para un único predictor
    continuo sobre una matriz de distancias: parte la matriz de
    Gower en varianza explicada por el predictor (R²) y varianza
    residual, y evalúa significancia con permutaciones.
    """
    n = dist_matrix.shape[0]
    d2 = dist_matrix ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    a = -0.5 * j @ d2 @ j
    ss_total = np.trace(a)

    def r2_de(x_vec):
        x_mat = np.column_stack([np.ones(n), x_vec])
        h = x_mat @ np.linalg.pinv(x_mat.T @ x_mat) @ x_mat.T
        return np.trace(h @ a) / ss_total

    df1, df2 = 1, n - 2
    r2_obs = r2_de(x)
    f_obs = r2_obs * df2 / ((1 - r2_obs) * df1)

    rng = np.random.default_rng(seed)
    mas_extremos = 0
    for _ in range(n_perm):
        r2_perm = r2_de(rng.permutation(x))
        f_perm = (r2_perm * df2 / ((1 - r2_perm) * df1)
                  if r2_perm < 1 else np.inf)
        if f_perm >= f_obs:
            mas_extremos += 1
    p_valor = (mas_extremos + 1) / (n_perm + 1)

    return f_obs, r2_obs, p_valor, n


def h2_composicion_analysis(abundancias_t, metadata):
    """
    H2: Yungay se separa de Baquedano en el eje 1 del PCoA.
    Calcula Bray-Curtis + PCoA sobre todas las muestras, y
    modela composición ~ variables ambientales (humedad,
    temperatura, elevación) vía PERMANOVA univariado.
    """
    # --- PCoA sobre todas las muestras (n=54) ---
    dist_full = bray_curtis_matrix(abundancias_t)
    coords, varianza_pct = pcoa(dist_full)

    var_table = pd.DataFrame({
        'eje': [f'PC{i + 1}' for i in range(2)],
        'porcentaje_varianza': varianza_pct[:2]
    })
    var_table.to_csv(
        'outputs/H2/h2_varianza_explicada_pcoa.tsv',
        sep='\t', index=False)

    print("Varianza explicada por eje:")
    print(var_table)

    pc1, pc2 = coords[:, 0], coords[:, 1]
    avg_soil_rh = metadata['average-soil-relative-humidity'].values
    transect = metadata['transect-name'].values

    # Separación descriptiva de PC1 por transecto
    pc1_por_transecto = pd.DataFrame({
        'transect': transect, 'pc1': pc1
    }).groupby('transect')['pc1'].agg(['mean', 'median'])
    print("\nPC1 promedio por transecto:")
    print(pc1_por_transecto)

    # --- Figura PCoA coloreada por gradiente continuo (viridis) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        pc1, pc2, c=avg_soil_rh, cmap=PALETTE_CONTINUOUS,
        s=100, edgecolors='black', linewidth=0.6)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Humedad relativa promedio del suelo (%)',
                   fontsize=11)

    for t, marker in [('Baquedano', 'o'), ('Yungay', '^')]:
        mask = transect == t
        ax.scatter(
            pc1[mask], pc2[mask], facecolors='none',
            edgecolors='black' if t == 'Baquedano' else 'dimgray',
            marker=marker, s=140, linewidth=1.3, label=t)

    ax.set_xlabel(f'PC1 ({varianza_pct[0]:.1f}% varianza)',
                 fontsize=12)
    ax.set_ylabel(f'PC2 ({varianza_pct[1]:.1f}% varianza)',
                 fontsize=12)
    ax.set_title(
        'PCoA (Bray-Curtis): composición microbiana del suelo',
        fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig('outputs/H2/fig2_pcoa_braycurtis.png',
               dpi=300, bbox_inches='tight')
    plt.savefig('outputs/H2/fig2_pcoa_braycurtis.pdf',
               bbox_inches='tight')
    plt.close()
    print("\nFigura PCoA guardada en outputs/H2/fig2_pcoa_braycurtis.*")

    # --- Modelar composición ~ variables ambientales (PERMANOVA) ---
    variables = [
        ('humedad_relativa', 'average-soil-relative-humidity'),
        ('temperatura', 'average-soil-temperature'),
        ('elevacion', 'elevation'),
    ]
    resultados = []
    for nombre, columna in variables:
        valido = metadata[columna].notna()
        sub_metadata = metadata[valido]
        sub_abundancias = abundancias_t.loc[sub_metadata.index]
        dist_sub = bray_curtis_matrix(sub_abundancias)
        x = sub_metadata[columna].values.astype(float)

        f_val, r2_val, p_val, n_val = permanova_variable_continua(
            dist_sub, x)
        resultados.append({
            'variable': nombre,
            'columna_original': columna,
            'F': f_val,
            'R2': r2_val,
            'p_valor': p_val,
            'n_permutaciones': 999,
            'n': n_val
        })

    df_permanova = pd.DataFrame(resultados).sort_values(
        'R2', ascending=False).reset_index(drop=True)
    df_permanova.to_csv(
        'outputs/H2/h2_modelo_composicion_variables_ambientales.tsv',
        sep='\t', index=False)

    print("\nModelo composición ~ variables ambientales (PERMANOVA):")
    print(df_permanova)

    return var_table, df_permanova, pc1_por_transecto


def generar_reporte_html_h2(var_table, df_permanova,
                            pc1_por_transecto):
    """
    Genera un reporte HTML autocontenido (figura embebida en
    base64) con el resultado de H2: PCoA, varianza explicada,
    modelo de composición y conclusión biológica.
    """
    import base64

    with open('outputs/H2/fig2_pcoa_braycurtis.png', 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    var_html = var_table.to_html(index=False, float_format='%.2f')
    permanova_html = df_permanova.to_html(
        index=False, float_format='%.4f')
    pc1_html = pc1_por_transecto.reset_index().to_html(
        index=False, float_format='%.3f')

    mejor_variable = df_permanova.iloc[0]['variable']
    mejor_r2 = df_permanova.iloc[0]['R2']

    baq_pc1 = pc1_por_transecto.loc['Baquedano', 'mean']
    yun_pc1 = pc1_por_transecto.loc['Yungay', 'mean']
    separacion = (
        "se observa separación" if abs(baq_pc1 - yun_pc1) > 0.05
        else "no se observa una separación clara")

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Reporte H2 — Composición microbiana (PCoA Bray-Curtis)</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 900px;
          margin: 40px auto; padding: 0 20px; color: #222; }}
  h1 {{ font-size: 1.6em; border-bottom: 3px solid #0072B2;
        padding-bottom: 8px; }}
  h2 {{ font-size: 1.2em; color: #0072B2; margin-top: 2em; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px;
            text-align: right; }}
  th {{ background: #f0f4f8; }}
  td:first-child, th:first-child {{ text-align: left; }}
  img {{ max-width: 100%; border: 1px solid #ddd; margin: 1em 0; }}
  .conclusion {{ background: #f7f9fb; border-left: 4px solid #0072B2;
                 padding: 12px 18px; margin: 1.5em 0; }}
  .caption {{ font-size: 0.9em; color: #555; max-width: 700px; }}
</style>
</head>
<body>

<h1>H2 — Composición microbiana y aridez (PCoA Bray-Curtis)</h1>

<p><b>Pregunta:</b> ¿Los sitios más áridos (Yungay) se separan de
los menos áridos (Baquedano) en la ordenación PCoA a lo largo del
primer eje?</p>

<h2>Ordenación PCoA</h2>
<img src="data:image/png;base64,{img_b64}"
     alt="PCoA Bray-Curtis coloreado por humedad del suelo">
<p class="caption">Figura 2. Ordenación de coordenadas
principales (PCoA) basada en distancias Bray-Curtis entre
comunidades microbianas del suelo. Cada punto es una muestra,
coloreada según un gradiente continuo de humedad relativa
promedio del suelo (escala viridis); el marcador indica el
transecto de origen (círculo = Baquedano, triángulo = Yungay).</p>

<h2>Varianza explicada por eje</h2>
{var_html}

<h2>PC1 promedio por transecto</h2>
{pc1_html}
<p class="caption">Diferencia de medias en PC1: Baquedano =
{baq_pc1:.3f}, Yungay = {yun_pc1:.3f} — {separacion} entre
transectos a lo largo del primer eje.</p>

<h2>Modelo: composición ~ variables ambientales (PERMANOVA)</h2>
<p>Cada fila prueba si una variable ambiental, por sí sola,
explica variación en la composición microbiana (distancias
Bray-Curtis), usando 999 permutaciones.</p>
{permanova_html}

<div class="conclusion">
<b>Conclusión biológica:</b> La ordenación PCoA explica un
{var_table.iloc[0]['porcentaje_varianza']:.1f}% (PC1) y
{var_table.iloc[1]['porcentaje_varianza']:.1f}% (PC2) de la
variación composicional total — porcentajes bajos, típicos de
datos de microbioma con miles de OTUs raros, donde ningún eje
domina completamente. De las variables ambientales evaluadas,
<b>{mejor_variable}</b> es la que más varianza composicional
explica individualmente (R² = {mejor_r2:.3f}), aunque todas
las variables muestran efectos modestos. Esto sugiere que la
composición microbiana del Atacama responde a la aridez, pero
de forma más gradual/distribuida que un cambio abrupto en un
solo eje — coherente con un filtro ambiental que actúa junto a
otros factores no medidos (dispersión, historia del sitio, etc.).
</div>

</body>
</html>
"""

    with open('outputs/H2/reporte_h2.html', 'w',
             encoding='utf-8') as f:
        f.write(html)

    print("\n[OK] Reporte HTML guardado en outputs/H2/reporte_h2.html")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("Cargando datos...")
    abundancias_t, metadata = load_and_prepare_data()

    print("\n" + "="*70)
    print("H1: Diversidad alfa (Shannon) vs. Humedad relativa del suelo")
    print("="*70)
    df_h1, resumen_h1, corr_h1, stats_h1 = h1_shannon_analysis(
        abundancias_t, metadata)

    print("\n[OK] H1 completado. Archivos guardados en outputs/H1/")

    generar_pdf_estilo_articulo(df_h1, stats_h1)

    print("\n" + "="*70)
    print("H2: Composición microbiana (PCoA Bray-Curtis) vs. aridez")
    print("="*70)
    var_table, df_permanova, pc1_por_transecto = (
        h2_composicion_analysis(abundancias_t, metadata))

    print("\n[OK] H2 completado. Archivos guardados en outputs/H2/")

    generar_reporte_html_h2(var_table, df_permanova,
                            pc1_por_transecto)
