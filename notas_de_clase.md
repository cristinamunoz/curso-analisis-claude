# Study notes — Atacama Desert soil microbiome

Personal study notebook for the Neilson et al. (2017)
reproduction. For each hypothesis: what the analysis is, why we
use it, what the numbers mean, and what my results say
biologically. Written as class notes to revisit later, not as a
lab report.

---

## Quick reference: which analysis for which question?

| Biological question | Analysis | Key statistic | What it tells you |
|---|---|---|---|
| Does diversity change along a gradient? | Shannon index + Spearman correlation | rho, R², p-value | Whether *how many* species and *how evenly* they're distributed shifts with an environmental variable |
| Do communities differ in *composition*, not just diversity? | Beta diversity: Bray-Curtis + PCoA | % variance per axis | Whether *which* taxa are present (and in what proportion) differs between samples |
| Which environmental variable best explains composition? | PERMANOVA (univariate or multivariate) | F, R², p-value | How much of the total compositional variation is attributable to a variable (or set of variables) |

Rule of thumb: alpha diversity answers "how much life is here?";
beta diversity answers "is it the *same kind* of life?".

---

## H1 — Alpha diversity vs. soil humidity

**Question:** does lower soil humidity (AvgSoilRH) mean lower
microbial diversity?

**Why Shannon index:** it doesn't just count how many OTUs
(species-level units) are present (that's richness) — it also
weighs how evenly abundance is distributed among them. A sample
dominated by one OTU gets a *lower* Shannon value than one with
the same number of OTUs spread evenly. That's important in
deserts, where a handful of stress-tolerant taxa (e.g.
Actinobacteria) can dominate hyperarid soils.

**Why Spearman, not Pearson:** Spearman's rho tests whether two
variables move together in *rank* order, without assuming a
straight-line relationship or normally-distributed data. Microbial
abundance data are rarely normal, so this is the safer default —
it's also literally the method the original paper used, which
makes our numbers directly comparable.

**What I did:** computed Shannon (natural log) per sample from raw
OTU counts, then correlated it against AvgSoilRH.

**Results (n=51):**
- Spearman rho = 0.469, p = 0.0005
- Linear regression (just for the plot's trend line): R² = 0.212

**What this means biologically:** samples with more soil moisture
tend to host more diverse microbial communities. The relationship
is real (very low p-value) but far from perfect (R² ≈ 0.21 means
humidity accounts for about a fifth of the variation — most of the
diversity differences between samples come from something else:
temperature, salinity, nutrients, or just local chance). This
matches the paper's own finding that aridity effects on diversity
are real but moderate, not deterministic.

**Report:** `outputs/h1_report.html`

---

## H2 — Community composition (beta diversity)

**Question:** do the most arid sites (Yungay) separate from the
least arid sites (Baquedano) in how their communities are
*composed* — not just how diverse they are?

**Why Bray-Curtis:** it's a dissimilarity measure built for
abundance data — two samples get a low Bray-Curtis distance if
they share the same OTUs in similar proportions, and a high
distance if their dominant taxa differ. It doesn't care about
evolutionary relationships between OTUs (unlike UniFrac, which the
original paper used and which needs a phylogenetic tree we don't
have in this course dataset).

**Why PCoA:** Principal Coordinates Analysis takes an n × n
distance matrix (54 samples × 54 samples here) and projects it
into a small number of axes (PC1, PC2, ...) that capture as much
of the between-sample variation as possible, so it can be plotted
in 2D. Each axis's "% variance explained" tells you how much of
the total compositional difference between all samples that axis
alone accounts for.

**What I did:** built the 54×54 Bray-Curtis distance matrix from
raw OTU counts, ran classical (metric) PCoA via double-centering
and eigendecomposition, and colored each point by its AvgSoilRH
value on a continuous (viridis) scale.

**Results:**
- PC1 = 8.31% variance explained, PC2 = 5.74%
- Humid samples (yellow) cluster toward negative PC1; arid
  samples (purple) spread toward positive PC1

**What this means biologically:** there is a visible gradient in
community composition that lines up with humidity — supporting
H2 — but it's a gradient, not two clean separate clusters, and it
only accounts for a small slice of total variance. That's normal
for OTU-level Bray-Curtis on a sparse table (1109 OTUs, most
present in only a few samples): a lot of the variance gets spread
thin across many minor axes instead of concentrating in PC1/PC2.

**Report:** `outputs/h2_h3_report.html`

---

## H3 — Modeling composition against environmental variables

**Question:** which environmental variable — humidity,
temperature, or elevation — explains the most compositional
variance? (Course expectation: humidity should win, with
R² between 0.20 and 0.50.)

**Why PERMANOVA:** PCoA is descriptive (it shows a pattern);
PERMANOVA is the test that asks "is this pattern statistically
associated with variable X, and how much variance does X
explain?" It works directly on the distance matrix (no need to
assume normality) by comparing the real association between the
variable and the distances to what you'd see if you randomly
shuffled the variable among samples many times (999 permutations
here). If the real F-statistic beats almost all shuffled versions,
the relationship is unlikely to be due to chance.

- **F** — signal-to-noise ratio: variance explained by the
  variable, divided by leftover (residual) variance. Higher F =
  stronger relationship.
- **R²** — the fraction of total compositional variance
  attributable to that variable (0 to 1, like in a regression).
- **p-value** — how often a random shuffle beat the real result.
  With 999 permutations, the smallest possible p-value is
  1/1000 = 0.001.

**What I did (univariate):** ran PERMANOVA separately for
humidity, temperature, and elevation, each on its own maximum
available sample subset (elevation is known for all 54 samples;
humidity and temperature are missing for the 3 BAQ4697 samples
whose logger was lost).

**Results (univariate, sorted by R²):**

| Variable | F | R² | p | n |
|---|---|---|---|---|
| Temperature | 3.00 | 0.058 | 0.001 | 51 |
| Elevation | 2.93 | 0.053 | 0.001 | 54 |
| Humidity | 2.57 | 0.050 | 0.001 | 51 |

All three are statistically significant, but **none reach the
expected 0.20–0.50 range**, and humidity — the variable H3 bets
on — actually explains the *least* variance of the three.

**Exploring alternatives (multivariate PERMANOVA):** the paper's
strongest result didn't come from testing one variable at a
time — it came from combining five variables together (their BEST
analysis reached rs = 0.776). So I tested humidity + temperature +
elevation *together* in one model:

- Combined R² = 0.129 (F = 2.31, p = 0.001, n = 51)
- Sum of the three individual R² values = 0.158
- Redundancy (overlap between variables) = only 0.029

**What this means biologically and methodologically:** combined,
the three variables explain **~2.5× more variance** than any one
alone, with very little redundancy — meaning humidity, temperature,
and elevation are mostly telling us *different* things about the
community, not repeating the same signal. This still falls short
of the paper's 0.20–0.50 range, most likely because:

1. **Distance metric**: the paper used UniFrac (phylogeny-aware),
   which down-weights noise from many closely-related rare OTUs.
   Bray-Curtis treats every OTU as independent, so sparse,
   rare-taxon-heavy tables like this one (1109 OTUs) get noisier.
2. **Univariate vs. combined power**: testing one variable at a
   time — as H3 is worded — will always underestimate the total
   environmental signal when multiple correlated variables act
   together, which is exactly what we saw here.

So: H3 as literally stated (humidity alone, R² 0.20–0.50) doesn't
hold with this data and method, but the underlying idea — that
environment shapes composition substantially — holds up much
better once you combine variables. Good reminder that a null (or
weak) univariate result doesn't always mean "no effect"; it can
mean "not the whole story."

**Report:** `outputs/h2_h3_report.html`

---

## Things to keep in mind for next time

- Always check whether every sample has data for the variable
  you're testing — subsets differ (51 vs. 54 here) and matter for
  n-dependent statistics.
- Spearman/PERMANOVA R² and p-values from different distance
  metrics or test types (Mantel r vs. PERMANOVA R²) are not
  directly comparable numbers, even when they're both called
  "R-squared" or "correlation" — check what each one actually
  measures before comparing to a paper's reported value.
- A weak univariate result is worth re-testing in combination with
  other plausible variables before concluding "no effect."
