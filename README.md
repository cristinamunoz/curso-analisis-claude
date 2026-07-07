# curso-analisis-claude

Ecología de Comunidades con Claude Code — curso de 2 mañanas donde
biólogos y ecólogos sin experiencia previa en programación usan
Claude Code para reproducir el análisis de
[Neilson et al. (2017)](paper/mSystems.00195-16.pdf) sobre el
efecto de la aridez en el microbioma de suelo del desierto de
Atacama.

Ver [`CLAUDE.md`](CLAUDE.md) para el contexto e instrucciones que
sigue el agente en este repositorio.

## Estructura del repositorio

```
curso-analisis-claude/
├── CLAUDE.md
├── README.md
├── docs/
│   ├── programa_curso_vJulio2026.pdf
│   ├── pasos_curso_manana1.md
│   └── pasos_curso_manana2.md
├── paper/
│   └── mSystems.00195-16.pdf
├── data/
│   ├── abundancias.tsv
│   └── metadata.tsv
└── expected_outputs/
    ├── fig1_shannon_por_transecto.png / .pdf
    ├── fig1_shannon_vs_avgsoilrh.png / .pdf
    ├── fig2_pcoa_braycurtis.png / .pdf
    ├── h1_correlacion_shannon_vs_humedad.tsv
    ├── h1_resumen_shannon_por_transecto.tsv
    ├── h2_varianza_explicada_pcoa.tsv
    └── h3_permanova_variables_ambientales.tsv
```

### [`docs/`](docs/)

- [`programa_curso_vJulio2026.pdf`](docs/programa_curso_vJulio2026.pdf) —
  programa oficial del curso: horario y contenidos por módulo.
- [`pasos_curso_manana1.md`](docs/pasos_curso_manana1.md) —
  checklist orientada a participantes para la Mañana 1 (aprender a
  trabajar con Claude Code). Por ahora es un esqueleto.
- [`pasos_curso_manana2.md`](docs/pasos_curso_manana2.md) —
  checklist orientada a participantes para la Mañana 2 (análisis
  real del microbioma de Atacama), con pasos detallados módulo por
  módulo.

### [`paper/`](paper/)

- [`mSystems.00195-16.pdf`](paper/mSystems.00195-16.pdf) — Neilson,
  J.W. et al. (2017). *Significant Impacts of Increasing Aridity on
  the Arid Soil Microbiome*. mSystems, 2(3): e00195-16. Paper base
  del curso; ver cita completa en [`CLAUDE.md`](CLAUDE.md).

### [`data/`](data/)

Datos del paper de Neilson et al. (2017), usados como insumo para
el análisis del curso:

- [`abundancias.tsv`](data/abundancias.tsv) — tabla de abundancias
  de OTUs (una fila por OTU, una columna por muestra de suelo),
  exportada desde el archivo `.biom` original del estudio. 1 109
  OTUs × 54 muestras.

  ```
  #OTU ID    BAQ2420.1.1  BAQ2420.1.2  BAQ2420.1.3  BAQ2420.2  BAQ2420.3  ...
  409faa5f5353e543bf6d99125c7c0e83  0.0  0.0  0.0    0.0    0.0  ...
  1237d5925a7176fced9dda961a86c684  0.0  0.0  13.0   103.0  0.0  ...
  ```

- [`metadata.tsv`](data/metadata.tsv) — metadata de cada muestra:
  transecto, sitio, variables ambientales (pH, humedad relativa del
  suelo, temperatura, elevación, vegetación, etc.) necesarias para
  correlacionar con la diversidad/composición microbiana. 75
  muestras × 21 columnas.

  ```
  sample-id    barcode-sequence  elevation  ...  transect-name  site-name  ...
  BAQ1370.1.2  GCCCAAGTTCAC      1370       ...  Baquedano      BAQ1370    ...
  BAQ1370.3    GCGCCGAATCTT      1370       ...  Baquedano      BAQ1370    ...
  ```

  Solo 54 de las 75 muestras de `metadata.tsv` tienen secuenciación
  en `abundancias.tsv` (las 21 restantes quedaron sin datos de
  abundancia) — al cruzar ambos archivos hay que filtrar por la
  intersección de IDs, no asumir que coinciden 1 a 1.

### [`expected_outputs/`](expected_outputs/)

Figuras y tablas de referencia contra las cuales cada participante
compara sus propios resultados. Cubren las tres hipótesis del
curso (ver [`CLAUDE.md`](CLAUDE.md)):

- **H1 — diversidad alfa vs. humedad relativa del suelo**:
  [`fig1_shannon_por_transecto.png`](expected_outputs/fig1_shannon_por_transecto.png)
  (boxplot Shannon por transecto),
  [`fig1_shannon_vs_avgsoilrh.png`](expected_outputs/fig1_shannon_vs_avgsoilrh.png)
  (dispersión con regresión),
  [`h1_resumen_shannon_por_transecto.tsv`](expected_outputs/h1_resumen_shannon_por_transecto.tsv)
  y
  [`h1_correlacion_shannon_vs_humedad.tsv`](expected_outputs/h1_correlacion_shannon_vs_humedad.tsv)
  (Spearman ρ, p-valor, n).
- **H2 — ordenación de comunidades (PCoA Bray-Curtis)**:
  [`fig2_pcoa_braycurtis.png`](expected_outputs/fig2_pcoa_braycurtis.png)
  (PC1 vs. PC2, color = humedad relativa del suelo) y
  [`h2_varianza_explicada_pcoa.tsv`](expected_outputs/h2_varianza_explicada_pcoa.tsv)
  (% de varianza por eje).
- **H3 — variables ambientales que explican la composición**:
  [`h3_permanova_variables_ambientales.tsv`](expected_outputs/h3_permanova_variables_ambientales.tsv)
  (PERMANOVA univariada de humedad, temperatura y elevación —
  F, R², p-valor, 999 permutaciones).

Cada PNG tiene su versión vectorial en PDF. Generados con
`generate_expected_outputs.py` (entorno en `requirements.txt`).

## Branches personales

Cada participante trabaja en su propio branch, con su propio
`CLAUDE.md` personalizado, `solution.py` y carpeta `outputs/`.

---

Este README se mantiene actualizado a medida que se agregan
archivos o carpetas al repositorio.
