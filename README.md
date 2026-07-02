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
│   ├── abundancias.csv
│   └── metadata.tsv
└── expected_outputs/
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

- [`abundancias.csv`](data/abundancias.csv) — tabla de abundancias
  de OTUs (una fila por OTU, una columna por muestra de suelo),
  exportada desde el archivo `.biom` original del estudio.
- [`metadata.tsv`](data/metadata.tsv) — metadata de cada muestra:
  transecto, sitio, variables ambientales (pH, humedad relativa del
  suelo, temperatura, elevación, vegetación, etc.) necesarias para
  correlacionar con la diversidad/composición microbiana.

### [`expected_outputs/`](expected_outputs/)

Carpeta destino para figuras y tablas de referencia contra las
cuales cada participante compara sus propios resultados
(actualmente vacía, pendiente de agregar los outputs esperados).

## Branches personales

Cada participante trabaja en su propio branch, con su propio
`CLAUDE.md` personalizado, `solution.py` y carpeta `outputs/`.

---

Este README se mantiene actualizado a medida que se agregan
archivos o carpetas al repositorio.
