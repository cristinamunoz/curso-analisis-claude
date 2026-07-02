# CLAUDE.md — Ecología de Comunidades con Claude Code

Contexto base que el agente debe leer antes de trabajar en este
repositorio.

## Sobre el curso

Curso de 2 mañanas para biólogos y ecólogos sin experiencia previa
en programación ni terminal. El hilo conductor es reproducir el
análisis de Neilson et al. (2017) sobre el efecto de la aridez en
el microbioma de suelo del desierto de Atacama: cada participante
dirige al agente para calcular métricas de diversidad y
composición microbiana, obtener las correlaciones (Spearman) entre
aridez/humedad relativa del suelo y esas métricas, y reproducir
las figuras del paper.

## Reglas para el agente

1. **Estilo de código**: todo el código en Python debe seguir la
   guía de estilo PEP8, con un largo máximo de línea de 79
   caracteres.

2. **Audiencia**: el curso está dirigido a científicos que no son
   bioinformáticos, con conocimientos básicos de informática. Las
   explicaciones, nombres de variables, mensajes de error y
   comentarios deben ser claros, evitando jerga técnica innecesaria.

3. **Confirmación antes de ejecutar**: nunca ejecutes comandos,
   instalaciones, scripts ni cambios sobre archivos del curso sin
   mostrar antes exactamente qué se va a ejecutar y esperar
   confirmación explícita del usuario.

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
```

`data/abundancias.tsv` y `data/metadata.tsv` están separados por
tabs. Solo 54 de las 75 muestras en `metadata.tsv` tienen
secuenciación en `abundancias.tsv` — filtrar por la intersección de
IDs antes de cruzar ambos archivos.

`docs/programa_curso_vJulio2026.pdf` es el programa oficial del
curso (horario y contenidos por módulo). `docs/pasos_curso_manana1.md`
y `docs/pasos_curso_manana2.md` traducen ese programa en una
checklist orientada a participantes, una por cada mañana; por
ahora solo la Mañana 2 está detallada, la Mañana 1 es un esqueleto
pendiente de completar.

Cada participante trabaja en su propio branch personal, con su
propio `CLAUDE.md` personalizado, `solution.py` y carpeta
`outputs/`.

## Paper base

Neilson, J.W., Califf, K., Cardona, C., Copeland, A., van Treuren,
W., Josephson, K.L., Knight, R., Gilbert, J.A., Quade, J.,
Caporaso, J.G., Maier, R.M. (2017). Significant Impacts of
Increasing Aridity on the Arid Soil Microbiome. mSystems, 2(3):
e00195-16.
