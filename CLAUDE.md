# CLAUDE.md — Ecología de Comunidades con Claude Code

Contexto base que el agente debe leer antes de trabajar en este
repositorio.

## Sobre el curso

Curso de 2 mañanas para biólogos y ecólogos sin experiencia previa
en programación ni terminal. El hilo conductor es reproducir
PERMANOVA (Anderson, 2001) en Python: cada participante dirige al
agente para implementar el algoritmo, calcular los estadísticos del
paper y reproducir sus figuras.

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
├── paper/
│   ├── Anderson2001.pdf
│   └── resumen_algoritmo.md
├── data/
│   └── macroinvertebrados.csv
└── expected_outputs/
    ├── pcoa_paper.png
    └── tabla_permanova_paper.csv
```

Cada participante trabaja en su propio branch personal, con su
propio `CLAUDE.md` personalizado, `solution.py` y carpeta
`outputs/`.

## Paper base

Anderson, M.J. (2001). A new method for non-parametric
multivariate analysis of variance. Austral Ecology, 26(1): 32–46.
