# CLAUDE.md — Ecología de Comunidades con Claude Code

Contexto base que el agente debe leer antes de trabajar en este
repositorio.

---

> **Nota para participantes del curso:** las secciones marcadas con ✏️
> son las que debes personalizar en tu branch antes del día 2. Las
> demás secciones son comunes a todos los participantes — puedes
> usarlas tal como están o extenderlas.

---

## ✏️ Quién soy

<!-- Reemplaza este bloque con tu información antes del día 2 -->

Soy Rosario Paz Vargas Araya, bióloga marina y candidata a doctora en
Ecología Integrativa (Universidad Mayor, Santiago, Chile), en cotutela
con la Universitat de les Illes Balears (UIB), España.

Investigo respuestas de comunidades microbianas a la contaminación por
PFAS (PFOS y PFOA) en ecosistemas de agua dulce antárticos, patagónicos
y mediterráneos. Trabajo en el Centro GEMA (Genómica, Ecología y Medio
Ambiente), donde también soy encargada de laboratorio.

**Métodos que uso:** secuenciación de amplicones, metatranscriptómica,
experimentos ecotoxicológicos.
**Disciplinas base:** biología marina, limnología, ecotoxicología,
ecología molecular.
**Experiencia de campo:** Antártica y Patagonia (expediciones ECA 58, 59 y 60). No tengo experiencia previa en
programación ni en Python. Tengo conocimientos básicos de
estadística (regresión, correlación) y de ecología de comunidades.
Necesito que me expliques los resultados en lenguaje biológico
además de estadístico.

## Sobre el curso

Curso de 2 mañanas para biólogos y ecólogos sin experiencia previa
en programación ni terminal. El hilo conductor es reproducir el
análisis de Neilson et al. (2017) sobre el efecto de la aridez en
el microbioma de suelo del desierto de Atacama: cada participante
dirige al agente para calcular métricas de diversidad y
composición microbiana, obtener las correlaciones (Spearman) entre
aridez/humedad relativa del suelo y esas métricas, y reproducir
las figuras del paper.

**Hipótesis:**
- H1: A menor humedad relativa del suelo (AvgSoilRH), menor
  diversidad alfa (Shannon).
- H2: Los sitios más áridos (Yungay) se separan de los menos áridos
  (Baquedano) en la ordenación PCoA a lo largo del primer eje.
- H3: AvgSoilRH explica más varianza composicional que temperatura
  o elevación (R² esperado: 0.20–0.50).

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

4. **Outputs**: guarda todas las figuras y tablas resultantes en
`outputs/` dentro del branch personal. Nunca escribas en `data/`
ni en `expected_outputs/`.

5. **Errores**: si encuentras un error, muéstralo completo y
explícalo en lenguaje simple. No intentes corregirlo más de
dos veces sin consultar primero.

6. **Verificación biológica**: al terminar cada análisis, pregunta
si el resultado tiene sentido biológico antes de continuar.

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

`docs/programa_curso_vJulio2026.pdf` es el programa oficial del
curso (horario y contenidos por módulo). `docs/pasos_curso_manana1.md`
es la checklist para la Mañana 1 (por ahora un esqueleto) y
`docs/pasos_curso_manana2.md` es la checklist detallada, módulo por
módulo, para la Mañana 2.

`data/abundancias.tsv` es la tabla de abundancias de OTUs (una fila
por OTU, una columna por muestra), exportada desde el `.biom`
original del estudio — 1 109 OTUs × 54 muestras:

```
#OTU ID    BAQ2420.1.1  BAQ2420.1.2  BAQ2420.1.3  BAQ2420.2  ...
409faa5f5353e543bf6d99125c7c0e83  0.0  0.0  0.0    0.0  ...
1237d5925a7176fced9dda961a86c684  0.0  0.0  13.0   103.0 ...
```

`data/metadata.tsv` es la metadata de cada muestra (transecto,
sitio, variables ambientales) — 75 muestras × 21 columnas:

```
sample-id    barcode-sequence  elevation  ...  transect-name  site-name  ...
BAQ1370.1.2  GCCCAAGTTCAC      1370       ...  Baquedano      BAQ1370    ...
BAQ1370.3    GCGCCGAATCTT      1370       ...  Baquedano      BAQ1370    ...
```

Solo 54 de las 75 muestras en `metadata.tsv` tienen secuenciación
en `abundancias.tsv` (las 21 restantes quedaron sin datos de
abundancia) — al cruzar ambos archivos, filtrar por la
intersección de IDs, no asumir que coinciden 1 a 1.

`expected_outputs/` contiene las figuras y tablas de referencia
contra las que cada participante compara sus propios resultados,
cubriendo H1 (Shannon vs. humedad relativa), H2 (PCoA Bray-Curtis)
y H3 (PERMANOVA univariada de humedad, temperatura y elevación).
Generadas con `generate_expected_outputs.py` (fuera de este
listado porque vive en la raíz del branch de instructores, no en
esta carpeta).

Cada participante trabaja en su propio branch personal, con su
propio `CLAUDE.md` personalizado, un script `solution.py`, y una
carpeta `outputs/` para sus figuras y tablas.

## Variables clave del dataset

Las columnas más relevantes de `data/metadata.tsv` para el
análisis del día 2:

| Columna | Descripción |
|---|---|
| `sample-id` | Identificador de muestra (clave para el join con abundancias) |
| `transect-name` | Transecto: `Baquedano` (BAQ, más húmedo) o `Yungay` (YUN, más árido) |
| `AvgSoilRH` | Humedad relativa promedio del suelo — variable independiente principal |
| `temperature` | Temperatura del suelo |
| `elevation` | Elevación del sitio en metros |
| `vegetation` | Presencia/ausencia de vegetación |

Nota: `average-soil-relative-humidity` en `data/metadata.tsv` va de
~15 % a 100 % en ambos transectos (medias similares entre
Baquedano ~64 % y Yungay ~67 %) — más amplio y solapado de lo que
se podría anticipar por la aridez relativa de cada transecto. No
asumas rangos estrechos por transecto al validar resultados.

## Skills

Las siguientes instrucciones se aplican automáticamente cuando la
tarea corresponde a cada categoría. No es necesario repetirlas en
cada prompt.

### Cuando generes una figura

**Trigger:** "genera una figura", "haz un plot", "visualiza",
"grafica"

1. Usa paleta daltónica: `viridis` para gradientes continuos,
   `Set2` para grupos discretos.
2. Exporta en PNG a 300 dpi y en PDF vectorial.
3. Guarda ambos archivos en `outputs/` con nombre descriptivo en
   inglés (ej. `fig1_shannon_vs_avgsoilrh.png`).
4. Tamaño de figura por defecto: 8 × 6 pulgadas.
5. Incluye siempre: título descriptivo, etiquetas de ejes con
   unidades, leyenda si hay más de un grupo.
6. Si es un scatter con regresión, incluye la banda de confianza
   al 95 % y muestra R² y p-valor en el gráfico.

### Cuando analices diversidad alfa

**Trigger:** "analiza diversidad alfa", "calcula Shannon",
"diversidad por muestra", "índice de diversidad"

1. Muestra primero un resumen estadístico de los índices (media,
   mediana, rango) por transecto (Baquedano vs. Yungay).
2. Genera un boxplot por transecto antes de hacer cualquier
   regresión.
3. Usa `AvgSoilRH` como variable independiente principal salvo
   que se indique otra.
4. Reporta siempre: coeficiente de correlación (r o Spearman ρ),
   R², p-valor y n.
5. Si R² < 0.05, avisa antes de continuar — puede indicar un
   problema con los datos o el análisis.

### Cuando analices composición (beta-diversidad)

**Trigger:** "analiza composición", "Bray-Curtis", "PCoA",
"beta-diversidad", "composición de comunidades"

1. Usa distancias Bray-Curtis salvo que se especifique otra
   métrica.
2. Para la ordenación, genera PCoA por defecto. Colorea por
   gradiente continuo de `AvgSoilRH` con paleta `viridis`.
3. Muestra el porcentaje de varianza explicado en los ejes de
   la ordenación (PC1 y PC2).
4. Para modelar composición ~ variables ambientales, usa dbRDA
   o PERMANOVA con 999 permutaciones.
5. Reporta para cada variable: F, R² y p-valor; ordena la tabla
   de mayor a menor R².
6. Avisa si n < 10 por grupo — las permutaciones pueden ser
   insuficientes.
7. Corre `betadisper` + `permutest` junto con la PERMANOVA. Si la
   dispersión es significativa, advierte que el resultado de adonis2
   puede reflejar heterogeneidad de varianza, no diferencia de
   composición. Reporta ambos.

### Cuando revises resultados

**Trigger:** "revisa estos resultados", "¿tiene sentido?",
"valida el análisis", "compara con la referencia"

1. Evalúa si el resultado es biológicamente plausible para un
   gradiente de aridez en suelo de desierto.
2. Compara los valores numéricos con `expected_outputs/` si el
   archivo correspondiente existe.
3. Señala cualquier valor fuera de rango (R² negativo, p-valor > 1,
   diversidades negativas, etc.).
4. Si hay discrepancia con `expected_outputs/`, identifica la
   causa probable (normalización, filtrado, índice diferente).
5. Resume el resultado en máximo dos oraciones en lenguaje
   biológico, no estadístico.

### Cuando visualices composición taxonómica

**Trigger:** "barplot", "composición taxonómica", "abundancia relativa
por taxón", "top taxa"

1. Aglomera al rango pedido (phylum, género, etc.) antes de graficar.
2. Convierte a abundancia relativa; muestra top-N taxa y agrupa el resto
   como "Other".
3. Recuerda que estos gráficos son descriptivos, no una prueba
   estadística — no infieras diferencias significativas de un barplot.
   
### Control de versiones (git)

**Trigger:** siempre que crees, modifiques o ejecutes un archivo

1. Después de crear o modificar cualquier archivo, corre `git add <archivo>`
   y luego `git commit`. No acumules cambios sin registrar.
2. Escribe mensajes de commit descriptivos en presente y en español:
   "agrega cálculo de Shannon", "corrige filtro de prevalencia".
   Evita mensajes genéricos como "update" o "cambios".
3. Un commit por unidad lógica de trabajo (un análisis, una corrección),
   no un commit gigante con todo mezclado.
4. Antes de commitear, muéstrame `git status` y `git diff --staged` para
   que confirme qué entra. No hagas `git add .` a ciegas.
5. Nunca ejecutes `git push`, `git reset --hard`, `git rebase` ni fuerces
   historial sin que yo lo pida explícitamente.
   
## ✏️ Notas personales

<!-- Espacio libre para contexto adicional que quieras agregar
     durante el curso: observaciones del dataset, decisiones de
     análisis, preguntas para el instructor, etc. -->

## Paper base

Neilson, J.W., Califf, K., Cardona, C., Copeland, A., van Treuren,
W., Josephson, K.L., Knight, R., Gilbert, J.A., Quade, J.,
Caporaso, J.G., Maier, R.M. (2017). Significant Impacts of
Increasing Aridity on the Arid Soil Microbiome. mSystems, 2(3):
e00195-16.
