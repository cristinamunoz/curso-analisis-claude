# Pasos del curso — Mañana 2: Análisis real: microbioma del desierto de Atacama

Guía operativa para participantes, basada en
`docs/programa_curso_vJulio2026.pdf`. Cada módulo lista qué hacer
tú (el participante) y qué deberías tener listo al terminar.

Ver también [pasos_curso_manana1.md](pasos_curso_manana1.md).

---

### Módulo 7 · El paper y la pregunta biológica (09:00–09:30)

1. Revisa y ajusta tu `CLAUDE.md` personal si te quedó pendiente
   algo del día 1.
2. Pide a Claude Code que lea `paper/mSystems.00195-16.pdf`
   (Neilson et al. 2017) y te resuma la pregunta central del
   estudio: **¿cómo afecta la aridez creciente a la diversidad y
   composición microbiana del suelo?**
3. Explora el dataset con el agente: pídele que liste los
   transectos, las variables ambientales disponibles y la
   estructura de los archivos en `data/`.
4. Antes de programar nada, discute con el agente (y anota en tu
   `CLAUDE.md` o en notas propias) qué análisis propondrías para
   responder la pregunta del paper. No es necesario decidir el
   análisis final todavía — es una lluvia de ideas guiada.

**Resultado esperado:** tienes claro qué pregunta biológica estás
respondiendo y qué datos tienes disponibles para hacerlo.

### Módulo 8 · Análisis libre: diversidad y composición (09:30–10:30)

1. Dirige a Claude Code hacia los análisis que consideres
   apropiados para responder la pregunta biológica del Módulo 7 —
   no hay prompts predefinidos, tú decides el camino.
2. Foco sugerido por el programa:
   - Diversidad alfa (riqueza, Shannon, etc.) vs. gradiente
     ambiental (aridez / humedad relativa del suelo). Usa
     correlación de Spearman, el mismo método que el paper.
   - Composición de comunidades a nivel de OTU: abundancia relativa
     de los OTUs más comunes a lo largo del gradiente, y/o
     diversidad beta (ordenación NMDS/PCoA con distancia
     Bray-Curtis entre muestras) — el mismo enfoque que usa el
     paper en su Fig. S2.
3. Pide plan antes de que el agente ejecute código (regla del
   `CLAUDE.md`) y confírmalo explícitamente en cada paso.
4. Si te atascas en la lógica científica (no en el código), pide
   ayuda a los instructores — ellos orientan la pregunta, no la
   implementación.
5. Guarda el código generado en tu carpeta de participante
   (`solution.py` u otro nombre que definas en tu branch).

**Resultado esperado:** al menos un análisis cuantitativo (tabla
de estadísticos o correlaciones) que relacione una métrica de
diversidad/composición con el gradiente de aridez.

### Coffee break (10:30–11:00)

### Módulo 9 · Visualización y cierre del análisis (11:00–12:00)

1. Pide al agente que genere las figuras que mejor representen tus
   hallazgos (por ejemplo, diversidad vs. aridez, ordenación de
   comunidades por OTU).
2. Compara tus resultados con los de referencia en
   `expected_outputs/` — la pregunta guía no es "¿son idénticos?"
   sino **¿coincide la dirección del efecto?**
3. Ajusta iterativamente vía el agente hasta quedar conforme con
   tus figuras.
4. Exporta tus figuras finales en PNG y PDF a tu carpeta de
   outputs.
5. Haz commit de tu trabajo a tu branch personal (código, figuras,
   `CLAUDE.md` actualizado).

**Resultado esperado:** figuras exportadas y comiteadas en tu
branch, con una conclusión clara sobre si tus resultados apuntan
en la misma dirección que el paper.

### Módulo 10 · Comparación de branches y reflexión colectiva (12:00–12:45)

1. Revisa los branches de otros participantes junto al grupo —
   mismo paper y dataset, caminos distintos.
2. Participa en la discusión guiada:
   - ¿Qué modelo y qué `CLAUDE.md` produjeron mejores resultados
     con menos iteraciones?
   - ¿Qué skills resultaron más útiles? ¿Qué faltó en tu
     `CLAUDE.md`?
   - ¿Qué análisis emergieron que el paper original no hizo?
3. Reflexiona sobre el prompting como variable experimental en
   ciencia reproducible.

**Resultado esperado:** una idea concreta de qué cambiarías en tu
`CLAUDE.md` o en tu forma de dirigir al agente para la próxima vez.

### Módulo 11 · Cierre y próximos pasos (12:45–13:00)

1. Repasa el flujo completo: `CLAUDE.md` + skills + modelo →
   pregunta biológica → agente → supervisión → resultado.
2. Anota los recursos compartidos (documentación de Anthropic,
   foros de QIIME 2, comunidades de práctica) para seguir
   aprendiendo.

**Resultado final por participante:** branch personal en GitHub
con el análisis del microbioma del desierto de Atacama
implementado en Python — figuras y tabla de resultados exportadas,
generadas de forma autónoma a partir de la pregunta biológica del
paper.
