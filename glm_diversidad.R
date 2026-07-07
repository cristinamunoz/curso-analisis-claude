# Modelo lineal (GLM) de Shannon en funcion de variables
# ambientales, con el transecto como efecto aleatorio (intercepto
# aleatorio). Requiere outputs/diversidad_alfa_por_muestra.tsv,
# generado por solution.py (exportar_tabla_para_gam).
#
# Se probo tambien un GAM con suavizados para las tres variables
# ambientales: mgcv penalizo los tres terminos a edf=1 (rectas) y
# el AIC fue practicamente identico al del modelo lineal, sin
# evidencia de no linealidad. Por eso se deja solo el GLM.

library(mgcv)

diversidad <- read.delim("outputs/diversidad_alfa_por_muestra.tsv")
diversidad$transect <- as.factor(diversidad$transect)

modelo <- gam(
  shannon ~ avg_soil_rh + temperature + elevation +
    s(transect, bs = "re"),
  data = diversidad,
  method = "REML"
)

cat("\n=== Resumen del modelo lineal (GLM + efecto aleatorio) ===\n")
print(summary(modelo))

tabla_coeficientes <- as.data.frame(summary(modelo)$p.table)
tabla_coeficientes$termino <- rownames(tabla_coeficientes)
tabla_coeficientes <- tabla_coeficientes[
  , c("termino", "Estimate", "Std. Error", "t value", "Pr(>|t|)")
]
write.table(
  tabla_coeficientes,
  "outputs/glm_shannon_variables_ambientales.tsv",
  sep = "\t", row.names = FALSE, quote = FALSE
)

cat("\nR2 ajustado:", summary(modelo)$r.sq, "\n")
cat("Devianza explicada:", summary(modelo)$dev.expl, "\n")

# Efectos parciales del modelo (predicciones a nivel poblacional,
# sin el efecto aleatorio de transecto), superpuestos a los
# residuos parciales de cada muestra.
variables <- c("avg_soil_rh", "temperature", "elevation")
etiquetas <- c(
  avg_soil_rh = "Average soil relative humidity (%)",
  temperature = "Average soil temperature (C)",
  elevation = "Elevation (m)"
)
colores_transecto <- c(Baquedano = "#66c2a5", Yungay = "#fc8d62")

prediccion_terminos <- predict(modelo, type = "terms", se.fit = TRUE)
residuos <- residuals(modelo, type = "response")

graficar_efectos_parciales <- function() {
  par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1))
  for (variable in variables) {
    x <- diversidad[[variable]]
    termino <- prediccion_terminos$fit[, variable]
    error_estandar <- prediccion_terminos$se.fit[, variable]
    residuo_parcial <- residuos + termino

    orden <- order(x)

    plot(
      x, residuo_parcial,
      col = colores_transecto[as.character(diversidad$transect)],
      pch = 19,
      xlab = etiquetas[variable], ylab = "Shannon (efecto parcial)",
      main = etiquetas[variable]
    )
    poligono_x <- c(x[orden], rev(x[orden]))
    poligono_y <- c(
      (termino + 1.96 * error_estandar)[orden],
      rev((termino - 1.96 * error_estandar)[orden])
    )
    polygon(poligono_x, poligono_y, col = "#00000022", border = NA)
    lines(x[orden], termino[orden], col = "black", lwd = 2)
  }
  legend(
    "topright", legend = names(colores_transecto),
    col = colores_transecto, pch = 19, bty = "n"
  )
}

png(
  "outputs/fig_glm_shannon_efectos_parciales.png",
  width = 12, height = 5, units = "in", res = 300
)
graficar_efectos_parciales()
dev.off()

pdf(
  "outputs/fig_glm_shannon_efectos_parciales.pdf",
  width = 12, height = 5
)
graficar_efectos_parciales()
dev.off()

cat("\nTabla y figura guardadas en outputs/\n")
