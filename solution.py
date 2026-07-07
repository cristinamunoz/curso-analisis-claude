"""Carga y filtra datos de abundancia y metadata para el analisis
de aridez y microbioma (Neilson et al. 2017)."""

import pandas as pd

RUTA_ABUNDANCIAS = "data/abundancias.tsv"
RUTA_METADATA = "data/metadata.tsv"


def cargar_datos():
    """Lee las tablas de abundancias y metadata desde disco."""
    abundancias = pd.read_csv(
        RUTA_ABUNDANCIAS, sep="\t", index_col=0
    )
    metadata = pd.read_csv(
        RUTA_METADATA, sep="\t", index_col="sample-id"
    )
    return abundancias, metadata


def filtrar_muestras_en_comun(abundancias, metadata):
    """Deja solo las muestras presentes en ambas tablas.

    Solo 54 de las 75 muestras de metadata tienen datos de
    secuenciacion, asi que no se puede asumir correspondencia
    1 a 1 entre ambos archivos.
    """
    muestras_comunes = abundancias.columns.intersection(
        metadata.index
    )
    abundancias_filtradas = abundancias[muestras_comunes]
    metadata_filtrada = metadata.loc[muestras_comunes]
    return abundancias_filtradas, metadata_filtrada


def main():
    abundancias, metadata = cargar_datos()
    abundancias_f, metadata_f = filtrar_muestras_en_comun(
        abundancias, metadata
    )
    print(f"OTUs: {abundancias_f.shape[0]}")
    print(f"Muestras en comun: {abundancias_f.shape[1]}")
    print(metadata_f["transect-name"].value_counts())


if __name__ == "__main__":
    main()
