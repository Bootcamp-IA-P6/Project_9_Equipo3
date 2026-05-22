"""
src/data/loader.py

Carga y valida el dataset de comentarios de YouTube.
Responsabilidad única: leer el CSV y verificar que tiene las columnas esperadas.
El preprocesamiento viene después (src/features/text_preprocessor.py).
"""

import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columnas obligatorias en el dataset
REQUIRED_COLUMNS = {"Text", "IsToxic"}

# Sublabels opcionales (pueden no estar presentes)
SUBLABEL_COLUMNS = {
    "IsAbusive", "IsProvocative", "IsHatespeech",
    "IsRacist", "IsObscene", "IsThreat",
}


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """
    Carga el CSV crudo de comentarios de YouTube.

    Args:
        path: Ruta al archivo CSV.

    Returns:
        DataFrame validado y limpio a nivel estructural.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si faltan columnas obligatorias.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    logger.info(f"Cargando dataset: {path}")
    df = pd.read_csv(path)
    logger.info(f"  Shape: {df.shape}")

    _validate_columns(df)
    df = _clean_structure(df)

    logger.info(f"  Toxicos: {df['IsToxic'].sum()} ({df['IsToxic'].mean()*100:.1f}%)")
    return df


def load_preprocessed_data(path: str | Path) -> pd.DataFrame:
    """
    Carga el CSV preprocesado (con columna clean_text).
    Generado por el notebook 02 o por run_pipeline.

    Args:
        path: Ruta al CSV preprocesado.

    Returns:
        DataFrame con columna clean_text lista para vectorizar.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Datos preprocesados no encontrados: {path}\n"
            f"Ejecuta: python -m src.pipeline.run_pipeline"
        )

    df = pd.read_csv(path)
    if "clean_text" not in df.columns:
        raise ValueError("El CSV no tiene columna 'clean_text'. Regenera el preprocesamiento.")

    logger.info(f"Datos preprocesados cargados: {df.shape}")
    return df


# ── Funciones internas ────────────────────────────────────────────────────────

def _validate_columns(df: pd.DataFrame) -> None:
    """Verifica que el dataset tenga las columnas obligatorias."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Columnas obligatorias ausentes: {missing}\n"
            f"Columnas encontradas: {list(df.columns)}"
        )
    logger.info(f"  Columnas validadas ✅")


def _clean_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza estructural mínima:
    - Elimina filas con Text vacío
    - Convierte IsToxic a bool
    - Convierte sublabels a bool si existen
    """
    df = df.copy()

    # Texto
    df["Text"] = df["Text"].fillna("").astype(str).str.strip()
    df = df[df["Text"] != ""].reset_index(drop=True)

    # Target binario
    df["IsToxic"] = df["IsToxic"].astype(bool)

    # Sublabels
    for col in SUBLABEL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Eliminar duplicados
    n_before = len(df)
    df = df.drop_duplicates(subset=["Text"]).reset_index(drop=True)
    if len(df) < n_before:
        logger.warning(f"  {n_before - len(df)} duplicados eliminados")

    return df