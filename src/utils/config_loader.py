"""
Utilidad para cargar archivos de configuración YAML.
Todos los módulos deben usar esto en lugar de hardcodear valores.
"""
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """
    Carga un archivo YAML de configuración.

    Args:
        config_path: Ruta al archivo YAML (relativa a la raíz del proyecto).

    Returns:
        Diccionario con la configuración.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_pipeline_config() -> dict:
    return load_config("configs/pipeline.yaml")


def load_features_config() -> dict:
    return load_config("configs/features.yaml")


def load_models_config() -> dict:
    return load_config("configs/models.yaml")
