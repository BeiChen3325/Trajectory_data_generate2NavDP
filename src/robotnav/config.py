"""Small TOML configuration helpers."""

from pathlib import Path
from typing import Any


def load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file on Python 3.10 and newer."""

    if not path.exists():
        raise FileNotFoundError(path)
    import tomli as tomllib

    with path.open("rb") as file:
        return tomllib.load(file)
