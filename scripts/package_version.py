"""Print the package version from pyproject.toml (a release tag is v + it)."""
import tomllib
from pathlib import Path

pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
print(tomllib.loads(pyproject.read_text())["project"]["version"])
