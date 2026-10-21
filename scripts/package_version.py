"""Print the package version from pyproject.toml (a release tag is v + it).

Runs on every supported Python (3.10+) and platform.
"""
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: no tomllib
    tomllib = None

root = Path(__file__).resolve().parents[1]


def read(name):
    # UTF-8 explicitly: the locale encoding (cp1252 on Windows) cannot
    # decode every file.
    return (root / name).read_text(encoding="utf-8")


def table_version(name, table):
    """`version` of the TOML table `table` (dotted) in file `name`."""
    text = read(name)
    if tomllib is not None:
        data = tomllib.loads(text)
        for key in table.split("."):
            data = data[key]
        return data["version"]
    # Without tomllib: the `version = "..."` line of that table's section.
    section = re.search(rf"^\[{re.escape(table)}\][ \t]*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    found = section and re.search(r'^version\s*=\s*"([^"]+)"', section.group(1), re.M)
    if not found:
        sys.exit(f"no version in [{table}] of {name}")
    return found.group(1)


print(table_version("pyproject.toml", "project"))
