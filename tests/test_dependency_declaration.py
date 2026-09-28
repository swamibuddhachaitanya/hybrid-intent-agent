"""Every third-party import under src/ must be declared in requirements.txt.

Regression test for the Docker image: the Dockerfile pre-downloads the MiniLM
weights with `from sentence_transformers import ...`, but requirements.txt never
listed sentence-transformers, so the build failed at that step. The two dependency
files had also drifted, and only requirements.txt is installed by the image.
"""

import ast
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

SRC_DIR = Path("src")
REQUIREMENTS = Path("requirements.txt")


def _declared_distributions() -> set:
    declared = set()
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("[")[0]
        for separator in (">", "=", "<", "!"):
            name = name.split(separator)[0]
        declared.add(name.strip().lower().replace("_", "-"))
    return declared


def _imported_top_level_modules() -> set:
    modules = set()
    for path in SRC_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def test_every_third_party_import_is_declared():
    declared = _declared_distributions()
    module_to_distributions = packages_distributions()
    local_packages = {child.name for child in SRC_DIR.iterdir() if child.is_dir()}

    undeclared = []
    for module in sorted(_imported_top_level_modules()):
        if module in sys.stdlib_module_names or module in local_packages or module == "src":
            continue
        distributions = module_to_distributions.get(module)
        if not distributions:
            continue  # not provided by an installed distribution
        if not any(d.lower().replace("_", "-") in declared for d in distributions):
            undeclared.append(f"{module} (from {distributions})")

    assert not undeclared, (
        "imports missing from requirements.txt, so the container image will fail: "
        f"{undeclared}"
    )
