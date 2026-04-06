from pathlib import Path
import sys


# -- Project information -----------------------------------------------------

project = "rshf"
copyright = "2026, rshf contributors"
author = "rshf contributors"
release = "0.1"


# -- General configuration ---------------------------------------------------

# Ensure autodoc can import the local package when building from `docs/`.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}


# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"