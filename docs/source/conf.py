# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import subprocess
import sys

project = "einx"
copyright = "Florian Fervers"
author = '<a href="https://fferflo.github.io/">Florian Fervers</a>'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = []
exclude_patterns = []

autosummary_generate = True

# TODO:
autodoc_type_aliases = {"ArrayLike": "numpy.typing.ArrayLike"}
autodoc_typehints_format = "fully-qualified"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_theme_options = {"show_toc_level": 1, "repository_url": "https://github.com/fferflo/einx", "use_repository_button": True}


# def run(app):
#     subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_compiled_code.py")])


# def setup(app):
#     app.connect("builder-inited", run)
