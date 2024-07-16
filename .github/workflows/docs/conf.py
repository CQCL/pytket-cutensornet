# -*- coding: utf-8 -*-

# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

copyright = "2024 Quantinuum"
author = "Quantinuum"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_favicon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "enum_tools.autoenum",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
]

autosectionlabel_prefix_document = True

html_theme = "furo"

html_theme_options = {}

html_theme = "furo"
templates_path = ["quantinuum-sphinx/_templates"]
html_static_path = ["quantinuum-sphinx/_static", "_static"]
html_favicon = "quantinuum_sphinx/_static/assets/quantinuum_favicon.svg"

favicons = [
    "favicon.svg",
]

pytketdoc_base = "https://tket.quantinuum.com/api-docs/"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pytket": (pytketdoc_base, None),
    "qiskit": ("https://qiskit.org/documentation/", None),
    "qulacs": ("http://docs.qulacs.org/en/latest/", None),
}

autodoc_member_order = "groupwise"
