# SPDX-License-Identifier: LGPL-3.0-or-later
"""Configuration file for the Sphinx documentation builder."""
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# import sys
# sys.path.insert(0, os.path.abspath('..'))
from datetime import datetime, timezone

# -- Project information -----------------------------------------------------

project = "DeePMD-GNN"
copyright = f"2024-{datetime.now(tz=timezone.utc).year}, DeepModeling"  # noqa: A001
author = "Jinzhe Zeng"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # "sphinxarg.ext",
    "myst_parser",
    # "sphinx_favicon",
    "deepmodeling_sphinx",
    "dargs.sphinx",
    # "sphinxcontrib.bibtex",
    # "sphinx_design",
    "autoapi.extension",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
# html_logo = "_static/logo.svg"
html_static_path = ["_static"]
html_js_files: list[str] = []
html_css_files = ["css/custom.css"]
html_extra_path = ["report.html", "fire.png", "bundle.js", "bundle.css"]

html_theme_options = {
    "github_url": "https://github.com/deepmodeling/deepmd-gnn",
    "gitlab_url": "https://gitlab.com/RutgersLBSR/deepmd-gnn",
    "logo": {
        "text": "DeePMD-GNN",
        "alt_text": "DeePMD-GNN",
    },
}

html_context = {
    "github_user": "deepmodeling",
    "github_repo": "deepmd-gnn",
    "github_version": "master",
    "doc_path": "docs",
}

myst_heading_anchors = 3

# favicons = [
#     {
#         "rel": "icon",
#         "static-file": "logo.svg",
#         "type": "image/svg+xml",
#     },
# ]

enable_deepmodeling = False

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
    "attrs_inline",
]
mathjax_path = (
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"
)
mathjax_options = {
    "integrity": "sha512-6FaAxxHuKuzaGHWnV00ftWqP3luSBRSopnNAA2RvQH1fOfnF/A1wOfiUWF7cLIOFcfb1dEhXwo5VG3DAisocRw==",
    "crossorigin": "anonymous",
}
mathjax3_config = {
    "loader": {"load": ["[tex]/mhchem"]},
    "tex": {"packages": {"[+]": ["mhchem"]}},
}

execution_mode = "off"
numpydoc_show_class_members = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "python": ("https://docs.python.org/", None),
    "deepmd": ("https://docs.deepmodeling.com/projects/deepmd/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
autoapi_dirs = ["../deepmd_gnn"]
