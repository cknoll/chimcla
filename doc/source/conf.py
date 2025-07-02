import chimcla

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'chimcla'
copyright = '2025, Carsten Knoll, Sascha Weber'
author = 'Carsten Knoll, Sascha Weber'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'autodoc2',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    # 'sphinx.ext.intersphinx',
]

autodoc2_render_plugin = "myst"
autodoc2_hidden_objects = []

autodoc2_hide_module_imports = False
autodoc2_packages = [
    "../../src/chimcla",
]

# markdown conversion extensions
myst_enable_extensions = [
    # "amsmath",
    # "attrs_inline",
    "colon_fence",
    # "deflist",
    # "dollarmath",
    # "fieldlist",
    # "html_admonition",
    # "html_image",
    # "linkify",
    # "replacements",
    # "smartquotes",
    # "strikethrough",
    # "substitution",
    # "tasklist",
]


templates_path = ['_templates']
exclude_patterns = []


source_suffix = [".rst", ".md"]
source_parsers = {
    ".md": "myst_parser.sphinx_",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
