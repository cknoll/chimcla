import recommonmark
from recommonmark.parser import CommonMarkParser

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
    # ...
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    # ...
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = ['.rst', '.md']
parser = CommonMarkParser()





# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
