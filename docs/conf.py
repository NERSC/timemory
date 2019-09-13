# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# sys.path.insert(0, os.path.abspath('.'))
import os
import sys
import shutil
import subprocess as sp


def install(package):
    sp.call([sys.executable, "-m", "pip", "install", package])

# -- Project information -----------------------------------------------------


project = 'TiMemory'
copyright = '2019, Jonathan R. Madsen'
author = 'Jonathan R. Madsen'

version = open(os.path.join('..', 'VERSION')).read().strip()
# The full version, including alpha/beta/rc tags
release = version

_docdir = os.path.realpath(os.getcwd())
_srcdir = os.path.realpath(os.path.join(os.getcwd(), ".."))
_bindir = os.path.realpath(os.path.join(os.getcwd(), "build-timemory"))
_doxbin = os.path.realpath(os.path.join(_bindir, "doc", "html"))
_doxdir = os.path.realpath(os.path.join(_docdir, "doxygen-docs"))
_sitedir = os.path.realpath(os.path.join(os.getcwd(), "..", "site"))

if not os.path.exists(_bindir):
    os.makedirs(_bindir)
os.chdir(_bindir)
sp.run(["cmake",
        "-DTIMEMORY_DOXYGEN_DOCS=ON", "-DENABLE_DOXYGEN_HTML_DOCS=ON",
        "-DENABLE_DOXYGEN_LATEX_DOCS=OFF", "-DENABLE_DOXYGEN_MAN_DOCS=OFF",
        _srcdir])
sp.run(["cmake", "--build", os.getcwd(), "--target", "doc"])
if os.path.exists(_doxdir):
    shutil.rmtree(_doxdir)
shutil.copytree(_doxbin, _doxdir)

install('mkdocs-cinder')
install('mkdocs-inspired')
os.chdir(_srcdir)
sp.run(["mkdocs", "build"])
os.chdir(_docdir)
html_extra_path = [_sitedir]

os.chdir(_docdir)
# shutil.rmtree(_bindir)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'recommonmark',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'default'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
