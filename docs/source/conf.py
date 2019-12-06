# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))

import e2cnn



# -- Project information -----------------------------------------------------

project = 'e2cnn'
copyright = '2019, Qualcomm Innovation Center, Inc. Developed by Gabriele Cesa, Maurice Weiler'
author = 'Gabriele Cesa, Maurice Weiler'

# The short X.Y version
version = e2cnn.__version__
# The full version, including alpha/beta/rc tags
release = e2cnn.__version__


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
#needs_sphinx = '1.8.5'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    #'sphinx.ext.autosummary',
    #'sphinx.ext.doctest',
    'sphinx.ext.todo',
    #'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    #'sphinx.ext.imgmath',
    #'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',    
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx_rtd_theme',
]

napoleon_numpy_docstring = False
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_rtype = False # more legible

napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False


autoclass_content = "both" #"init"
#autodoc_member_order = "groupwise"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = False

# needs to be empty so we can use automodule to link to all subpackages without adding automatically the docs of all members
autodoc_default_flags = ['members']

add_module_names = False

typehints_fully_qualified = False
set_type_checking_flag = False
typehints_document_rtype = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

mathjax_config = {                  
    "TeX": {         
        #"packages": {'[+]': ['bm']},
        "Macros": {
            "bold": ['{\\bf #1}', 1],
            "R": '{\\mathbb R}',
            "N": '{\\mathbb N}',
            "Z": '{\\mathbb Z}',
            "Ind": ['{\operatorname{Ind}_{#1}^{#2}}', 2],
            "Res": ['{\operatorname{Res}_{#1}^{#2}}', 2],
            "GL" : ['{\operatorname{GL}(#1)}', 1],
            "E" : ['{\operatorname{E}(#1)}', 1],
            "SE" : ['{\operatorname{SE}(#1)}', 1],
            "O" : ['{\operatorname{O}(#1)}', 1],
            "SO" : ['{\operatorname{SO}(#1)}', 1],
            "U" : ['{\operatorname{U}(#1)}', 1],
            "D" : ['{\operatorname{D}_{#1}}', 1],
            "C" : ['{\operatorname{C}_{#1}}', 1],
            "DN" : '{\operatorname{D}_{\!N}}',
            "CN" : '{\operatorname{C}_{\!N}}',            
        },                       
    }                           
}       

source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
#pygments_style = 'default'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'sticky_navigation': True,
    'navigation_depth': 2, #4
    'titles_only': False,
    #'logo_only': True,
}




# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'e2cnndoc'

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
#intersphinx_mapping = {'https://docs.python.org/': None}
intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'PyTorch': ('http://pytorch.org/docs/master/', None),
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

show_authors = True



def setup(app):
    app.add_css_file('custom.css')
    
    
    

