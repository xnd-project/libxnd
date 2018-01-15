import sys, os, docutils


extensions = ['sphinx.ext.autodoc', 'sphinx.ext.imgmath',
              'sphinx.ext.intersphinx', 'sphinx.ext.coverage',
              'sphinx.ext.autosummary']

source_suffix = '.rst'
master_doc = 'index'
project = 'xnd'
copyright = '2017-2018, Plures Project'
version = 'v0.2.0b1'
release = 'v0.2.0b1'
exclude_patterns = ['doc', 'build']
pygments_style = 'sphinx'
html_static_path = ['_static']

primary_domain = 'py'
add_function_parentheses = False


def setup(app):
    app.add_crossref_type('topic', 'topic', 'single: %s',
                          docutils.nodes.strong)
    app.add_javascript("copybutton.js")



