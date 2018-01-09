import sys, os, docutils

master_doc = 'index'

project = 'libxnd'
copyright = '2016-2017, Plures Project'
version = 'XYZ' # RELEASE_VERSION
release = 'XYZ'

primary_domain = 'c'

extensions = ['sphinx.ext.ifconfig']
source_suffix = '.rst'
add_function_parentheses = False
pygments_style = 'sphinx'
exclude_patterns = ['doc']

html_title = 'libxnd'
#html_logo = None
#html_favicon = None
#html_static_path = ['_static']
html_domain_indices = False
html_use_index = False
html_show_sourcelink = False
#html_add_permalinks = None
html_copy_source = False

def setup(app):
    app.add_crossref_type('topic', 'topic', 'single: %s',
                          docutils.nodes.strong)


