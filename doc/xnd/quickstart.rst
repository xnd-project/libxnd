.. meta::
   :robots: index,follow
   :description: xnd quickstart
   :keywords: xnd, install

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


Quick Start
===========

Prerequisites
~~~~~~~~~~~~~

Python2 is not supported. If not already present, install the Python3
development packages:

.. code-block:: sh

   # Debian, Ubuntu:
   sudo apt-get install gcc make
   sudo apt-get install python3-dev

   # Fedora, RedHat:
   sudo yum install gcc make
   sudo yum install python3-devel

   # openSUSE:
   sudo zypper install gcc make
   sudo zypper install python3-devel

   # BSD:
   # You know what to do.

   # Mac OS X:
   # Install Xcode and Python 3 headers.


Install
~~~~~~~

If `pip <http://pypi.python.org/pypi/pip>`_ is present on the system, installation
should be as easy as:

.. code-block:: sh

   pip install xnd


Otherwise:

.. code-block:: sh

   tar xvzf xnd.2.0b1.tar.gz
   cd xnd.2.0b1
   python3 setup.py install


Windows
~~~~~~~

Refer to the instructions in the *vcbuild* directory in the source distribution.
