"""
pyCompressor: Python implementation of compressor presented
in this paper https://arxiv.org/pdf/1504.06469 and implemented
here https://github.com/scarrazza/compressor.

This program has been developed within the framework of the
N3PDF group n3pdf.mi.infn.it/

Authors: - Tanjona R. Rabemananjara
         - Juan Cruz Martinez
         - Stefano Carrazza

License: GPL-3.0
"""


import os
import re
import sys
from setuptools import setup
from setuptools import find_packages

PACKAGE = "pycompressor"

requirements = [
        "cma",
        "tqdm",
        "scipy",
        "numpy",
        "numba",
        "matplotlib",
        "recommonmark",
        "sphinx_rtd_theme",
        'sphinxcontrib-bibtex'
    ]

# Check python version
if sys.version_info < (3, 6):
    print("cyclejet requires Python 3.6 or later", file=sys.stderr)
    sys.exit(1)

# Check if LHAPDF is installed
try:
    import lhapdf
except ImportError:
    print(f"Note: {PACKAGE} requires the installation of LHAPDF")

# Read through Readme
try:
    with open("README.md") as f:
        long_desc = f.read()
except IOError:
    print("Read me file not found")


# Get package version
def get_version():
    """
    Gets the version from the package's __init__ file
    if there is some problem, let it happily fail
    """
    VERSIONFILE = os.path.join("src", PACKAGE, "__init__.py")
    initfile_lines = open(VERSIONFILE, "rt").readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)


setup(
    name=PACKAGE,
    version=get_version(),
    description="PDF set compressor",
    author="Tanjona R. Rabemananjara, Juan Cruz Martinez, Stefano Carrazza",
    author_email="tanjona.rabemananjara@mi.infn.it",
    url="https://github.com/N3PDF/pyCompressor",
    long_description=long_desc,
    install_requires=requirements,
    entry_points={"console_scripts": ["pycompressor = pycompressor.run:main",]},
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    classifiers=[
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
