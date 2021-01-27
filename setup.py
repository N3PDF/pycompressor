"""
pyCompressor is q python package for Monte Carlo Parton Distribution Functions
(PDFs) compression.

This program has been developed within the N3PDF group. (n3pdf.mi.infn.it/)

Authors: - Stefano Carrazza
         - Juan E. Cruz-Martinez
         - Tanjona R. Rabemananjara

License: GPL-3.0, 2020
"""


import pathlib
from setuptools import setup
from setuptools import find_packages

PACKAGE = "pycompressor"
THIS_DIR = pathlib.Path(__file__).parent
LONG_DESCRIPTION = (THIS_DIR / "README.md").read_text()
REQUIREMENTS = (THIS_DIR / "requirements.txt").read_text()

try:
    import lhapdf
except ImportError:
    print(f"Note: {PACKAGE} requires the installation of LHAPDF")

try:
    import validphys
except ImportError:
    print(f"Note: {PACKAGE} requires the installation of VALIDPHYS")


setup(
    name=PACKAGE,
    version='1.0.0-dev',
    description="PDF Compression",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="N3PDF",
    author_email="tanjona.rabemananjara@mi.infn.it",
    license="GPL 3.0",
    url="https://github.com/N3PDF/pyCompressor",
    zip_safe=False,
    project_urls={
        "Documentation": "https://n3pdf.github.io/pycompressor/",
        "Source": "https://github.com/N3PDF/pyCompressor"
    },
    entry_points={
        "console_scripts": [
            "pycomp = pycompressor.scripts.main:main",
            "validate = pycompressor.scripts.validate:main",
            "cmp-fids = pycompressor.scripts.fids:main",
            "cmp-dist = pycompressor.scripts.distributions:main",
            "cmp-corr = pycompressor.scripts.correlations:main",
            "get-grid = pycompressor.scripts.compressed_set:main"
        ]
    },
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=REQUIREMENTS,
    classifiers=[
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    setup_requires=["wheel"],
    python_requires='>=3.6'
)
