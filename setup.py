import os
from glob import glob

import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = "auto"
gh_ref = os.environ.get("GITHUB_REF")
if gh_ref:
    *_, tag = gh_ref.split("/")
    __version__ = tag.replace("v", "")

ext = Pybind11Extension(
    "modcma.cpp", 
    glob("src/*cpp"), 
    include_dirs=["src"],
    cxx_std=17
)

setuptools.setup(
    name='modcma',
    version=__version__,
    author="Jacob de Nobel",
    author_email="jacobdenobel@gmail.com",
    description="Package Containing Modular CMA-ES optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'numba',
        'scipy',
        'iohexperimenter'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
