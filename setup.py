"""Setup script."""
# pylint: disable=protected-access

import os
import platform
from glob import glob

import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open("README.md", "r", encoding="Latin-1") as fh:
    long_description = fh.read()

__version__ = "1.0.10"

ext = Pybind11Extension(
    "modcma.c_maes.cmaescpp",
    [x for x in glob("src/*cpp") if "main" not in x],
    include_dirs=["include", "external"],
    cxx_std=17,
)
if platform.system() in ("Linux", "Darwin"):
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
    flags = ["-O3", "-fno-math-errno", ] #"-fopenmp"
    if platform.system() == "Darwin":
        flags.append("-mmacosx-version-min=10.15")
    else:
        flags.append("-march=native")

    ext._add_cflags(flags)
    ext._add_ldflags(flags)
else:
    ext._add_cflags(["/O2"])


setuptools.setup(
    name="modcma",
    version=__version__,
    author="Jacob de Nobel",
    author_email="jacobdenobel@gmail.com",
    description="Package Containing Modular CMA-ES optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    cmdclass={"build_ext": build_ext},
    ext_modules=[ext],
    install_requires=[
        "numpy", 
        "scipy", 
        "ioh>=0.3.12,!=0.3.15"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
