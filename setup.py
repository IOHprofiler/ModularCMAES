"""Setup script."""
# pylint: disable=protected-access

import os
import platform
from glob import glob

import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open("README.md", "r", encoding="Latin-1") as fh:
    long_description = fh.read()

__version__ = "1.1.0"

if platform.system() in ("Linux", "Darwin"):
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
    c_flags = [
        "-O3", 
        "-fno-math-errno",
        "-funroll-loops", 
        "-ftree-vectorize",
    ]
    l_flags = [
        "-flto",
    ]
    if platform.system() == "Darwin":
        c_flags.append("-mmacosx-version-min=10.15")
    else:
        c_flags.extend([
            "-march=native",
            "-mtune=native",
        ])
else:
    c_flags = ["/O2"]
    l_flags = []

ext = Pybind11Extension(
    "modcma.c_maes.cmaescpp",
    [x for x in glob("src/*cpp") if "main" not in x],
    include_dirs=["include", "external"],
    cxx_std=17,
    extra_link_args=l_flags,
    extra_compile_args=c_flags
)

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
    python_requires=">=3.8",
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
