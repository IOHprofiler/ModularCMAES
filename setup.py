"""Setup script."""
# pylint: disable=protected-access

import os
import platform
from glob import glob

import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open("README.md", "r", encoding="Latin-1") as fh:
    long_description = fh.read()

__version__ = "1.0.2"

ext = Pybind11Extension(
    "modcma.c_maes.cmaescpp",
    [x for x in glob("src/*cpp") if "main" not in x],
    include_dirs=["include", "external"],
    cxx_std=17,
)
if platform.system() in ("Linux", "Darwin"):
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
    flags = "-O3", "-fno-math-errno", "-march=native", #"-fopenmp"
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
    install_requires=["numpy", "scipy", "ioh"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
