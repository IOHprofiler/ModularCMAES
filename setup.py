import os
from glob import glob
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = "0.0.2.8.4"

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
        'scipy',
        'ioh'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
