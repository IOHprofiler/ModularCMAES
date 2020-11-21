import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

    
__version__ = "auto"

# CI will change the above to match the github tag,
# if you run this manually, change __version__. 
assert __version__ != "auto"

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
        'ghalton',
        'sobol_seq',
        'iohexperimenter'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
