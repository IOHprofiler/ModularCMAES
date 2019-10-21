import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ccmaes',
    version='1.0.4',
    author="Jacob de Nobel",
    author_email="jacobdenobel@gmail.com",
    description="Package Containing Configurable version CMA ES optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy', 
        'scipy', 
        'sphinx', 
        'ghalton', 
        'sobol_seq'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
)
