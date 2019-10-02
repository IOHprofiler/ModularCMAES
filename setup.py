import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ccmaes',
    version='0.1.1',
    author="Jacob de Nobel",
    author_email="jacobdenobel@gmail.com",
    description="Package Containing Configurable version CMA ES optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/jacobdenobel/ccmaes/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
)
