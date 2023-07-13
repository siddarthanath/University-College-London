"""
This file setups the installation phase of QAFNet.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library 
from setuptools import setup

# Third Party

# Private Party

# -------------------------------------------------------------------------------------------------------------------- #

# Install QAFNet
exec(open("qafnet/version.py").read())
# Read the description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
# Execute the setup phase for the package
setup(
    name="qafnet",
    version=__version__,
    description="QAFNet - Interactive Online Learning through Questions, Answers and Feedback Network.",
    author="Siddartha Nath",
    author_email="ucabsn4@ucl.ac.uk",
    license="MIT",
    packages=["qafnet"],
    install_requires=[
        "numpy",
        "langchain",
        "openai",
        "faiss-cpu",
        "scipy",
        "pandas",
        "tiktoken",
    ],
    extras_require={"gpr": ["scikit-learn", "torch", "botorch", "gpytorch"]},
    test_suite="tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
