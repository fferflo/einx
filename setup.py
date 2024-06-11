#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="einx",
    version="0.3.0",
    python_requires=">=3.8",
    description="Universal Tensor Operations in Einstein-Inspired Notation for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Florian Fervers",
    author_email="florian.fervers@gmail.com",
    url="https://github.com/fferflo/einx",
    packages=find_packages(),
    license="MIT",
    include_package_data=True,
    install_requires=[
        "numpy",
        "sympy",
        "frozendict",
    ],
    extras_require={
        "torch": ["torch>=2"],
        "keras": ["keras>=3"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
