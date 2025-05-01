# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

setup(
    name="gsba603_replication",
    version="0.1.0",
    description="Replication of Kinna, Samphantharak, Townsend and Vera-Cossio (2024).",
    long_description=readme,
    author="Mario Morales",
    author_email="mario@moralesalfaro.cl",
    url="https://github.com/marioles/methods",
    packages=find_packages(exclude=("tests", "docs"))
)
