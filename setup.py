import itertools

from setuptools import setup, find_packages

# load requirements from requirements.txt
reqs = open("requirements.txt").read().splitlines()

setup(
    name="llm_food_taxonomy",
    version="0.1.0",
    description="",
    packages=find_packages(),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Pascal Wullschleger",
    author_email="pascal.wullschleger@hslu.ch",
    install_requires=reqs,
)
