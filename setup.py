from setuptools import setup, find_packages
from os import path

__author__ = "NMA2022DL_TrashDetect_Team"
__version__ = "0.1.1"

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="trashdetect_engine",
    version=__version__,
    description="trashdetect_engine",
    packages=find_packages(),
    # package_data={'mipkit': ['faces/*']},
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    author=__author__,
    license="MIT",
    zip_safe=True,
    install_requires=requirements,
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ),
)
