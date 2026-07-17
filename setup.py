"""Compatibility shim for editable installs with older pip versions."""

from setuptools import find_packages, setup


setup(
    name="hcorap-research",
    version="0.1.0",
    description="Reproducible multi-objective MaxSAT methods for HCORAP",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    package_dir={"": "src/proposed"},
    packages=find_packages("src/proposed"),
    install_requires=["python-sat>=1.8.dev20"],
    extras_require={"cpsat": ["ortools>=9.10"], "test": ["pytest>=8"]},
    entry_points={"console_scripts": ["hcorap=hcorap.cli:main"]},
)
