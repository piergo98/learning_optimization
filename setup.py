"""
Setup script for learning-optimization package.

This is maintained for backwards compatibility with older pip versions.
Modern installations should use pyproject.toml.
"""
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="learning-optimization",
        # Discover and install the actual package modules present in this repo.
        # The project places its code under the `src/` package directory.
        packages=find_packages(include=["src", "src.*", "data", "data.*"]),
        package_data={
            "src": ["*.md", "*.txt"],
            "data": ["*.mat", "*.csv"],
        },
    )
